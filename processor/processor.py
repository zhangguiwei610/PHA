import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from PIL import  Image
from torchvision import transforms as trans
import  numpy
def dequantize(image, q_table):
    """[summary]
    TODO: Add discription
    Args:
        image ([type]): [description]
        q_table ([type]): [description]

    Returns:
        [type]: [description]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    q_table = q_table.to(device)
    dequantitize_img = image * q_table
    return dequantitize_img

def phi_diff(x, alpha):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
    s = 1/(1-alpha).to(device)
    k = torch.log(2/alpha -1).to(device)
    phi_x = torch.tanh((x - (torch.floor(x) + 0.5)) * k) * s
    x_ = (phi_x + 1)/2 + torch.floor(x)
    return x_
def quantize(image, q_table,alpha):
    """[summary]
    TODO: add disciption.

    Args:
        image ([type]): [description]
        q_table ([type]): [description]
    """
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    q_table = q_table.to(device)
    pre_img = image/(q_table)
    after_img = phi_diff(pre_img, alpha)
    # after_img = sgn(after_img)
    # after_img = torch.round(pre_img) + torch.empty_like(pre_img).uniform_(0.0, 1.0)
    # diff = after_img - pre_img
    # print("Max difference: ", torch.max(diff))
    # image = torch.round(image)
    # image = diff_round(image)
    # after_img = diff_round(pre_img)
    return after_img

from timm.data.random_erasing import RandomErasing


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
    # J为分解的层次数,wave表示使用的变换方法
    xfm = DWTForward(J=1, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='haar')
    transform = trans.Compose([
        trans.Resize([256, 128], interpolation=3),
        trans.ToTensor()
    ])


    transform_norm = trans.Compose([
        trans.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view,img_paths) in enumerate(train_loader):
            factor = min(epoch * 0.05, 0.5)
            topk_num=int(128*64*0.4)
            wavelet_list = []
            for per_path in img_paths:
                img_tmp = Image.open(per_path)
                img_tmp = transform(img_tmp)
                wavelet_list.append(img_tmp)
            wavelet_list = torch.stack(wavelet_list)  # bs,3,256,128
            x_dwt_l, x_dwt_h = xfm(wavelet_list)  # input bs,3,256,128  Yl bs,3,128,64  Yh[0] bs,c,3,128,64
            high_frequency = x_dwt_h[0].sum(dim=2)  # x_dwt_h[0] bs,c,3,128,64，output bs,c,128,64
            high_frequency = torch.norm(high_frequency, dim=1)  # bs,128,64

            cluster = high_frequency.flatten(1)  # bs,128*64
            _, topk = cluster.topk(topk_num, dim=1, largest=True, sorted=True)  # bs,256
            mask_map = torch.zeros_like(cluster)
            idx_batch = torch.arange(len(img_paths))[:, None].expand(-1, topk_num)
            mask_map[idx_batch, topk] = 255
            mask_map = mask_map.reshape(len(img_paths), 128, 64)  # bs,128,64
            mask_map_adv=mask_map
            mask_map = torch.nn.functional.interpolate(mask_map[None, :, :, :], size=(21, 10)).squeeze(0)  # bs,21,10

            # adv
            # Value for quantization range
            alpha_range = [0.1, 1e-20]
            alpha = torch.tensor(alpha_range[0]).cuda()

            q_ini_table = numpy.empty((32, 128, 64), dtype=numpy.float32)
            q_ini_table.fill(5)
            q_ini_table=numpy.where(mask_map_adv>0,q_ini_table,numpy.ones_like(q_ini_table))
            q_tables = {"y": torch.from_numpy(q_ini_table).cuda(),
                        "cb": torch.from_numpy(q_ini_table).cuda(),
                        "cr": torch.from_numpy(q_ini_table).cuda()}
            alpha = alpha.to('cuda')


            x_dwt_l_q = 255.0 * x_dwt_l.clone().detach().to('cuda')  # bs,c,h,w
            x_dwt_l_q = x_dwt_l_q.permute(0, 2, 3, 1)  # bs,h,w,c
            components = {'y': x_dwt_l_q[:, :, :, 0], 'cb': x_dwt_l_q[:, :, :, 1],
                          'cr': x_dwt_l_q[:, :, :, 2]}  # y,cb,cr bs,h,w
            # q_tables["y"].requires_grad = True
            # q_tables["cb"].requires_grad = True
            # q_tables["cr"].requires_grad = True
            upresults = {}
            for k in components.keys():
                comp = quantize(components[k], q_tables[k], alpha)  # output bs，784，8，8
                comp = dequantize(comp, q_tables[k])  # output bs，784，8，8
                upresults[k] = comp
            rgb_images = torch.cat(
                [upresults['y'].unsqueeze(3), upresults['cb'].unsqueeze(3), upresults['cr'].unsqueeze(3)], dim=3)
            rgb_images = rgb_images.permute(0, 3, 1, 2) / 255.  # bs,3,128,64

            img_adv = ifm((rgb_images.cpu(), x_dwt_h)).cuda()
            img_adv = transform_norm(img_adv)




            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat ,mask,mask_adv,loss_contra= model([img,img_adv], target, cam_label=target_cam, view_label=target_view ,mask=mask_map)

                loss = loss_fn(score, feat, target, target_cam)
                loss_mask=loss_fn(mask[0],mask[1],target,target_cam)*factor

                loss_mask_adv = loss_fn(mask_adv[0], mask_adv[1], target, target_cam) * factor


            scaler.scale(loss+loss_mask+loss_mask_adv+loss_contra).backward()


            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_mask :{:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg,loss_mask.item(), acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % 40 == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


