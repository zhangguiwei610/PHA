

# [CVPR2023] PHA: Patch-wise High-frequency Augmentation for Transformer-based Person Re-identification [[pdf]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_PHA_Patch-Wise_High-Frequency_Augmentation_for_Transformer-Based_Person_Re-Identification_CVPR_2023_paper.pdf)

Official Code for the CVPR 2023 paper [PHA: Patch-wise High-frequency Augmentation for Transformer-based Person Re-identification].




## Requirements

### Installation

```bash
pip install -r requirements.txt
(we use 32G V100 for training and evaluation.)
```



### Prepare  ViT Pre-trained Models

You need to download the ImageNet pretrained transformer model : [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth),

## Training

We utilize 1  GPU for training.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/Cuhk03_labeled/vit_transreid_stride.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/Market/vit_transreid_stride.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/MSMT17/vit_transreid_stride.yml
```

## Citation

If you find this code useful for your research, please cite our paper

```
@InProceedings{Zhang_2023_CVPR,
    author    = {Guiwei Zhang, Yongfei Zhang, Tianyu Zhang, Bo Li1, Shiliang Pu},
    title     = {PHA: Patch-wise High-frequency Augmentation for Transformer-based Person Re-identification},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14133-14142}
}
```
