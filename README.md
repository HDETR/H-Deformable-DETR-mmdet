# H-Deformable-DETR for MMDet

This is the official implementation of the paper "[DETRs with Hybrid Matching](https://arxiv.org/abs/2207.13080)". 

Authors: Ding Jia, Yuhui Yuan, Haodi He, Xiaopei Wu, Haojun Yu, Weihong Lin, Lei Sun, Chao Zhang, Han Hu

## Citing H-Deformable-DETR
If you find H-Deformable-DETR useful in your research, please consider citing:
```bibtex
@article{jia2022detrs,
  title={DETRs with Hybrid Matching},
  author={Jia, Ding and Yuan, Yuhui and He, Haodi and Wu, Xiaopei and Yu, Haojun and Lin, Weihong and Sun, Lei and Zhang, Chao and Hu, Han},
  journal={arXiv preprint arXiv:2207.13080},
  year={2022}
}
```
## Model ZOO

We provide a set of baseline results and trained models available for download:

### Models with the ResNet-50 backbone
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">query</th>
<th valign="bottom">epochs</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/deformable_detr/deformable_detr_twostage_refine_r50_dim2048_16x2_12e_coco.py">Deformable-DETR</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">43.7</td>
<td align="center"><a href="">model</a></td>
 <tr><td align="left"><a href="configs/deformable_detr/deformable_detr_twostage_refine_r50_dim2048_16x2_36e_coco.py">Deformable-DETR</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">46.8</td>
<td align="center"><a href="">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/deformable_detr/deformable_detr_twostage_refine_r50_dp0_mqs_lft_dim2048_16x2_12e_coco.py">Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">47.0</td>
<td align="center"><a href="">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/deformable_detr/deformable_detr_twostage_refine_r50_dp0_mqs_lft_dim2048_16x2_36e_coco.py">Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">49.0</td>
<td align="center"><a href="">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/h-deformable-detr/h_deformable_detr_twostage_refine_r50_group6_t1500_dp0_mqs_lft_dim2048_16x2_12e_coco.py">H-Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">48.7</td>
<td align="center"><a href="">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/h-deformable-detr/h_deformable_detr_twostage_refine_r50_group6_t1500_dp0_mqs_lft_dim2048_16x2_36e_coco.py">H-Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">50.0</td>
<td align="center"><a href="">model</a></td>
</tr>
</tbody></table>

### Models with Swin Transformer backbones

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">query</th>
<th valign="bottom">epochs</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="configs/deformable_detr/deformable_detr_twostage_refine_swin_tiny_dim2048_16x2_12e_coco.py">Deformable-DETR</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">45.3</td>
<td align="center"><a href="">model</a></td>
 <tr><td align="left"><a href="configs/deformable_detr/deformable_detr_twostage_refine_swin_tiny_dim2048_16x2_36e_coco.py">Deformable-DETR</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">49.0</td>
<td align="center"><a href="">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/deformable_detr/deformable_detr_twostage_refine_swin_tiny_dp0_mqs_lft_dim2048_16x2_12e_coco.py">Deformable-DETR + tricks</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">49.3</td>
<td align="center"><a href="">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/deformable_detr/deformable_detr_twostage_refine_swin_tiny_dp0_mqs_lft_dim2048_16x2_36e_coco.py">Deformable-DETR + tricks</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">51.8</td>
<td align="center"><a href="">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/h-deformable-detr/h_deformable_detr_twostage_refine_swin_tiny_group6_t1500_dp0_mqs_lft_dim2048_16x2_12e_coco.py">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">50.6</td>
<td align="center"><a href="">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/h-deformable-detr/h_deformable_detr_twostage_refine_swin_tiny_group6_t1500_dp0_mqs_lft_dim2048_16x2_36e_coco.py">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">53.2</td>
<td align="center"><a href="">model</a></td>
</tr>
</tbody></table>

## Installation
We test our models under ```python=3.7.10,pytorch=1.10.1,cuda=10.2```. Other versions might be available as well.

1. Clone this repo
```sh
git https://github.com/HDETR/H-Deformable-DETR-mmdet.git
cd H-Deformable-DETR-mmdet
```

2. Install Pytorch and torchvision

Follow the instruction on https://pytorch.org/get-started/locally/.
```sh
# an example:
conda install -c pytorch pytorch torchvision
```

3. Install other needed packages
```sh
pip install -r requirements.txt
pip install openmim
mim install mmcv-full
pip install -e .
```

## Data

Please download [COCO 2017](https://cocodataset.org/) dataset and organize them as following:
```
mmdetection
├── data
│   ├── coco
│   │   ├── train2017
│   │   ├── val2017
│   │   └── annotations
|   |        ├── instances_train2017.json
|   |        └── instances_val2017.json
```
## Run
### To train a model using 8 cards

```Bash
GPUS_PER_NODE=8  ./tools/dist_train.sh \
    <config path> \
    8
```

To train/eval a model with the swin transformer backbone, you need to download the backbone from the [offical repo](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models) frist and specify argument`checkpoint` like [our config](./configs/h-deformable-detr/h_deformable_detr_twostage_refine_swin_tiny_group6_t1500_dp0_mqs_lft_dim2048_16x2_12e_coco.py).

### To eval a model using 8 cards

```Bash
GPUS_PER_NODE=8 tools/dist_test.sh \
    <config path> \
    <checkpoint path> \
    8 --eval bbox
```

## Modified files compared to vanilla Deformable DETR

* configs/deformable_detr: add baseline configs
* configs/h-deformable-detr: add h-deformable-detr configs
* mmdet/models/utils/transformer.py: enable tricks and decoder_self_attn_mask
* mmdet/models/dense_heads/hybrid_branch_deformable_detr_head.py: enable hybrid branch strategy and tricks
* mmdet/models/dense_heads/deformable_detr_head.py: enable tricks
