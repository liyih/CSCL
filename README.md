<div align="center">
<!-- <h1>RCTrans</h1> -->
<h3> Unleashing the Potential of Consistency Learning for Detecting and Grounding Multi-Modal Media Manipulation</h3>
<h4>Yiheng Li, Yang Yang, Zichang Tan, Huan Liu, Weihua Chen, Xu Zhou and Zhen Lei<h4>
<h5>MAIS&CASIA, UCAS, Sangfor, BJTU and Alibaba<h5>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2506.05890)

## Introduction

This repository is an official implementation of CSCL.

## News
- [2025/6/9] Camera Ready version is released.
- [2025/6/9] Codes and weights are released.
- [2025/2/27] CSCL is accepted by CVPR 2025🎉🎉.

## Environment Setting
```
conda create -n CSCL python=3.8
conda activate CSCL
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r code/MultiModal-DeepFake-main/requirements.txt
```
## Data Preparation
Here are the pre-trained model:

  Download meter_clip16_224_roberta_pretrain.ckpt: [link](https://drive.usercontent.google.com/download?id=1x4qm2rlYKxpYF3F_xI5ZKurTtFnndq3l&export=download&authuser=0&confirm=t&uuid=9356ef04-1b7b-444c-80be-4bf21fab8bda&at=AIrpjvOLjj-J08OdrxQf_rCxV7Zp:1739190851890)
  
  Download ViT-B-16.pt: [link](https://drive.usercontent.google.com/download?id=1GL3kOw-lmbD5abJCaaktLrODqxMllpd6&export=download&authuser=0&confirm=t&uuid=5a286816-fa87-4fd0-a75d-825ec03966e4&at=AIrpjvMiXdIVW3BRne33Y_-pvh1D:1739190843518)
  
  Download roberta-base: [link](https://huggingface.co/FacebookAI/roberta-base/tree/main)

Download Datasets: [link](https://huggingface.co/datasets/rshaojimmy/DGM4)

The Folder structure:
```
./
├── code
│   └── MultiModal-Deepfake (this github repo)
│       ├── configs
│       │   └──...
│       ├── dataset
│       │   └──...
│       ├── models
│       │   └──...
│       ...
│       ├── roberta-base
│       ├── ViT-B-16.pt
│       └── meter_clip16_224_roberta_pretrain.ckpt
└── datasets
    └── DGM4
        ├── manipulation
        ├── origin
        └── metadata
```

Our pre-trained CSCL model: [link](https://drive.usercontent.google.com/download?id=1ZW4akTzcB9QjsS6FcX4zQ5l2YOjl7zNy&export=download&authuser=0&confirm=t&uuid=e8e37fa5-46fd-48bb-be4b-be765ca86059&at=AIrpjvM1Jjby7_AjinIBFS9d61TL:1739189602615) (96.34 AUC, 92.48 mAP, 84.07 IoUm, 76.62 F1) (We use train and val set for training and use test set for evaluation.)

Make a folder ./results/CSCL/ and put the pre-trained model in it.

## Inference

Evaluation
```
sh test.sh
```
Visualization
```
use visualize_res function in utils.py (refer to test.py for details).
```
Evaluation on text or image subset
```
refer to line 136 in test.py.
```
## Acknowledgements
We thank these great works and open-source codebases:
[DGM4](https://github.com/rshaojimmy/MultiModal-DeepFake?tab=readme-ov-file), [METER](https://github.com/zdou0830/METER),

## Citation
If you find our work is useful, please give this repo a star and cite our work as:
```bibtex
@inproceedings{li2025unleashing,
  title={Unleashing the Potential of Consistency Learning for Detecting and Grounding Multi-Modal Media Manipulation},
  author={Li, Yiheng and Yang, Yang and Tan, Zichang and Liu, Huan and Chen, Weihua and Zhou, Xu and Lei, Zhen},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={9242--9252},
  year={2025}
}
```
