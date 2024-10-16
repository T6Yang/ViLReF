## ViLReF:  An Expert Knowledge Enabled Vision-Language Retinal Foundation Model

| [Paper](https://arxiv.org/abs/2408.10894) |

## Download model checkpoints
| Model | Vision Backbone  | Text Backbone |
|-----------|-----------|------------|
| [ViLReF_ViT](https://drive.google.com/file/d/13YY2Qto4Xzx-gcOJB1kLdp1pqfjZEnxA/view?usp=drive_link) | ViT-b/16 | RoBERTa-wwm-ext-base-chinese |
| [ViLReF_RN50](https://drive.google.com/file/d/1xNCNJl_XWsXCgUiMN9O7xaxQ5hop8H2N/view?usp=drive_link) | ResNet50 | RoBERTa-wwm-ext-base-chinese |

## Requirements
```
asposestorage==1.0.2
lmdb==1.3.0
numpy==1.24.4
onnx==1.16.1
onnxmltools==1.12.0
onnxruntime==1.18.1
pandas==1.3.2
Pillow==10.4.0
scikit_learn==1.3.2
six==1.16.0
tensorrt==10.2.0.post1
timm==0.9.2
torch==1.13.1
torchvision==0.14.1
tqdm==4.64.0
```

## Usage
#### For training a new model:
```bash
bash train.sh
```
#### For loading the model to test:
```bash
bash load_model.sh
```

## Citation
```
@misc{yang2024vilrefchinesevisionlanguageretinal,
      title={ViLReF: An Expert Knowledge Enabled Vision-Language Retinal Foundation Model}, 
      author={Shengzhu Yang and Jiawei Du and Jia Guo and Weihang Zhang and Hanruo Liu and Huiqi Li and Ningli Wang},
      year={2024},
      eprint={2408.10894},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.10894}, 
}
```
