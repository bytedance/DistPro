# DistPro ECCV 2022

This is the official release of ["DistPro: Searching A Fast Knowledge Distillation Process via Meta Optimization"](https://arxiv.org/abs/2204.05547) ECCV 2022. 
![Alt text](distpro_overview.jpg?raw=true "Comparisons to other methods")

## Installation

```bash
pip install -r requirements.txt
```

## Usage
#### 1. Search distillation process on CIFAR datasets with sample epochs.

```bash
python train_with_distillation.py --paper_setting SETTING_CHOICE --epochs 40
```
#### 2. Retrain the data with full epochs 
```bash
python train_with_distillation.py --paper_setting SETTING_CHOICE --epochs 240 \
--alpha_normalization_style 333
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)


```BibTeX
@inproceedings{deng2022distpro,
  title={DistPro: Searching A Fast Knowledge Distillation Process via Meta Optimization},
  author={Xueqing Deng and Dawei Sun and Shawn Newsam and Peng Wang},
  journal={ECCV},
  year={2022}
}
```