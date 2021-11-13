# TinyAction
Baseline code for [[TinyAction Challenge]](https://www.crcv.ucf.edu/tiny-actions-challenge-cvpr2021/)  [[Paper]](https://arxiv.org/pdf/2107.11494.pdf)  [[Dataset]](https://www.crcv.ucf.edu/tiny-actions-challenge-cvpr2021/data/TinyVIRAT-v2.zip)


## Getting started
### Setup
-Prerequisites: Python 3.6, NumPy, SciPy, OpenCV, PyTorch (>=1.6), torchvision

Download the dataset and extract the files. Once done, change the dataset path in `configuration.py` file accordingly.

(Optional) Download the pretrained weights for the model.

### Training
We provide training setup for I3D, R3D, R2+1D and wideresnet models. If you wish to use pretrained weights, change the path for `--pretrain_path`. The other options are related to each model (from their papers). Use the following command for each model:

#### I3D
```
python main.py --result_path results --sub_path i3d --model i3d --model_depth 18 --n_classes 26 --n_pretrain_classes 157 --pretrain_path weights/rgb_charades.pt --num_frames 32 --skip_frames 1 --sample_size 224 --learning_rate 0.001 --optimizer adam --batch_size 24 --n_threads 4 --checkpoint 5
```

#### R3D
```
python main.py --result_path results --sub_path r3d_18 --model resnet --model_depth 18 --n_classes 26 --n_pretrain_classes 700 --pretrain_path weights/r3d18_K_200ep.pth --num_frames 16 --skip_frames 1 --sample_size 112 --learning_rate 0.001 --optimizer adam --batch_size 96 --n_threads 4 --checkpoint 5 
```

#### R2+1D
```
python main.py --result_path results --sub_path r2p1d50 --model resnet2p1d --model_depth 50 --n_classes 26 --n_pretrain_classes 700 --pretrain_path weights/r2p1d50_K_200ep.pth --num_frames 16 --skip_frames 1 --sample_size 112 --learning_rate 0.001 --optimizer adam --batch_size 64 --n_threads 4 --checkpoint 5 
```

#### wideresnet 
```
python main.py --result_path results --sub_path wideresnet --model wideresnet --model_depth 50 --n_classes 26 --n_pretrain_classes 400 --pretrain_path weights/wideresnet-50-kinetics.pth --num_frames 16 --skip_frames 1 --sample_size 112 --learning_rate 0.001 --optimizer adam --batch_size 64 --n_threads 4 --checkpoint 5 --resnet_shortcut B
```

### Evaluation
To prepare the evaluation file for submission, use the following after changing `--pretrain_path` with your trained model:

#### I3D
```
python evaluate_variants.py --model i3d --model_depth 18 --n_classes 26 --n_pretrain_classes 26 --pretrain_path results/i3d/save.pth --num_frames 32 --skip_frames 1 --sample_size 224 --n_threads 4
```

#### R3D 
```
python evaluate.py --sub_path r3d_18 --model resnet --model_depth 18 --n_classes 26 --n_pretrain_classes 700 --pretrain_path results/r3d/save.pth --num_frames 16 --skip_frames 1 --sample_size 112 --n_threads 4
```

#### R2+1D
```
python evaluate_variants.py --model resnet2p1d --model_depth 50 --n_classes 26 --n_pretrain_classes 26 --pretrain_path results/r2p1d50/save.pth --num_frames 16 --skip_frames 1 --sample_size 112 --n_threads 4
```

#### wideresnet 
```
python evaluate_variants.py --model wideresnet --model_depth 50 --n_classes 26 --n_pretrain_classes 26 --pretrain_path results/wideresnet/save.pth --num_frames 16 --skip_frames 1 --sample_size 112 --n_threads 4 --resnet_shortcut B
```