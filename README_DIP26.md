# ERRNet DIP26 Guide

## 1. Clone and Download

### 1.1 Clone the repository

```bash
git clone https://github.com/innerway-xq/ERRNet
cd ERRNet
git checkout dip26
```

### 1.2 Setup the environment

```bash
conda create -n errnet python=3.10 -y
conda activate errnet
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -U pip wheel "setuptools<82"
pip install visdom==0.2.4 --no-build-isolation
```

Notes:
- Install the correct `torch`/`torchvision` version for your machine. Use a CUDA build if you have a GPU, and a CPU build otherwise. (Refer to https://pytorch.org/get-started/previous-versions/)

### 1.3 Download files

Download through [BaiduYun](https://pan.baidu.com/s/1MWb4eT18ySjogKVlcfPozg?pwd=egv2) or [GoogleDrive](https://drive.google.com/drive/folders/1_tN6JDlAmKZTgaqniQep1YJXmbFwGav7?usp=drive_link), then unzip and place files in ERRNet like this: 
```text
ERRNet/
  checkpoints/
    errnet/
      errnet_060_00463920.pt
  datasets/
    raw_data/
      VOCdevkit/
      CEILNet/
      real89/
      robustsirr_test_dataset/
      Dataset/
```

## 2. Prepare the Training and Testing Data


Run:

```bash
python datasets/prepare_test_data.py
python datasets/prepare_train_data.py
```


## 3. Testing

The testing script is `test_errnet.py`.

### 3.1 Benchmark testing

Supported benchmark names:

- `ceilnet_table2`
- `real20`
- `postcard`
- `objects`
- `wild`
- `sir2_withgt`

```bash
# gpu
python test_errnet.py --name errnet --dataset real20 -r --icnn_path checkpoints_att/errnet/errnet_050_00096650.pt --hyper
# cpu
python test_errnet.py --name errnet_cpu --dataset [dataset] -r --gpu_ids -1 --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
```


### 3.2 Test on your own images

If you only want to run the model on your own reflection images, ground truth is not required. Put your images in any folder, for example:

```text
datasets/raw_data/my_test_images/
  img1.jpg
  img2.jpg
```

Run:
```bash
python test_errnet.py --name errnet --dataset custom --input_dir /data/zhuchenyao/dip/ERRNet/datasets/raw_data/your_5_photos_small/input -r --icnn_path /data/zhuchenyao/dip/ERRNet/checkpoints/errnet/errnet_060_00231960.pt --hyper
```

Each image will have its own subfolder. You will usually see:

- `m_input.png`: the input image
- `errnet.png` or `errnet_cpu.png`: the model output

## 4. Training

There are two training stages:

- Aligned-data training: `train_errnet.py`
- Unaligned-data finetuning: `train_errnet_unaligned.py`

### 4.1 Train the aligned baseline

```bash
# gpu
python train_errnet.py --name errnet --hyper

python train_errnet.py --name errnet --hyper --gpu_ids 0 --batchSize 4 --checkpoints_dir ./checkpoints_att_plus --resume
# cpu
python train_errnet.py --name errnet_cpu --hyper --gpu_ids -1
```
### 4.2 Finetune on unaligned data


#### GPU version

```bash
# gpu
python train_errnet_unaligned.py --name errnet_unaligned_ft --hyper -r --icnn_path /data/zhuchenyao/dip/ERRNet/checkpoints_att_plus/errnet_unaligned_ft/errnet_085_00165565.pt --unaligned_loss vgg --gpu_ids 2 --batchSize 4   --checkpoints_dir ./checkpoints_att_plus
# cpu
python train_errnet_unaligned.py --name errnet_unaligned_ft_cpu --hyper -r --gpu_ids -1 --icnn_path checkpoints/errnet/errnet_060_00463920.pt --unaligned_loss vgg
```

python test_errnet.py --name errnet --dataset real20 -r --gpu_ids 1  --icnn_path /data/zhuchenyao/dip/ERRNet/checkpoints_att_noinp/errnet/errnet_055_00106315.pt --hyper


## 5. Baseline Result

> checkpoints/errnet/errnet_060_00463920.pt

| Dataset         |  PSNR ↑ | SSIM ↑ | NCC ↑  | LMSE ↓ |
| ---             |   ---   |   ---  |   ---  |   ---  |
| CEILNet Table 2 | 27.8771 | 0.9407 | 0.9808 | 0.0048 |
| real20          | 23.5531 | 0.8285 | 0.8877 | 0.0201 |
| objects         | 24.8530 | 0.8980 | 0.9817 | 0.0029 |
| postcard        | 22.0700 | 0.8773 | 0.9463 | 0.0044 |
| wild            | 25.1780 | 0.8861 | 0.9359 | 0.0083 |

80, b4
| PSNR: 26.3778 | SSIM: 0.9272 | NCC: 0.9735 | LMSE: 0.0057 | 
60, b4
| PSNR: 27.2806 | SSIM: 0.9370 | NCC: 0.9782 | LMSE: 0.0050 | 
60, b2
LMSE: 0.0045 | NCC: 0.9796 | PSNR: 27.8031 | SSIM: 0.9411 | 

ATT 46:
| PSNR: 28.7080 | SSIM: 0.9501 | NCC: 0.9830 | LMSE: 0.0039 |   LMSE: 0.0046 | NCC: 0.9801 | PSNR: 27.7647 | SSIM: 0.9437 | 
| PSNR: 23.5233 | SSIM: 0.8305 | NCC: 0.8904 | LMSE: 0.0196 |   LMSE: 0.0217 | NCC: 0.8849 | PSNR: 22.9430 | SSIM: 0.8231 | 
| PSNR: 24.1823 | SSIM: 0.8880 | NCC: 0.9810 | LMSE: 0.0034 |   LMSE: 0.0035 | NCC: 0.9816 | PSNR: 24.4285 | SSIM: 0.8914 |
| PSNR: 22.1604 | SSIM: 0.8813 | NCC: 0.9421 | LMSE: 0.0047 |   LMSE: 0.0118 | NCC: 0.8467 | PSNR: 19.7415 | SSIM: 0.8263 | 
| PSNR: 25.1537 | SSIM: 0.9034 | NCC: 0.9453 | LMSE: 0.0055 |   LMSE: 0.0048 | NCC: 0.9521 | PSNR: 25.0877 | SSIM: 0.9081 | 

ATTPLUS 65
LMSE: 0.0039 | NCC: 0.9824 | PSNR: 28.8965 | SSIM: 0.9472 |    LMSE: 0.0045 | NCC: 0.9778 | PSNR: 27.0444 | SSIM: 0.9312 |   
LMSE: 0.0187 | NCC: 0.8987 | PSNR: 23.4023 | SSIM: 0.8255 |    LMSE: 0.0198 | NCC: 0.8955 | PSNR: 22.1274 | SSIM: 0.7944 | 
LMSE: 0.0035 | NCC: 0.9831 | PSNR: 24.3270 | SSIM: 0.8899 |    LMSE: 0.0032 | NCC: 0.9846 | PSNR: 25.2879 | SSIM: 0.8996 | 
LMSE: 0.0047 | NCC: 0.9428 | PSNR: 21.9909 | SSIM: 0.8820 |    LMSE: 0.0057 | NCC: 0.9350 | PSNR: 22.0771 | SSIM: 0.8667 |
LMSE: 0.0044 | NCC: 0.9522 | PSNR: 25.6564 | SSIM: 0.9097 |    LMSE: 0.0049 | NCC: 0.9525 | PSNR: 24.8144 | SSIM: 0.9003 | 

LMSE: 0.0045 | NCC: 0.9783 | PSNR: 27.3639 | SSIM: 0.9331 |
LMSE: 0.0199 | NCC: 0.8941 | PSNR: 22.3884 | SSIM: 0.8020 |
LMSE: 0.0033 | NCC: 0.9844 | PSNR: 24.9445 | SSIM: 0.8970 | 
LMSE: 0.0057 | NCC: 0.9304 | PSNR: 21.8459 | SSIM: 0.8681 | 
LMSE: 0.0047 | NCC: 0.9523 | PSNR: 24.8462 | SSIM: 0.9052 |

ATTPLUS 75
LMSE: 0.0039 | NCC: 0.9823 | PSNR: 28.6804 | SSIM: 0.9441 | 
LMSE: 0.0179 | NCC: 0.8996 | PSNR: 23.2889 | SSIM: 0.8201 | 
LMSE: 0.0035 | NCC: 0.9835 | PSNR: 24.3648 | SSIM: 0.8901 | 
LMSE: 0.0046 | NCC: 0.9470 | PSNR: 22.0353 | SSIM: 0.8838 | 
LMSE: 0.0044 | NCC: 0.9526 | PSNR: 25.4260 | SSIM: 0.9097 | 

ATTPLUS 80
LMSE: 0.0039 | NCC: 0.9824 | PSNR: 28.7643 | SSIM: 0.9459 | 
LMSE: 0.0191 | NCC: 0.8957 | PSNR: 22.9729 | SSIM: 0.8147 | 
LMSE: 0.0036 | NCC: 0.9832 | PSNR: 24.2506 | SSIM: 0.8888 | 
LMSE: 0.0047 | NCC: 0.9453 | PSNR: 21.8313 | SSIM: 0.8833 | 
LMSE: 0.0044 | NCC: 0.9523 | PSNR: 25.4238 | SSIM: 0.9081 |


ATTPLUS 70
LMSE: 0.0039 | NCC: 0.9819 | PSNR: 28.7163 | SSIM: 0.9458 | 
LMSE: 0.0184 | NCC: 0.8980 | PSNR: 23.2466 | SSIM: 0.8239 | 
LMSE: 0.0035 | NCC: 0.9829 | PSNR: 24.4459 | SSIM: 0.8919 | 
LMSE: 0.0048 | NCC: 0.9410 | PSNR: 21.7179 | SSIM: 0.8787 | 
LMSE: 0.0047 | NCC: 0.9500 | PSNR: 25.3933 | SSIM: 0.9032 | 


ATTPLUS 85
LMSE: 0.0038 | NCC: 0.9827 | PSNR: 28.8723 | SSIM: 0.9464 | 
LMSE: 0.0190 | NCC: 0.8956 | PSNR: 22.9979 | SSIM: 0.8177 | 
LMSE: 0.0037 | NCC: 0.9827 | PSNR: 24.1514 | SSIM: 0.8872 | 
LMSE: 0.0048 | NCC: 0.9407 | PSNR: 21.4863 | SSIM: 0.8794 | 
LMSE: 0.0045 | NCC: 0.9518 | PSNR: 25.3562 | SSIM: 0.9088 |

for ds in ceilnet_table2 real20 objects postcard wild; do echo "=============== $ds ==============="; python test_errnet.py --name errnet --dataset $ds -r --gpu_ids 0 --icnn_path /data/zhuchenyao/dip/ERRNet/checkpoints_att_plus/errnet_unaligned_ft/errnet_latest.pt --hyper; done

python test_errnet.py --name errnet --dataset real20 -r --icnn_path /data/zhuchenyao/dip/ERRNet/checkpoints_att_noinp/errnet/errnet_050_00096650.pt --hyper
- `ceilnet_table2`
- `real20`
- `objects`
- `postcard`
- `wild`
- `sir2_withgt`

batchsize 8 
Epoch: 0
 [=========================== 967/967 ============================>]  Step: 347ms | Tot: 5m36s | IPixel: 0.0266 | VGG: 3.8837 |

batchsize 4
Epoch: 0
 [=========================== 1933/1933 ==========================>]  Step: 167ms | Tot: 5m34s | IPixel: 0.0229 | VGG: 3.6754 |    

batchsize 2
Epoch: 0
 [=========================== 3866/3866 ==========================>]  Step: 97ms | Tot: 6m53s | IPixel: 0.0215 | VGG: 3.5980 |  


 [====================== 3866/3866 ===========================>]  Step: 135ms | Tot: 9m8s | D: 0.6787 | G: 0.7550 | IPixel: 0.0087 | VGG: 1.8254 |
saving the latest model at the end of epoch 21, iters 81186
Time Taken: 557 sec 
 
 [========================== 3866/3866 =======================>]  Step: 153ms | Tot: 9m47s | D: 0.6017 | G: 1.0442 | IPixel: 0.0079 | VGG: 1.6039 |                      
saving the latest model at the end of epoch 31, iters 119846
Time Taken: 598 sec

Epoch: 52
 [============================ 3866/3866 =========================>]  Step: 140ms | Tot: 9m37s | D: 0.5927 | G: 1.0949 | IPixel: 0.0081 | VGG: 1.3257 |                      
saving the latest model at the end of epoch 53, iters 204898
Time Taken: 586 sec


batchsize 1
 Epoch: 0
 [=========================== 7732/7732 ==========================>]  Step: 80ms | Tot: 10m25s | IPixel: 0.0226 | VGG: 3.6028 |   