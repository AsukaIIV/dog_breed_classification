```
系统版本： Windows 11 26H1
python版本：python-3.13.12-amd64
cuda版本：cuda_13.0.0_windows
cudnn版本：cudnn_9.13.1_windows
nvidia显卡驱动版本：581.42

处理器：AMD Ryzen 7 5800X 8-Core Processor 
内存：DDR4 3600Mhz 16GBx2
显卡：Nvidia RTX 5060 8G
```
### 安装环境
按照如下顺序安装
- [ ] 确保nvidia显卡驱动版本在581.42以上
- [ ] 安装python-3.13.12-amd64（勾选添加到PATH）
- [ ] 安装cuda_13.0.0_windows
- [ ] 安装cudnn_9.13.1_windows
#### 安装使用cuda13.0.0的pytorch
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130 
```
#### 验证 pytorch 已安装
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))" ```
```
#### 安装其余依赖
```bash
pip install timm tensorboard scikit-learn seaborn matplotlib tqdm opencv-python scipy pillow
```
#### 测试 pytorch 是否能调用GPU
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

```bash 
True
NVIDIA GeForce RTX 5060
//理想的输出
```

### 进行训练
#### 运行本体
切换到 /dog_breed_classification 目录，控制台输入如下命令即可运行
```bash
python train.py --data ./data/stanford_dogs
```
会保存trainning_log.csv

### 评估
#### 评估stage2的模型
在/eval_results/即可查看训练结果
```bash
python evaluate.py --data ./data/stanford_dogs --ckpt checkpoints/best_stage2.pth
```
#### 评估stage3的模型
在/eval_results/即可查看训练结果
```bash
python evaluate.py --data ./data/stanford_dogs --ckpt checkpoints/best_stage3.pth
```

#### 在linux下的虚拟环境
```bash
//激活虚拟环境
source ~/下载/dog_breed_classification\(1\)/dog_breed_classification/venv/bin/activate
```

#### 对比模型并输出图
```bash
python compare_models.py --data ./data/stanford_dogs --ckpt_base checkpoints/best_baseline_stage3.pth --ckpt_imp checkpoints/best_stage3.pth --save_dir ./figures
```

#### 识别狗
```bash
python predict.py --img 你的狗图片.jpg --model checkpoints/best_stage3.pth --data ./data/stanford_dogs

python predict.py --img_dir ./test_images --model checkpoints/best_stage3.pth --data ./data/stanford_dogs --max_imgs 6
```
