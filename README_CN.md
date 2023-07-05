<br>
<div align="center">
<img src="https://user-images.githubusercontent.com/18592211/232830417-0b21a874-516e-4420-8984-4de414a35085.png" width="400px"></img>
<h2></h2>
<h3>Towards Any Structural Pruning<h3>
<img src="assets/intro.png" width="50%">
</div>
  
<p align="center">
  <a href="https://github.com/VainF/Torch-Pruning/actions"><img src="https://img.shields.io/badge/tests-passing-9c27b0.svg" alt="Test Status"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.8 %20%7C%201.12 %20%7C%202.0-673ab7.svg" alt="Tested PyTorch Versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-4caf50.svg" alt="License"></a>
  <a href="https://pepy.tech/project/Torch-Pruning"><img src="https://pepy.tech/badge/Torch-Pruning?color=2196f3" alt="Downloads"></a>
  <a href="https://github.com/VainF/Torch-Pruning/releases/latest"><img src="https://img.shields.io/badge/Latest%20Version-1.1.6-3f51b5.svg" alt="Latest Version"></a>
  <a href="https://colab.research.google.com/drive/1TRvELQDNj9PwM-EERWbF3IQOyxZeDepp?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
  <a href="https://arxiv.org/abs/2301.12900" target="_blank"><img src="https://img.shields.io/badge/arXiv-2301.12900-009688.svg" alt="arXiv"></a>
</p>


Torch-Pruning (TP)是一个通用的结构化网络剪枝框架，主要包括以下功能：
* **通用的结构化剪枝工具:** 支持 *[LLaMA](https://github.com/horseee/LLaMA-Pruning), [Vision Transformers](benchmarks/prunability), [Yolov7](benchmarks/prunability/readme.md#3-yolo-v7), [yolov8](benchmarks/prunability/readme.md#2-yolo-v8), FasterRCNN, SSD, KeypointRCNN, MaskRCNN, ResNe(X)t, ConvNext, DenseNet, ConvNext, RegNet, FCN, DeepLab* 等神经网络. 不同于[torch.nn.utils.prune](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)中利用掩码(Masking)实现的“模拟剪枝”, Torch-Pruning采用了一种名为DepGraph的非深度图算法, 能够“物理”地移除模型中的耦合参数和通道。
* **可复线的[性能基准线](benchmarks)和[可剪枝性基准线](benchmarks/prunability)**: 目前, Torch-Pruning已经覆盖了 **81/85=95.3%** 的Torchvision预训练模型(v0.13.1). 您可以访问[Colab Demo](https://colab.research.google.com/drive/1TRvELQDNj9PwM-EERWbF3IQOyxZeDepp?usp=sharing)来快速体验Torchvision预训练模型的剪枝。

更多技术细节请参考我们的论文： 

> [**DepGraph: Towards Any Structural Pruning**](https://arxiv.org/abs/2301.12900)   
> [Gongfan Fang](https://fangggf.github.io/), [Xinyin Ma](https://horseee.github.io/), [Mingli Song](https://person.zju.edu.cn/en/msong), [Michael Bi Mi](https://dblp.org/pid/317/0937.html), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)   

### Update:
* 2023.04.15 [Pruning and Post-training for YOLOv7 / YOLOv8](benchmarks/prunability)
* 2023.04.10 [Structural Pruning for LLaMA (pruning-only)](https://github.com/horseee/LLaMA-Pruning)
* 2023.04.21 加入技术分享群（微信群或Telegram Group）:
  * Wechat: <img width="100" alt="image" src="https://user-images.githubusercontent.com/18592211/233550290-169a2a3e-bbe2-4a21-a2a3-62c7ac2c6a1a.png" >
  * Telegram: https://t.me/+NwjbBDN2ao1lZjZl
  
如有任何框架、论文相关的问题, 请新建[discussion](https://github.com/VainF/Torch-Pruning/discussions)或者[issue](https://github.com/VainF/Torch-Pruning/issues). 非常乐意回复您的问题.
  
### **特性:**
- [x] 结构化(通道)剪枝: 支持[CNNs](benchmarks/prunability/torchvision_pruning.py#L19), [Transformers](benchmarks/prunability/torchvision_pruning.py#L11)，各类检测器以及语言模型。
- [x] 高级剪枝器(High-level pruners): [MagnitudePruner](https://arxiv.org/abs/1608.08710), [BNScalePruner](https://arxiv.org/abs/1708.06519), [GroupNormPruner](https://arxiv.org/abs/2301.12900) (论文所使用的组剪枝器), RandomPruner等  
- [x] DepGraph, 支持计算图构建和依赖建模.
- [x] 支持的基础模块: Conv, Linear, BatchNorm, LayerNorm, Transposed Conv, PReLU, Embedding, MultiheadAttention, nn.Parameters and [自定义层(customized modules)](tests/test_customized_layer.py).
- [x] 支持的操作: split, concatenation, skip connection, flatten, reshape, view, all element-wise ops等
- [x] [底层剪枝函数 (Low-level pruning functions)](torch_pruning/pruner/function.py)
- [x] [Benchmarks](benchmarks) and [tutorials](tutorials)
- [x] 资源列表[resource list](practical_structural_pruning.md) for practical structrual pruning.

### **后续开发计划:**
- [ ] 剪枝适配性基准线, 覆盖 [Torchvision](https://pytorch.org/vision/stable/models.html) (**81/85=95.3%** , :heavy_check_mark:)和[timm](https://github.com/huggingface/pytorch-image-models)等常见模型库.
- [ ] Pruning from Scratch / at Initialization.
- [ ] 语言、语音、生成式模型剪枝
- [ ] 更多的高级剪枝器, 例如[FisherPruner](https://arxiv.org/abs/2108.00708), [GrowingReg](https://arxiv.org/abs/2012.09243)等.
- [ ] 更多的标准层: GroupNorm, InstanceNorm, Shuffle Layers, etc.
- [ ] 更多的Transformer网络: Vision Transformers (:heavy_check_mark:), Swin Transformers, PoolFormers.
- [ ] Block/Layer/Depth Pruning
- [ ] 性能基准线: 支持CIFAR, ImageNet and COCO.

## 安装
Torch-Pruning支持Pytorch 1.0和2.0版本。本项目主要使用Pytorch>=1.13.1进行开发和测试。
```bash
pip install torch-pruning # v1.1.6
```
或者
```bash
git clone https://github.com/VainF/Torch-Pruning.git # recommended
```

## Quickstart
  
本节内容提供了Torch-Pruning的简单例子, 用于快速了解项目的主要功能. 更多细节请参考[tutorals](./tutorials/)

### 0. 工作原理

在复杂的网络结构中, 参数之间可能存在依赖关系, 这种依赖要求算法对这类参数进行同步移除以保证结构正确性，这就涉及到耦合参数的分组问题. 我们的工作通过提供一种自动化机制来对参数进行分组. 具体而言, Torch-Pruning使用伪输入来运行您的模型, 跟踪网络计算图, 并记录层之间的依赖关系. 当您剪枝某一层时, Torch-Pruning会识别所有耦合层, 并返回包含这些耦合信息的``tp.Group``.此外, 如果存在像 torch.split 或 torch.cat 这样的操作, 所有剪枝索引都将自动对齐.

<div align="center">
<img src="assets/dep.png" width="100%">
</div>

### 1. A minimal example

```python
import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True).eval()

# 1. 构建依赖图
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. 指定剪枝的通道维度
pruning_idxs = [2, 6, 9]
pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )

print(pruning_group.details())  # or print(pruning_group)

# 3. 检查剩余通道数是否<=0, 并执行剪枝
if DG.check_pruning_group(pruning_group):
    pruning_group.prune()
```
  
这个例子演示了使用 DepGraph剪枝的基本流程.值得注意的是, resnet.conv1实际上会与多个层耦合在一起.通过打印返回的组, 我们观察到组内各个层之间的剪枝是如何互相“触发”的.在以下输出中, “A => B”表示剪枝操作“A”触发剪枝操作“B”.group[0]是用户在DG.get_pruning_group中给出的剪枝操作.

```
--------------------------------
          Pruning Group
--------------------------------
[0] prune_out_channels on conv1 (Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)) => prune_out_channels on conv1 (Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)), idxs=[2, 6, 9] (Pruning Root)
[1] prune_out_channels on conv1 (Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)) => prune_out_channels on bn1 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), idxs=[2, 6, 9]
[2] prune_out_channels on bn1 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => prune_out_channels on _ElementWiseOp(ReluBackward0), idxs=[2, 6, 9]
[3] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_out_channels on _ElementWiseOp(MaxPool2DWithIndicesBackward0), idxs=[2, 6, 9]
[4] prune_out_channels on _ElementWiseOp(MaxPool2DWithIndicesBackward0) => prune_out_channels on _ElementWiseOp(AddBackward0), idxs=[2, 6, 9]
[5] prune_out_channels on _ElementWiseOp(MaxPool2DWithIndicesBackward0) => prune_in_channels on layer1.0.conv1 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), idxs=[2, 6, 9]
[6] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on layer1.0.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), idxs=[2, 6, 9]
[7] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on _ElementWiseOp(ReluBackward0), idxs=[2, 6, 9]
[8] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_out_channels on _ElementWiseOp(AddBackward0), idxs=[2, 6, 9]
[9] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_in_channels on layer1.1.conv1 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), idxs=[2, 6, 9]
[10] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on layer1.1.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), idxs=[2, 6, 9]
[11] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on _ElementWiseOp(ReluBackward0), idxs=[2, 6, 9]
[12] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_in_channels on layer2.0.downsample.0 (Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)), idxs=[2, 6, 9]
[13] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_in_channels on layer2.0.conv1 (Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)), idxs=[2, 6, 9]
[14] prune_out_channels on layer1.1.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => prune_out_channels on layer1.1.conv2 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), idxs=[2, 6, 9]
[15] prune_out_channels on layer1.0.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => prune_out_channels on layer1.0.conv2 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), idxs=[2, 6, 9]
--------------------------------
```
更多细节请参考[tutorials/2 - Exploring Dependency Groups](https://github.com/VainF/Torch-Pruning/blob/master/tutorials/2%20-%20Exploring%20Dependency%20Groups.ipynb)

#### 如何遍历所有分组:
正如我们在[MetaPruner](https://github.com/VainF/Torch-Pruning/blob/b607ae3aa61b9dafe19d2c2364f7e4984983afbf/torch_pruning/pruner/algorithms/metapruner.py#L197)中所实现的, 我们可以利用``DG.get_all_groups(ignored_layers, root_module_types)``来按顺序扫描所有的分组. 每个分组都会以一个"root_module_types"中所指定的层作为起点. 默认情况下,  这些组包含了完整的剪枝索引``idxs=[0,1,2,3,...,K]``, 这个索引列表包含了所有的可修剪参数的索引. 如果我们希望对一个group进行剪枝, 我们需要使用``group.prune(idxs=idxs)``来指定具体的修剪通道/维度.

```python
for group in DG.get_all_groups(ignored_layers=[model.conv1], root_module_types=[nn.Conv2d, nn.Linear]):
    # handle groups in sequential order
    idxs = [2,4,6] # your pruning indices
    group.prune(idxs=idxs)
    print(group)
```



### 2. 高级剪枝器（High-level Pruners）

利用 DependencyGraph, 我们在项目中开发了几个高级剪枝器, 以便实现一键式剪枝.通过指定所需的通道稀疏性, 您可以对整个模型进行修剪, 并使用自己的训练代码进行微调.关于此过程的详细信息, 我们建议您查阅[Tutorial-1](https://github.com/VainF/Torch-Pruning/blob/master/tutorials/1%20-%20Customize%20Your%20Own%20Pruners.ipynb), 该文档演示了如何基于Torch-Pruning快速实现一个经典的[slimming算法](https://arxiv.org/abs/1708.06519).此外, 您可以在[benchmarks/main.py](benchmarks/main.py)中找到更多实用的示例.

```python
import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True)

# 重要性指标
example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.MagnitudeImportance(p=2)

ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

iterative_steps = 5 # 迭代式剪枝, 该示例会分五步完成50%通道剪枝 (10%->20%->...->50%)
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps,
    ch_sparsity=0.5, # 整体移除50%通道, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # finetune your model here
    # finetune(model)
    # ...
```

#### 稀疏训练
一些剪枝器pruners例如[BNScalePruner](https://github.com/VainF/Torch-Pruning/blob/dd59921365d72acb2857d3d74f75c03e477060fb/torch_pruning/pruner/algorithms/batchnorm_scale_pruner.py#L45)和 [GroupNormPruner](https://github.com/VainF/Torch-Pruning/blob/dd59921365d72acb2857d3d74f75c03e477060fb/torch_pruning/pruner/algorithms/group_norm_pruner.py#L53)依赖稀疏训练来寻找冗余参数. 这个过程可以通过向您的训练代码中加入一行``pruner.regularize(model)``来实现. 该操作会将稀疏训练的梯度叠加到网络的梯度上, 您可以使用任意的优化器进行优化.
```python
for epoch in range(epochs):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, target)
        loss.backward()
        pruner.regularize(model) # <== for sparse learning
        optimizer.step()
```

#### 交互式剪枝
所有的高级剪枝器都支持交互式剪枝. 你可以利用``pruner.step(interactive=True)``来获得所有的待剪枝分组, 并根据需要调用``group.prune()``来完成修剪. 这一功能可以用于控制/监控整个剪枝过程.

```python
for i in range(iterative_steps):
    for group in pruner.step(interactive=True): # Warning: 分组必须按顺序进行处理, 因为剪枝会影响模型建构, 改变坐标索引.
        print(group) 
        # do whatever you like with the group 
        # ...
        group.prune() # 此处需要手动调用group.prune()
        # group.prune(idxs=[0, 2, 6]) # 您甚至可以手动修改剪枝的索引
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # finetune your model here
    # finetune(model)
    # ...
```

#### 组剪枝                                       
利用DepGraph, 我们可以比较轻松地设计出各种组级别重要性评估指标(group-level criteria), 用于一组参数的重要性. 这不同于过去仅用于单层的重要性评估. 在我们的论文中, 我们构造了一种简单的组剪枝器[GroupNormPruner](https://github.com/VainF/Torch-Pruning/blob/745f6d6bafba7432474421a8c1e5ce3aad25a5ef/torch_pruning/pruner/algorithms/group_norm_pruner.py#L8) (如下图c所示).该剪枝器通过组级别的稀疏来学习到具有一致重要性的耦合参数, 确保被移除的参数均具有较小的重要性得分.

<div align="center">
<img src="assets/group_sparsity.png" width="80%">
</div>

### 3. 模型的保存与读取
#### 最简单的方式
以下代码直接将模型对象序列化为.pth文件，该方式足够简单但是会导致存储文件偏大，不方便通过互联网分享。
```python
torch.save(model, 'model.pth') # without .state_dict
model = torch.load('model.pth') # load the model object
```

#### 剪枝历史（Pruning History）
我们介绍一种利用 ``pruning_history`` 来存储和读取剪枝后模型的方法，该方法与PyTorch采用的``state_dict``非常相似。请参考样例 [tests/test_load.py](https://github.com/VainF/Torch-Pruning/blob/master/tests/test_load.py)
```python
...
# Save
state_dict = {
    'model': model.state_dict(), # 标准的Pytorch存储方式
    'pruning': pruner.pruning_history(), # 依赖图DG支持相同的DG.pruning_history & DG.load_pruning_history接口
}
torch.save(state_dict, 'pruned_model.pth')

# Load
model = resnet18() # 创建一个未剪枝的模型
DG = tp.DependencyGraph().build_dependency(model, example_inputs) # 创建一个依赖图DG或者Pruner
state_dict = torch.load('pruned_model.pth') # 读取模型参数
DG.load_pruning_history(state_dict['pruning']) # 读取剪枝历史，并对网络重复相同的裁剪
model.load_state_dict(state_dict['model']) # 重新剪枝后，模型可以读取存储的参数
print(model)
```
  
### 4. 底层剪枝函数（Low-level pruning functions）

虽然使用低级函数可以手动修剪模型, 但这种方法可能非常繁琐, 因为它需要手动管理相关依赖项.因此, 我们建议利用前面提到的高级剪枝器来简化剪枝过程.

```python
tp.prune_conv_out_channels( model.conv1, idxs=[2,6,9] )

# fix the broken dependencies manually
tp.prune_batchnorm_out_channels( model.bn1, idxs=[2,6,9] )
tp.prune_conv_in_channels( model.layer2[0].conv1, idxs=[2,6,9] )
...
```

您可以使用以下的剪枝函数:
```python
tp.prune_conv_out_channels,
tp.prune_conv_in_channels,
tp.prune_depthwise_conv_out_channels,
tp.prune_depthwise_conv_in_channels,
tp.prune_batchnorm_out_channels,
tp.prune_batchnorm_in_channels,
tp.prune_linear_out_channels,
tp.prune_linear_in_channels,
tp.prune_prelu_out_channels,
tp.prune_prelu_in_channels,
tp.prune_layernorm_out_channels,
tp.prune_layernorm_in_channels,
tp.prune_embedding_out_channels,
tp.prune_embedding_in_channels,
tp.prune_parameter_out_channels,
tp.prune_parameter_in_channels,
tp.prune_multihead_attention_out_channels,
tp.prune_multihead_attention_in_channels,
```

### 5. 自定义层

请参考[tests/test_customized_layer.py](https://github.com/VainF/Torch-Pruning/blob/master/tests/test_customized_layer.py). 该示例演示了如何剪枝用户自定义的层.

### 6. 基准线 Benchmarks

Our results on {ResNet-56 / CIFAR-10 / 2.00x}

| Method | Base (%) | Pruned (%) | $\Delta$ Acc (%) | Speed Up |
|:--    |:--:  |:--:    |:--: |:--:      |
| NIPS [[1]](#1)  | -    | -      |-0.03 | 1.76x    |
| Geometric [[2]](#2) | 93.59 | 93.26 | -0.33 | 1.70x |
| Polar [[3]](#3)  | 93.80 | 93.83 | +0.03 |1.88x |
| CP  [[4]](#4)   | 92.80 | 91.80 | -1.00 |2.00x |
| AMC [[5]](#5)   | 92.80 | 91.90 | -0.90 |2.00x |
| HRank [[6]](#6) | 93.26 | 92.17 | -0.09 |2.00x |
| SFP  [[7]](#7)  | 93.59 | 93.36 | +0.23 |2.11x |
| ResRep [[8]](#8) | 93.71 | 93.71 | +0.00 |2.12x |
||
| Ours-L1 | 93.53 | 92.93 | -0.60 | 2.12x |
| Ours-BN | 93.53 | 93.29 | -0.24 | 2.12x |
| Ours-Group | 93.53 | 93.77 | +0.38 | 2.13x |

详细信息请参考[benchmarks](benchmarks)

## Citation
```
@article{fang2023depgraph,
  title={DepGraph: Towards Any Structural Pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
