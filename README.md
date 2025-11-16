# PyTorch 学习教程

这是一个全面的 PyTorch 学习资源库，包含从基础到进阶的系统化教程，适合深度学习初学者和进阶学习者。

## 📚 项目简介

本项目包含了一系列精心组织的 Jupyter Notebook 教程，涵盖 PyTorch 的核心概念、深度学习基础知识以及实际应用案例。通过这些教程，你可以系统地学习 PyTorch 框架并掌握深度学习的实践技能。

## 🎯 学习路径

### 第一阶段：PyTorch 基础
掌握 PyTorch 的核心概念和基本操作

- **1-01 Tensor 教程** (`1-01tensor_tutorial.ipynb`)
  - 张量的创建、操作和运算
  - PyTorch 的核心数据结构

- **1-02 自动微分教程** (`1-02autograd_tutorial.ipynb`)
  - 自动求导机制
  - 反向传播的实现原理

- **1-03 神经网络教程** (`1-03neural_networks_tutorial.ipynb`)
  - 使用 nn.Module 构建神经网络
  - 损失函数和优化器

- **1-04 CIFAR10 教程** (`1-04cifar10_tutorial.ipynb`)
  - 图像分类实战
  - 训练和测试流程

- **1-05 数据并行教程** (`1-05data_parallel_tutorial.ipynb`)
  - 单机多GPU训练
  - DataParallel 的使用

### 第二阶段：深度学习核心
深入理解深度学习的基础知识和核心网络结构

- **2-01-1 PyTorch 基础（上）** (`2-01-1PyTorch_basics.ipynb`)
- **2-01-2 PyTorch 基础（下）** (`2-01-2PyTorch_basics.ipynb`)
  - PyTorch 基础操作进阶
  - 数据加载和预处理

- **2-02 深度学习基础** (`2-02deeplearning_basic.ipynb`)
  - 深度学习基本概念
  - 前向传播和反向传播

- **2-03 神经网络进阶** (`2-03deeplearning_NeuralNetwork.ipynb`)
  - 多层神经网络设计
  - 激活函数和正则化

- **2-04 卷积神经网络 (CNN)** (`2-04cnn.ipynb`)
  - 卷积层、池化层的原理
  - 经典 CNN 架构

- **2-05 循环神经网络 (RNN)** (`2-05rnn.ipynb`)
  - RNN 的基本原理
  - LSTM 和 GRU

### 第三阶段：经典应用实战
通过经典案例掌握实际应用技能

- **3-01 逻辑回归** (`3-01logistic_regression.ipynb`)
  - 二分类问题
  - 损失函数和优化

- **3-02 MNIST 手写数字识别** (`3-02mnist.ipynb`)
  - 经典计算机视觉任务
  - 完整的训练和评估流程

- **3-03 RNN 应用** (`3-03rnn.ipynb`)
  - 序列数据处理
  - RNN 实战案例

### 第四阶段：高级技术
掌握生产环境中的高级技术和工具

- **4-01 模型微调 (Fine-tuning)** (`4-01fine_tuning.ipynb`)
  - 迁移学习
  - 预训练模型的使用

- **4-02 可视化工具** (`4-02visdom&tensorboardx .ipynb`)
  - Visdom 的使用
  - TensorBoardX 可视化训练过程

- **4-03 FastAI 框架** (`4-03fastai.ipynb`)
  - FastAI 库的使用
  - 快速构建深度学习模型

- **4-05 多GPU并行训练** (`4-05multiply_gpu_parallel_training.ipynb`)
  - 分布式训练策略
  - 多GPU训练优化

## 🛠️ 环境要求

### 基础环境
- Python 3.7+
- PyTorch 1.0+
- CUDA（可选，用于GPU加速）

### 依赖库
```bash
pip install torch torchvision
pip install jupyter notebook
pip install numpy matplotlib
pip install visdom tensorboardX
pip install fastai
```

### 推荐安装
```bash
# 使用 conda 创建虚拟环境
conda create -n pytorch-learning python=3.8
conda activate pytorch-learning

# 安装 PyTorch（根据你的CUDA版本选择）
conda install pytorch torchvision torchaudio -c pytorch

# 安装其他依赖
pip install jupyter visdom tensorboardX fastai
```

## 🚀 快速开始

1. **克隆项目**
```bash
git clone <repository-url>
cd PyTorch-learning
```

2. **启动 Jupyter Notebook**
```bash
jupyter notebook
```

3. **按顺序学习**
   - 建议从第一阶段开始，循序渐进
   - 每个 notebook 都包含详细的代码和注释
   - 运行每个单元格，理解输出结果

## 📖 学习建议

1. **循序渐进**：按照学习路径的顺序进行学习，不要跳跃
2. **动手实践**：运行每一行代码，修改参数观察结果变化
3. **记录笔记**：记录重要概念和遇到的问题
4. **项目实战**：学完基础后，尝试自己的项目
5. **社区交流**：遇到问题积极查找资料和交流

## 📝 学习资源

### 官方文档
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch 教程](https://pytorch.org/tutorials/)

### 推荐书籍
- 《深度学习入门：基于Python的理论与实现》
- 《动手学深度学习》
- 《PyTorch深度学习实战》

### 在线课程
- Andrew Ng 的深度学习课程
- Fast.ai 实践课程
- PyTorch 官方教程

## 🤝 贡献

欢迎提交问题和改进建议！如果你发现了错误或有更好的实现方式，请：

1. Fork 本项目
2. 创建你的特性分支
3. 提交你的修改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目仅供学习交流使用。

## 💡 提示

- 建议使用 GPU 进行训练，可以大大提高速度
- 每个 notebook 的运行时间取决于你的硬件配置
- 某些高级教程需要先完成基础教程的学习
- 遇到问题时，先检查 PyTorch 版本是否兼容

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交 Issue
- 发起 Discussion

---

⭐ 如果这个项目对你有帮助，请给一个 Star！

**祝学习愉快！** 🎉
