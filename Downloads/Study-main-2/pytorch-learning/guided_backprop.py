"""
Guided Backpropagation 实现
用于生成神经网络的可视化梯度
"""

import torch
import torch.nn as nn


class GuidedBackprop:
    """
    使用 Guided Backpropagation 生成梯度可视化
    """
    
    def __init__(self, model):
        """
        初始化 Guided Backpropagation
        
        Args:
            model: 要可视化的神经网络模型
        """
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.model.eval()
        self.update_relus()
        self.hook_layers()
    
    def hook_layers(self):
        """Hook 第一个卷积层来获取梯度"""
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        
        # Hook 第一层
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)
    
    def update_relus(self):
        """
        更新 ReLU 激活函数，使其在反向传播时只传播正梯度
        这是 Guided Backpropagation 的核心
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            如果有 ReLU 层，在反向传播时：
            1. 克隆输出梯度
            2. 将负梯度设为 0（这是 guided backprop 的关键）
            """
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # 移除已使用的前向输出
            return (modified_grad_out,)
        
        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            在前向传播时存储 ReLU 的输出
            """
            self.forward_relu_outputs.append(ten_out)
        
        # 遍历所有层，为 ReLU 层添加 hook
        for module in self.model.features.modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
    
    def generate_gradients(self, input_image, target_class):
        """
        生成 guided backpropagation 梯度
        
        Args:
            input_image: 输入图像张量 [1, C, H, W]
            target_class: 目标类别索引
            
        Returns:
            梯度图像（numpy 数组）
        """
        # 前向传播
        model_output = self.model(input_image)
        
        # 清零梯度
        self.model.zero_grad()
        
        # 目标类别的 one-hot 编码
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[0][target_class] = 1
        
        # 反向传播
        model_output.backward(gradient=one_hot_output)
        
        # 将梯度转换为 numpy 数组
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        
        return gradients_as_arr


class GuidedBackpropReLUModel:
    """
    简化版本的 Guided Backpropagation（适用于简单模型）
    """
    
    def __init__(self, model, use_cuda=False):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = self.model.cuda()
        
        # 替换 ReLU
        self.update_relus()
    
    def update_relus(self):
        """用 GuidedBackpropReLU 替换所有 ReLU"""
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
    
    def forward(self, x):
        return self.model(x)
    
    def __call__(self, x, index=None):
        if self.cuda:
            x = x.cuda()
        
        x.requires_grad = True
        output = self.forward(x)
        
        if index is None:
            index = torch.argmax(output)
        
        one_hot = torch.zeros_like(output)
        one_hot[0][index] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        return x.grad.cpu().data.numpy()

