from typing import Union
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch
from torch.nn.modules.conv import _ConvNd
# from cnnbase import ConvBase
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t


class Conv2d(_ConvNd):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def conv2d(self, input, kernel, bias=0, stride=1, padding=0):
        '''TODO forword的计算方法'''
        # print(input.shape, padding, kernel)
        self.input = input
        OC, IC, KH, KW = kernel.shape
        N, IC, H, W = input.shape

        self.paddedX = torch.zeros((N, IC, H + 2 * padding, W + 2 * padding), dtype=input.dtype)
        self.paddedX[:, :, padding:H + padding, padding:W + padding] = input

        nOutRows = (H + 2 * padding - KH) // stride + 1
        nOutCols = (W + 2 * padding - KW) // stride + 1

        out = torch.zeros(N, OC, nOutRows, nOutCols, dtype=input.dtype)
        for outCh in range(OC):
            for iRow in range(nOutRows):
                startRow = iRow * stride
                for iCol in range(nOutCols):
                    startCol = iCol * stride
                    out[:, outCh, iRow, iCol] = \
                        (self.paddedX[:, :, startRow:startRow + KH, startCol:startCol + KW] \
                         * kernel[outCh, :, 0:KH, 0:KW]).sum(axis=(1, 2, 3))

        if bias is not None:
            out += bias.view(1, -1, 1, 1)
        self.output = out
        return self.output

    def forward(self, input: Tensor):
        weight = self.weight
        bias = self.bias
        return self.conv2d(input, weight, bias)

    def backward(self, ones: Tensor):
        '''TODO backward的计算方法'''
        padding, stride = self.padding, self.stride
        N, IC, nPadImgRows, nPadImgCols = self.paddedX.shape
        OC, IC, KH, KW = self.weight.shape
        N, OC, nOutRows, nOutCols = ones.shape

        grad_padX = torch.zeros_like(self.paddedX, requires_grad=False)
        grad_weight = torch.zeros_like(self.weight, requires_grad=False)
        for outCh in range(OC):
            for iRow in range(nOutRows):
                startRow = iRow * stride[0]
                for iCol in range(nOutCols):
                    startCol = iCol * stride[1]

                    grad_padX[:, :, startRow:startRow + KH, startCol:startCol + KW] += \
                        ones[:, outCh, iRow, iCol].reshape(-1, 1, 1, 1) * \
                        self.weight[outCh, :, 0:KH, 0:KW].detach()

                    grad_weight[outCh, :, 0:KH, 0:KW] += \
                        (self.paddedX[:, :, startRow:startRow + KH, startCol:startCol + KW] * \
                         ones[:, outCh, iRow, iCol].reshape(-1, 1, 1, 1)).sum(axis=0).detach()
        grad_inputX = grad_padX[:, :, padding[0]:nPadImgRows - padding[0], padding[1]:nPadImgCols - padding[1]]

        grad_bias = ones.sum(axis=(0, 2, 3))
        self.input.grad = grad_inputX
        self.weight.grad = grad_weight
        self.bias.grad = grad_bias
        return grad_inputX


class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))  # 随机weight
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))

    def forward(self, input):
        self.input = input
        self.output = torch.matmul(input, self.weight.T)
        if hasattr(self, 'bias'):
            self.output += self.bias
        return self.output

    def backward(self, ones: Tensor):
        self.input.grad = torch.mm(ones, self.weight).detach()  # 计算输入梯度
        self.weight.grad = torch.mm(ones.t(), self.input).detach()  # 计算权重梯度
        self.bias.grad = ones.sum(0).detach()  # 计算偏置梯度
        return self.input.grad


class CrossEntropyLoss():
    def __init__(self):
        pass

    def __call__(self, input, target):
        self.input = input
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        self.target = target_one_hot
        self.log_softmax_input = torch.log_softmax(input, dim=1)
        self.loss = -torch.mean(torch.sum(self.log_softmax_input * self.target, dim=1))
        self.output = self.loss
        return self.output

    def backward(self):
        self.input.grad = (torch.exp(self.log_softmax_input) - self.target) / self.target.size()[0]
        return self.input.grad

