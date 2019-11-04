# Import PyTorch
import torch  # import main library
from torch.autograd import Function  # import Function to create custom activations
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

class BinarizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # save input for backward pass

        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone()
        grad_input[torch.abs(input) > 1.] = 0.
        grad_input[torch.abs(input) <= 1.] = 1.
        grad_input = grad_input * grad_output

        return grad_input

class Binarization(nn.Module):

    def __init__(self, min=-1, max=1):
        super(Binarization, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return 0.5*(BinarizeFunction.apply(input)*(self.max - self.min) + self.min + self.max)


class BinarizedLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizedLinear, self).__init__(*kargs, **kwargs)
        self.binarization = Binarization(min=min_weight, max=max_weight)
        self.min_weight = -1
        self.max_weight = +1

    def forward(self, input):
        self.weight.data = nn.functional.hardtanh_(self.weight.data) 
        return nn.functional.linear(input, self.binarization(self.weight), bias=self.bias)  # linear layer with binarized weights

class BinarizedConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizedConv2d, self).__init__(*kargs, **kwargs)
        self.min_weight = -1
        self.max_weight = +1
        self.binarization = Binarization(min=self.min_weight, max=self.max_weight)

    def forward(self, input):
        self.weight.data = nn.functional.hardtanh_(self.weight.data)
        return nn.functional.conv2d(input, self.binarization(self.weight), self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
