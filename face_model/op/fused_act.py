import os
from pathlib import Path
from re import T

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load, _import_module_from_library

TEMP_GPU_FILE_NAME = 'gpu_name.txt'

compile = False

def compiler():
    print('GPEN is compiling fused Pytorch extension. Please wait, it might take a while.')
    module_path = os.path.dirname(__file__)
    if torch.cuda.is_available():
        fused = load(
            'fused',
            sources=[
                os.path.join(module_path, 'fused_bias_act.cpp'),
                os.path.join(module_path, 'fused_bias_act_kernel.cu'),
            ],
        )
    return fused

# save last choosed device on a txt file
# ---------------------------------------------
temp_gpu_name_path = Path(os.environ['localappdata'] + f'/GPEN_temp/{TEMP_GPU_FILE_NAME}')

# Check if file path exists
if Path.exists(temp_gpu_name_path):
    # if it exists, reads the GPU name from the file
    with open(temp_gpu_name_path, 'r') as f:
        gpu_name = f.readline()
    # if the current gpu name is different from the saved gpu name
    if torch.cuda.get_device_name(torch.cuda.current_device()) != gpu_name:
        # save the new gpu name on the file
        with open(temp_gpu_name_path, 'w') as f:
            f.write(torch.cuda.get_device_name(torch.cuda.current_device()))
        # and compile the C++/CUDA extensions
        compile = True
else:
    # if the file doesn't exist, create the path to the file
    os.makedirs(temp_gpu_name_path.parent, exist_ok=True)
    # and it saves the current gpu name on the file
    with open(temp_gpu_name_path, 'w') as f:
        f.write(torch.cuda.get_device_name(torch.cuda.current_device()))
    # and compile C++/CUDA extensions
    compile = True

if compile:
    # if running GPEN without cuda, please comment line 10-18
    fused = compiler()
else:
    module_path = Path(os.environ['localappdata'] + '/torch_extensions/torch_extensions/Cache/fused')
    try:
        fused = _import_module_from_library('fused', module_path, True)
    except:
        fused = compiler()

class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5, device='cpu'):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale
        self.device = device

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale, self.device)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5, device='cpu'):
    if torch.cuda.is_available() and device != 'cpu':
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
    else:
        return scale * F.leaky_relu(input + bias.view((1, -1)+(1,)*(len(input.shape)-2)), negative_slope=negative_slope)
