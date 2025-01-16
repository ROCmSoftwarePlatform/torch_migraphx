#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################
import operator
import warnings

import torch
from typing import cast, Iterable, List, Sequence

from sys import modules
try:
    import torchvision
except ImportError:
    # torchvision is not a mandatory dependency, so may not be present
    pass
import torch.nn as nn
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from packaging import version

from . import acc_utils
from .acc_normalizer import (
    register_acc_op,
    register_acc_op_mapping,
    register_custom_acc_mapper_fn,
)
from .acc_op_properties import AccOpProperty, register_acc_op_properties

this_arg_is_optional = True
move_to_qparams = True
dont_move_to_qparams = False


def reduce_op_mapper(node: torch.fx.Node, mod: torch.fx.GraphModule,
                     func) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        kwargs = dict(node.kwargs)
        if "dim" in kwargs and isinstance(kwargs["dim"], int):
            kwargs["dim"] = (kwargs["dim"], )
        new_node = node.graph.call_function(func, kwargs=kwargs)
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op
def mean(*, input, dim=None, keepdim=False, dtype=None):
    if dim is not None:
        return torch.mean(input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        return input.mean(dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "mean"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.mean),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def mean_mapper(node, mod):
    return reduce_op_mapper(node, mod, mean)

@register_acc_op
def std(*, input, dim=None, correction=1, keepdim=False):
    if dim is not None:
        return torch.std(input, dim=dim, correction=correction, keepdim=keepdim)
    else:
        return input.std(correction=correction, keep_dim=keepdim)
    
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "std"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("correction", "correction", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.std),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("correction", "correction", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
def std_mapper(node, mod):
    return reduce_op_mapper(node, mod, std)


@register_acc_op
def sum(*, input, dim=None, keepdim=False, dtype=None):
    if dim is not None:
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        return input.sum(dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "sum"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.sum),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def sum_mapper(node: torch.fx.Node,
               mod: torch.fx.GraphModule) -> torch.fx.Node:
    return reduce_op_mapper(node, mod, sum)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def prod(*, input, dim=None, keepdim=False, dtype=None):
    if dim is not None:
        return torch.prod(input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        return input.prod(dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "prod"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.prod),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def prod_mapper(node: torch.fx.Node,
                mod: torch.fx.GraphModule) -> torch.fx.Node:
    func = prod
    with node.graph.inserting_before(node):
        kwargs = dict(node.kwargs)
        new_node = node.graph.call_function(func, kwargs=kwargs)
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_mapping(op_and_target=("call_function", torch.maximum))
@register_acc_op_mapping(op_and_target=("call_method", "maximum"))
@register_acc_op
def maximum(*, input, other):
    return torch.maximum(input=input, other=other)


@register_acc_op_mapping(
    op_and_target=("call_method", "max"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.max),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_acc_op
def max(*, input, dim=None, keepdim=False):
    if dim is not None:
        return torch.max(input, dim=dim, keepdim=keepdim)
    else:
        return torch.max(input)


@register_acc_op_mapping(
    op_and_target=("call_method", "min"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.min),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_acc_op
def min(*, input, dim=None, keepdim=False):
    if dim is not None:
        return torch.min(input, dim=dim, keepdim=keepdim)
    else:
        return torch.min(input)


@register_acc_op_mapping(op_and_target=("call_method", "minimum"))
@register_acc_op_mapping(op_and_target=("call_function", torch.minimum))
@register_acc_op
def minimum(*, input, other):
    return torch.minimum(input=input, other=other)


@register_acc_op_mapping(op_and_target=("call_function", operator.getitem))
@register_acc_op
def getitem(*, input, idx):
    return input[idx]


@register_acc_op
def size(*, input):
    return input.size()


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.numel))
@register_acc_op
def numel(*, input):
    return torch.numel(input)


@register_acc_op_mapping(op_and_target=("call_function", torch.slice_scatter))
@register_acc_op
def slice_scatter(*, input, src, dim=0, start=None, end=None, step=1):
    return torch.slice_scatter(input=input,
                               src=src,
                               dim=dim,
                               start=start,
                               end=end,
                               step=step)


@register_acc_op_mapping(op_and_target=("call_function", torch.select_scatter))
@register_acc_op
def select_scatter(*, input, src, dim, index):
    return torch.select_scatter(input=input, src=src, dim=dim, index=index)



@register_acc_op_mapping(op_and_target=("call_function", torch.index_select))
@register_acc_op
def index_select(*, input, dim, index):
    return torch.index_select(input, dim, index)

  
@register_acc_op_mapping(op_and_target=("call_function", torch.scatter_reduce))
@register_acc_op_mapping(op_and_target=("call_method", "scatter_reduce"))
@register_acc_op
def scatter_reduce(*, input, dim, index, src, reduce, include_self=True):
    return torch.scatter_reduce(input=input,
                                dim=dim,
                                index=index,
                                src=src,
                                reduce=reduce,
                                include_self=include_self)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.scatter_add),
    arg_replacement_tuples=[("input", "input"), ("dim", "dim"),
                            ("index", "index"), ("src", "src")],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "scatter_add"),
    arg_replacement_tuples=[("input", "input"), ("dim", "dim"),
                            ("index", "index"), ("src", "src")],
)
def scatter_add_mapper(node: torch.fx.Node, _: nn.Module):
    with node.graph.inserting_before(node):
        kwargs = {k: v for k, v in node.kwargs.items()}
        kwargs["reduce"] = "sum"
        kwargs["include_self"] = True
        new_node = node.graph.create_node("call_function",
                                          scatter_reduce,
                                          kwargs=kwargs)
        new_node.meta = node.meta.copy()
    return new_node


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.linear))
@register_acc_op
def linear(*, input, weight, bias):
    return nn.functional.linear(input=input, weight=weight, bias=bias)

@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.nll_loss),
    arg_replacement_tuples=[
        ("input", "input"),  
        ("target", "target"),
        ("weight",  "weight", this_arg_is_optional),
        ("size_average", "size_average", this_arg_is_optional),
        ("reduce", "reduce", this_arg_is_optional),
        ("reduction", "reduction", this_arg_is_optional),
        ("ignore_index" , "ignore_index" , this_arg_is_optional)
    ],
)

# registering an acc op defines the list of arguments recognized when we define a forward 
# function in a torch.nn.Module .
@register_acc_op
def nll_loss(*, input, target, weight=None, reduce=None, reduction='mean', size_average=None, ignore_index=-100):
    return torch.nn.functional.nll_loss(input=input, target=target, weight=weight,
                                        reduce=reduce, reduction=reduction,
                                        size_average=size_average, ignore_index=ignore_index)
    
@register_acc_op_mapping(
    op_and_target=("call_function", torch.clamp),
    arg_replacement_tuples=[
        ("input", "input", False),
        ("min", "min", True),
        ("max", "max", True),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "clamp"),
    arg_replacement_tuples=[
        ("input", "input", False),
        ("min", "min", True),
        ("max", "max", True),
    ],
)
@register_acc_op
def clamp(*, input, min=None, max=None):
    return torch.clamp(input=input, min=min, max=max)


if 'torchvision' in modules:
    @register_acc_op_mapping(
        op_and_target=("call_function", torchvision.ops.roi_align),
        arg_replacement_tuples=[
            ("input", "input", False),
            ("boxes", "boxes", False),
            ("output_size", "output_size", False),
            ("spatial_scale", "spatial_scale", this_arg_is_optional),
            ("sampling_ratio", "sampling_ratio", this_arg_is_optional),
            ("aligned", "aligned", this_arg_is_optional),
        ],
    )
    @register_acc_op
    def roi_align(*, input, boxes, output_size,
                spatial_scale = 1.0,
                sampling_ratio = -1,
                aligned = False):
        return torchvision.ops.roi_align(input=input, boxes=boxes, output_size = output_size, 
                                        spatial_scale = spatial_scale, 
                                        sampling_ratio = sampling_ratio, aligned = aligned)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "tile"))
@register_acc_op_mapping(op_and_target=("call_function", torch.tile))
@register_acc_op
def tile(*, input, dims):
    return torch.tile(input=input, dims=dims)


@register_acc_op_mapping(
    op_and_target=("call_method", "repeat"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "repeats"),
    ],
)
@register_acc_op
def repeat(*, input, repeats):
    return input.repeat(*repeats)


@register_acc_op_mapping(op_and_target=("call_function", torch.unbind))
@register_acc_op
def unbind(*, input, dim=0):
    return torch.unbind(input, dim=dim)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.stack),
    arg_replacement_tuples=[
        ("tensors", "tensors"),
        ("dim", "dim"),
    ],
)
def stack_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Map torch.stack to unsqueeze + cat.
    """
    with node.graph.inserting_before(node):
        inputs = node.kwargs["tensors"]
        unsqueeze_nodes = []
        assert isinstance(inputs, Sequence)
        for i, t in enumerate(inputs):
            new_node = node.graph.create_node(
                "call_function",
                unsqueeze,
                kwargs={
                    "input": t,
                    "dim": node.kwargs["dim"]
                },
                name=f"{node.name}_unsqueeze_{i}",
            )
            new_node.meta["type"] = torch.Tensor
            unsqueeze_nodes.append(new_node)
        cat_node = node.graph.create_node(
            "call_function",
            cat,
            kwargs={
                "tensors": unsqueeze_nodes,
                "dim": node.kwargs["dim"]
            },
        )
        cat_node.meta = node.meta.copy()
        return cat_node


@register_acc_op_mapping(op_and_target=("call_function", torch.conv1d))
@register_acc_op
def conv1d(*, input, weight, bias, stride, padding, dilation, groups):
    return nn.functional.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@register_acc_op_mapping(op_and_target=("call_function", torch.conv2d))
@register_acc_op
def conv2d(*, input, weight, bias, stride, padding, dilation, groups):
    return nn.functional.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@register_acc_op_mapping(op_and_target=("call_function", torch.conv3d))
@register_acc_op
def conv3d(*, input, weight, bias, stride, padding, dilation, groups):
    return nn.functional.conv3d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@register_acc_op_mapping(op_and_target=("call_function",
                                        torch.nn.functional.conv_transpose2d))
@register_acc_op
def conv_transpose2d(
    *,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    return nn.functional.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


@register_acc_op_mapping(op_and_target=("call_function",
                                        torch.nn.functional.conv_transpose3d))
@register_acc_op
def conv_transpose3d(
    *,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    return nn.functional.conv_transpose3d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.relu))
@register_acc_op_mapping(
    op_and_target=("call_function", torch.relu),
    arg_replacement_tuples=[("input", "input")],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "relu"),
    arg_replacement_tuples=[("input", "input")],
)
@register_acc_op
def relu(*, input, inplace=False):
    return nn.functional.relu(input=input, inplace=inplace)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function",
                                        torch.nn.functional.leaky_relu))
@register_acc_op
def leaky_relu(*, input, negative_slope=0.01, inplace=False):
    return nn.functional.leaky_relu(input=input,
                                    negative_slope=negative_slope,
                                    inplace=inplace)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function",
                                        torch.nn.functional.elu))
@register_acc_op
def elu(*, input, alpha=1.0, inplace=False):
    return nn.functional.elu(input=input, alpha=alpha, inplace=inplace)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function",
                                        torch.nn.functional.selu))
@register_acc_op
def selu(*, input, inplace=False):
    return nn.functional.selu(input=input, inplace=inplace)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function",
                                        torch.nn.functional.softsign))
@register_acc_op
def softsign(*, input):
    return nn.functional.softsign(input=input)


@register_acc_op_mapping(op_and_target=("call_function",
                                        torch.nn.functional.gelu))
@register_acc_op
def gelu(*, input):
    return torch.nn.functional.gelu(input=input)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.glu),
    arg_replacement_tuples=[("input", "input"), ("dim", "dim")],
)
@register_acc_op
def glu(*, input, dim=-1):
    return torch.nn.functional.glu(input=input, dim=dim)


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.hardtanh), )
@register_acc_op
def hardtanh(*, input, min_val=-1.0, max_val=1.0):
    return nn.functional.hardtanh(input=input,
                                  min_val=min_val,
                                  max_val=max_val)


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.hardsigmoid))
@register_acc_op
def hardsigmoid(*, input):
    return nn.functional.hardsigmoid(input)


@register_acc_op_mapping(
    op_and_target=("call_method", "softmax"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.softmax),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op
def softmax(*, input, dim, dtype=None):
    """
    _stacklevel are ignored here.
    """
    return torch.nn.functional.softmax(input=input, dim=dim, dtype=dtype)


@register_acc_op_mapping(
    op_and_target=("call_method", "log_softmax"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.log_softmax),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op
def log_softmax(*, input, dim, dtype=None):
    """
    _stacklevel input is ignored here.
    """
    return torch.nn.functional.log_softmax(input=input, dim=dim, dtype=dtype)


@register_acc_op_mapping(op_and_target=("call_function", torch.linalg.norm))
@register_acc_op
def linalg_norm(*, input, ord, dim, keepdim):
    return torch.linalg.norm(input=input, ord=ord, dim=dim, keepdim=keepdim)


@register_acc_op_mapping(op_and_target=("call_function", torch.linalg.vector_norm))
@register_acc_op
def linalg_vector_norm(*, input, ord, dim, keepdim):
    return torch.linalg.vector_norm(input=input, ord=ord, dim=dim, keepdim=keepdim)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.cumsum),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "cumsum"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op
def cumsum(*, input, dim, dtype=None):
    return torch.cumsum(input=input, dim=dim, dtype=dtype)


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.adaptive_avg_pool2d))
@register_acc_op
def adaptive_avg_pool2d(*, input, output_size):
    return nn.functional.adaptive_avg_pool2d(input=input,
                                             output_size=output_size)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sign))
@register_acc_op
def sign(*, input):
    return torch.sign(input)


@register_acc_op_mapping(
    op_and_target=("call_method", "flatten"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("start_dim", "start_dim", this_arg_is_optional),
        ("end_dim", "end_dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(op_and_target=("call_function", torch.flatten))
@register_acc_op
def flatten(*, input, start_dim=0, end_dim=-1):
    return torch.flatten(input=input, start_dim=start_dim, end_dim=end_dim)


@register_acc_op_mapping(
    op_and_target=("call_method", "squeeze"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.squeeze),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op
def squeeze(*, input, dim=None):
    if dim is None:
        return input.squeeze()
    return input.squeeze(dim=dim)


@register_acc_op_mapping(op_and_target=("call_method", "unsqueeze"))
@register_acc_op_mapping(op_and_target=("call_function", torch.unsqueeze))
@register_acc_op
def unsqueeze(*, input, dim):
    return torch.unsqueeze(input=input, dim=dim)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.reshape),
    arg_replacement_tuples=[
        ("input", "input"),
        ("shape", "shape"),
    ],
)
@register_acc_op
def reshape(*, input, shape):
    return input.reshape(shape)


@register_acc_op_mapping(
    op_and_target=("call_method", "expand"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "sizes"),
    ],
)
@register_acc_op
def expand(*, input, sizes):
    return input.expand(*sizes)


@register_acc_op_mapping(
    op_and_target=("call_method", "permute"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "permutation"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.permute),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dims", "permutation"),
    ],
)
@register_acc_op
def permute(*, input, permutation):
    return input.permute(*permutation)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.masked_fill),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mask", "mask"),
        ("value", "value"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "masked_fill"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mask", "mask"),
        ("value", "value"),
    ],
)
@register_acc_op
def masked_fill(*, input, mask, value):
    return input.masked_fill(mask, value)


@register_acc_op_mapping(op_and_target=("call_function", torch.where))
@register_acc_op
def where(*, condition, input, other):
    return torch.where(condition, input, other)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.square),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def square_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(mul,
                                            kwargs={
                                                "input": input_node,
                                                "other": input_node
                                            })
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_mapping(op_and_target=("call_method", "contiguous"))
@register_acc_op
def contiguous(*, input):
    return input.contiguous()


@register_acc_op_mapping(op_and_target=("call_function", torch.chunk))
@register_acc_op_mapping(op_and_target=("call_method", "chunk"))
@register_acc_op
def chunk(*, input, chunks, dim=0):
    return torch.chunk(input=input, chunks=chunks, dim=dim)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "split"),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_size_or_sections", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "split_with_sizes"),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_sizes", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.split),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_size_or_sections", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
def torch_split_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    """
    If split_size_or_sections is sections, map the node to slice_tensors
    + tuple_construct. Otherwise, if split_size_or_sections is split_size,
    map the node to acc_ops.split.
    """
    split_size_or_sections = node.kwargs["split_size_or_sections"]
    with node.graph.inserting_before(node):
        if isinstance(split_size_or_sections, int):
            new_kwargs = {
                "input": node.kwargs["input"],
                "split_size": split_size_or_sections,
                "dim": node.kwargs["dim"],
            }
            new_node = node.graph.call_function(split, kwargs=new_kwargs)
            new_node.meta = node.meta.copy()
            return new_node

        assert isinstance(split_size_or_sections, Sequence)
        start = 0
        slice_nodes = []
        for i in split_size_or_sections:
            assert isinstance(i, int)
            new_kwargs = {
                "input": node.kwargs["input"],
                "dim": node.kwargs["dim"],
                "start": start,
                "stop": start + i,
                "step": 1,
            }
            new_node = node.graph.call_function(slice_tensor,
                                                kwargs=new_kwargs)
            new_node.meta["type"] = torch.Tensor
            slice_nodes.append(new_node)
            start += i

        new_node = node.graph.call_function(
            tuple_construct, kwargs={"tensors": tuple(slice_nodes)})
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def split(*, input, split_size, dim):
    return torch.split(input, split_size, dim)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.tensor_split),
    arg_replacement_tuples=[
        ("input", "input"),
        (("tensor_indices_or_sections", "sections", "indices"),
         "indices_or_sections"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "tensor_split"),
    arg_replacement_tuples=[
        ("input", "input"),
        (("tensor_indices_or_sections", "sections", "indices"),
         "indices_or_sections"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op
def tensor_split(*, input, indices_or_sections, dim=0):
    # Need to de-coalesce the indices_or_sections because tensor_split accepts
    # one of three kwarg signatures:
    #  * (Tensor input, Tensor tensor_indices_or_sections, int dim)
    #  * (Tensor input, int sections, int dim)
    #  * (Tensor input, tuple of ints indices, int dim)
    if isinstance(indices_or_sections, torch.Tensor):
        indices_or_sections = indices_or_sections.tolist()
    if isinstance(indices_or_sections, int):
        return torch.tensor_split(input, sections=indices_or_sections, dim=dim)
    elif isinstance(indices_or_sections, Iterable):
        return torch.tensor_split(input,
                                  indices=tuple(indices_or_sections),
                                  dim=dim)
    else:
        raise RuntimeError(
            f"Expected int, Iterable or Tensor for "
            f"indices_or_sections arg, got: {type(indices_or_sections)}")


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.max_pool2d))
@register_acc_op
def max_pool2d(
    *,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    return_indices,
):
    return nn.functional.max_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.avg_pool2d))
@register_acc_op
def avg_pool2d(
    *,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    return nn.functional.avg_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.batch_norm))
@register_acc_op
def batch_norm(
    *,
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    eps,
):
    return nn.functional.batch_norm(
        input=input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


@register_acc_op_mapping(op_and_target=("call_function", torch.bitwise_and))
@register_acc_op
def bitwise_and(*, input, other):
    return torch.bitwise_and(input=input, other=other)


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.layer_norm))
@register_acc_op
def layer_norm(*, input, normalized_shape, weight, bias, eps):
    return nn.functional.layer_norm(
        input=input,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.group_norm))
@register_acc_op
def group_norm(*, input, num_groups, weight, bias, eps):
    return nn.functional.group_norm(
        input=input,
        num_groups=num_groups,
        weight=weight,
        bias=bias,
        eps=eps,
    )


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.fmod))
@register_acc_op_mapping(op_and_target=("call_method", "fmod"))
@register_acc_op
def fmod(*, input, other):
    return torch.fmod(input=input, other=other)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sin))
@register_acc_op
def sin(*, input):
    return torch.sin(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.cos))
@register_acc_op
def cos(*, input):
    return torch.cos(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.tan))
@register_acc_op
def tan(*, input):
    return torch.tan(input=input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.topk))
@register_acc_op
def topk(*, input, k, dim, largest, sorted):
    return torch.topk(input=input,
                      k=k,
                      dim=dim,
                      largest=largest,
                      sorted=sorted)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.argmax))
@register_acc_op_mapping(op_and_target=("call_method", "argmax"))
@register_acc_op
def argmax(*, input, dim, keepdim):
    return torch.argmax(input=input, dim=dim, keepdim=keepdim)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.argmin))
@register_acc_op_mapping(op_and_target=("call_method", "argmin"))
@register_acc_op
def argmin(*, input, dim, keepdim):
    return torch.argmin(input=input, dim=dim, keepdim=keepdim)


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.embedding))
@register_acc_op
def embedding(
    *,
    input,
    weight,
    padding_idx,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    sparse,
):
    return torch.nn.functional.embedding(input=input,
                                         weight=weight,
                                         padding_idx=padding_idx,
                                         max_norm=max_norm,
                                         norm_type=norm_type,
                                         scale_grad_by_freq=scale_grad_by_freq,
                                         sparse=sparse)

@register_acc_op_mapping(op_and_target=("call_function", torch.gather))
@register_acc_op
def gather(
    *,
    input,
    dim,
    index
):
    return torch.gather(input, dim, index)


@register_acc_op_mapping(op_and_target=("call_function", torch.cat))
@register_acc_op
def cat(*, tensors, dim):
    return torch.cat(tensors=tensors, dim=dim)


@register_acc_op_mapping(op_and_target=("call_function", torch.sigmoid))
@register_acc_op_mapping(op_and_target=("call_method", "sigmoid"))
@register_acc_op
def sigmoid(*, input):
    return torch.sigmoid(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sinh))
@register_acc_op
def sinh(*, input):
    return torch.sinh(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.cosh))
@register_acc_op
def cosh(*, input):
    return torch.cosh(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.tanh))
@register_acc_op_mapping(op_and_target=("call_method", "tanh"))
@register_acc_op
def tanh(*, input):
    return torch.tanh(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.asin))
@register_acc_op
def asin(*, input):
    return torch.asin(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.acos))
@register_acc_op
def acos(*, input):
    return torch.acos(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.atan))
@register_acc_op
def atan(*, input):
    return torch.atan(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.exp))
@register_acc_op
def exp(*, input):
    return torch.exp(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sqrt))
@register_acc_op_mapping(op_and_target=("call_method", "sqrt"))
@register_acc_op
def sqrt(*, input):
    return torch.sqrt(input=input)

@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.rsqrt))
@register_acc_op_mapping(op_and_target=("call_method", "rsqrt"))
@register_acc_op
def rsqrt(*, input):
    return torch.rsqrt(input=input)

@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.reciprocal))
@register_acc_op
def reciprocal(*, input):
    return torch.reciprocal(input=input)


@register_acc_op_mapping(op_and_target=("call_function", operator.add))
@register_acc_op_mapping(op_and_target=("call_method", "add"))
@register_acc_op
def add(*, input, other):
    return input + other


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.add),
    # Note that we may have aliases for inputs here due to issues with deterministically
    # knowing the correct target that will be resolved by pytorch.
    arg_replacement_tuples=[
        (("input", "a"), "input"),
        (("other", "b"), "other"),
        ("alpha", "alpha", this_arg_is_optional),
    ],
)
def custom_torch_add_mapper(node: torch.fx.Node,
                            mod: nn.Module) -> torch.fx.Node:
    """
    Add custom mapping for torch.add because it has an `alpha` parameter which scales
    the `other` input, and we want to make that mul a separate node.
    """
    with node.graph.inserting_before(node):
        # If alpha is in kwargs check if we need to add a mul, and use correct kwargs.
        if "alpha" in node.kwargs:
            # Add mul node only if it has a numerical impact, i.e. alpha != 1.0.
            if node.kwargs["alpha"] != 1.0:
                other_node = node.graph.create_node(
                    "call_function",
                    mul,
                    kwargs={
                        "input": node.kwargs["other"],
                        "other": node.kwargs["alpha"],
                    },
                    name=node.name + "_mul_alpha",
                )
                other_node.meta = node.meta
            else:
                other_node = node.kwargs["other"]
            add_kwargs = {"input": node.kwargs["input"], "other": other_node}
        else:
            add_kwargs = node.kwargs

        new_node = node.graph.create_node("call_function",
                                          add,
                                          kwargs=add_kwargs,
                                          name=node.name)
        new_node.meta = node.meta
        return new_node


@register_acc_op_mapping(op_and_target=("call_function", torch.sub))
@register_acc_op_mapping(op_and_target=("call_function", operator.sub))
@register_acc_op_mapping(op_and_target=("call_method", "sub"))
@register_acc_op
def sub(*, input, other):
    return input - other


@register_acc_op_mapping(op_and_target=("call_function", torch.mul))
@register_acc_op_mapping(op_and_target=("call_function", operator.mul))
@register_acc_op_mapping(op_and_target=("call_method", "mul"))
@register_acc_op
def mul(*, input, other):
    return input * other


@register_acc_op_mapping(op_and_target=("call_function", torch.abs))
@register_acc_op
def abs(*, input):
    return torch.abs(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.logical_not))
@register_acc_op
def logical_not(*, input):
    return torch.logical_not(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", operator.neg))
@register_acc_op_mapping(op_and_target=("call_function", torch.neg))
@register_acc_op
def neg(*, input):
    return torch.neg(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.floor))
@register_acc_op
def floor(*, input):
    return torch.floor(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.ceil))
@register_acc_op
def ceil(*, input):
    return torch.ceil(input=input)

@register_acc_op_mapping(op_and_target=("call_function", operator.floordiv))
@register_acc_op
def floor_div(*, input, other):
    if isinstance(input, torch.Tensor) or isinstance(other, torch.Tensor):
        return torch.div(input, other, rounding_mode="floor")
    return input // other

@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op
def trunc_div(*, input, other):
    return torch.div(input, other, rounding_mode="trunc")

@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.floor_divide),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
    ],
)
def div_floor_mapper(node: torch.fx.Node,
               mod: torch.fx.GraphModule) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        div_kwargs = dict(node.kwargs)

        if version.parse(torch.__version__) < version.parse("1.13"):
            div_node = node.graph.call_function(
                trunc_div,
                kwargs={
                    "input": div_kwargs["input"],
                    "other": div_kwargs["other"]
                },
            )
        else:
            div_node = node.graph.call_function(
                floor_div,
                kwargs={
                    "input": div_kwargs["input"],
                    "other": div_kwargs["other"]
                },
            )
        div_node.meta = node.meta.copy()
        return div_node
    
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.div),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
        ("rounding_mode", "rounding_mode", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "div"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
        ("rounding_mode", "rounding_mode", this_arg_is_optional),
    ],
)
def div_mapper(node: torch.fx.Node,
               mod: torch.fx.GraphModule) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        div_kwargs = dict(node.kwargs)
        if "rounding_mode" not in div_kwargs or div_kwargs[
                "rounding_mode"] is None:
            div_node = node.graph.call_function(div,
                                                kwargs={
                                                    "input":
                                                    div_kwargs["input"],
                                                    "other":
                                                    div_kwargs["other"]
                                                })
        elif div_kwargs["rounding_mode"] == "trunc":
            div_node = node.graph.call_function(
                trunc_div,
                kwargs={
                    "input": div_kwargs["input"],
                    "other": div_kwargs["other"]
                },
            )
        elif div_kwargs["rounding_mode"] == "floor":
            div_node = node.graph.call_function(
                floor_div,
                kwargs={
                    "input": div_kwargs["input"],
                    "other": div_kwargs["other"]
                },
            )
        else:
            raise RuntimeError(
                f"Unhandled div rounding mode {div_kwargs['rounding_mode']}")
        div_node.meta = node.meta.copy()
        return div_node


@register_acc_op_mapping(op_and_target=("call_function", operator.truediv))
@register_acc_op
def div(*, input, other):
    return input / other


@register_acc_op_mapping(op_and_target=("call_function", torch.log))
@register_acc_op
def log(*, input):
    return torch.log(input=input)

@register_acc_op_mapping(op_and_target=("call_function", torch.log2))
@register_acc_op
def log2(*, input):
    return torch.log2(input=input)

@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.pow))
@register_acc_op_mapping(op_and_target=("call_method", "pow"))
@register_acc_op
def pow(*, input, exponent):
    return torch.pow(input, exponent)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.log1p),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def torch_log1p_mapper(node: torch.fx.Node,
                       _: torch.nn.Module) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        add_kwargs = {"input": node.kwargs["input"], "other": 1.0}
        add_node = node.graph.call_function(add, kwargs=add_kwargs)
        add_node.meta = node.meta.copy()
        log_kwargs = {"input": add_node}
        log_node = node.graph.call_function(log, kwargs=log_kwargs)
        log_node.meta = node.meta.copy()
        return log_node


@register_acc_op_mapping(
    op_and_target=("call_method", "mm"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", operator.matmul),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.bmm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.mm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(op_and_target=("call_function", torch.matmul))
@register_acc_op
def matmul(*, input, other):
    return torch.matmul(input=input, other=other)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.addmm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat1", "mat1"),
        ("mat2", "mat2"),
        ("beta", "beta"),
        ("alpha", "alpha"),
    ],
)
def addmm_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Mapping from torch.addmm to acc_ops.mm -> acc_ops.add, if alpha or beta is not 1
    then we also insert acc_ops.mul to the right place.
    """
    with node.graph.inserting_before(node):
        mm_kwargs = {
            "input": node.kwargs["mat1"],
            "other": node.kwargs["mat2"]
        }
        mm_node = node.graph.create_node("call_function",
                                         matmul,
                                         kwargs=mm_kwargs,
                                         name=f"{node.name}_mm")
        mm_node.meta = node.meta.copy()

        if node.kwargs["alpha"] != 1:
            mul_kwargs = {"input": mm_node, "other": node.kwargs["alpha"]}
            mm_node = node.graph.create_node("call_function",
                                             mul,
                                             kwargs=mul_kwargs,
                                             name=f"{mm_node.name}_mul")
        mm_node.meta = node.meta.copy()

        input_node = node.kwargs["input"]
        if node.kwargs["beta"] != 1:
            mul_kwargs = {"input": input_node, "other": node.kwargs["beta"]}
            new_input_node = node.graph.create_node(
                "call_function",
                mul,
                kwargs=mul_kwargs,
                name=f"{node.name}_input_mul")
            assert isinstance(input_node, torch.fx.Node)
            new_input_node.meta = input_node.meta.copy()
            input_node = new_input_node

        add_kwargs = {"input": mm_node, "other": input_node}
        add_node = node.graph.create_node("call_function",
                                          add,
                                          kwargs=add_kwargs,
                                          name=f"{node.name}_add")
        add_node.meta = node.meta.copy()
        return add_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.silu),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def silu(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        sigmoid_node = node.graph.call_function(sigmoid,
                                                kwargs={"input": input_node})
        sigmoid_node.meta = node.meta.copy()
        new_node = node.graph.call_function(mul,
                                            kwargs={
                                                "input": sigmoid_node,
                                                "other": input_node
                                            })
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.dropout),
    arg_replacement_tuples=[("input", "input")],
)
@register_custom_acc_mapper_fn(op_and_target=("call_method", "detach"),
                               arg_replacement_tuples=[("input", "input")])
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.detach),
    arg_replacement_tuples=[("input", "input")],
)
def dropout_mapper(node: torch.fx.Node, mod: nn.Module):
    """
    Remove dropout node and directly map its input to output.
    """
    return node.kwargs["input"]


try:
    from torchvision.ops import stochastic_depth

    assert callable(stochastic_depth)
except Exception as e:
    warnings.warn(f"Unable to import torchvision related libraries.: {e}")
else:

    @register_custom_acc_mapper_fn(
        op_and_target=("call_function", stochastic_depth),
        arg_replacement_tuples=[("input", "input")],
    )
    def stochastic_depth_mapper(node: torch.fx.Node, mod: nn.Module):
        """
        Remove dropout node and directly map its input to output.
        """
        return node.kwargs["input"]


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.hardswish),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def hardswish_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        new_sigmoid_node = node.graph.call_function(
            hardsigmoid, kwargs={"input": input_node})
        new_sigmoid_node.meta = node.meta.copy()
        new_node = node.graph.call_function(mul,
                                            kwargs={
                                                "input": new_sigmoid_node,
                                                "other": input_node
                                            })
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "size"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
def tensor_size_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Mapping from Tensor.size() to acc_ops.size. We map size() to acc_ops.size directly
    and map size(dim) to acc_ops.size + acc_ops.getitem.
    """

    with node.graph.inserting_before(node):
        size_node = node.graph.call_function(
            size, kwargs={"input": node.kwargs["input"]})

        if "dim" not in node.kwargs:
            size_node.meta = node.meta.copy()
            return size_node

        size_node.meta["type"] = torch.Size
        getitem_node = node.graph.call_function(getitem,
                                                kwargs={
                                                    "input": size_node,
                                                    "idx": node.kwargs["dim"]
                                                })
        getitem_node.meta = node.meta.copy()
        return getitem_node


@register_acc_op
def device(*, input):
    return input.device


@register_acc_op
def dtype(*, input):
    return input.dtype


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", getattr),
    arg_replacement_tuples=[],
)
def custom_getattr_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Custom function for mapping a call_function getattr to other ops.
    Supports:
    * getattr on a torch.Tensor with "shape", "device", or "dtype" attributes
    * getattr for accessing named tuples
    """
    # Have to use args here since getattr forces positional args.
    input_obj = node.args[0]
    attr_name = node.args[1]
    assert isinstance(input_obj, torch.fx.Node)
    input_obj_type = input_obj.meta["type"]

    # Handle named tuple access. NamedTupleMeta and the namedtuple factory function
    # create a subclass of tuple with an extra _fields attribute.
    if issubclass(input_obj_type, tuple) and hasattr(input_obj_type,
                                                     "_fields"):
        idx = None
        for i, name in enumerate(input_obj_type._fields):
            if name == attr_name:
                idx = i
                break
        assert (
            idx is not None
        ), f"Named tuple type {input_obj_type} does not have field {name}"

        with node.graph.inserting_before(node):
            getitem_node = node.graph.call_function(getitem,
                                                    kwargs={
                                                        "input": input_obj,
                                                        "idx": idx
                                                    })
            getitem_node.meta = node.meta.copy()
            return getitem_node

    assert (input_obj_type == torch.Tensor
            ), f"Expected torch.Tensor type for {input_obj_type}"
    assert (
        attr_name == "shape" or attr_name == "device" or attr_name == "dtype"
    ), f"Only supporting shape, device and dtype getattr for now, not {attr_name}"
    if attr_name == "shape":
        func = size
    elif attr_name == "device":
        func = device
    elif attr_name == "dtype":
        func = dtype
    with node.graph.inserting_before(node):
        size_node = node.graph.call_function(func, kwargs={"input": input_obj})
        size_node.meta = node.meta.copy()
        return size_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "reshape"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "shape"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "view"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "shape"),
    ],
)
def custom_tensor_reshape_mapper(node: torch.fx.Node,
                                 _: nn.Module) -> torch.fx.Node:
    """
    For Tensor.reshape and Tensor.view nodes, args could be (input, 1, 2, 3) or (input,
    (1, 2, 3)).  Here we do some special handling with the `shape` arg in order to map
    it to acc_ops.reshape. It also handles the case when `shape` is a list instead of
    tuple.
    """
    input_node = node.kwargs["input"]
    shape = node.kwargs["shape"]

    assert isinstance(shape, Sequence)
    if isinstance(shape[0], (tuple, list)):  # type: ignore[index]
        shape = shape[0]  # type: ignore[index]

    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(
            reshape,
            kwargs={
                "input": input_node,
                "shape": shape,
            },
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.transpose),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim0", "dim0"),
        ("dim1", "dim1"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "transpose"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim0", "dim0"),
        ("dim1", "dim1"),
    ],
)
def transpose_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    # Get the dim-permutation/shuffle

    rank = node.meta['tensor_rank']
    shuffle = list(range(rank))
    dim0 = cast(int, node.kwargs["dim0"])
    dim1 = cast(int, node.kwargs["dim1"])
    shuffle[dim0] = dim1
    shuffle[dim1] = dim0

    # Create the new acc_ops.permute node. Update all uses of the transpose
    # node and then delete the transpose node.
    with node.graph.inserting_after(node):
        permute_node = node.graph.call_function(
            the_function=permute,
            kwargs={
                "input": node.kwargs.get("input"),
                "permutation": shuffle,
            },
        )
        permute_node.meta = node.meta.copy()
        node.replace_all_uses_with(permute_node)

    permute_node.graph.erase_node(node)
    return permute_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.t),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "t"),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def t_mapper(node: torch.fx.Node, _: nn.Module):
    ranks = node.meta["tensor_rank"]
    shuffle = [1, 0] if (ranks > 1) else [0]

    with node.graph.inserting_before(node):
        new_node = node.graph.create_node(
            "call_function",
            permute,
            kwargs={
                "input": node.kwargs["input"],
                "permutation": shuffle
            },
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_mapping(op_and_target=("call_function",
                                        torch.nn.functional.pad))
@register_acc_op
def pad(*, input, pad: List[int], mode: str, value: float):
    return torch.nn.functional.pad(input=input,
                                   pad=pad,
                                   mode=mode,
                                   value=value)


@register_acc_op_mapping(
    op_and_target=("call_method", "new_zeros"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size"),
    ],
)
@register_acc_op
def new_zeros(*, input, size):
    return input.new_zeros(size)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def slice_tensor(*, input, dim, start, stop, step):
    slc = slice(start, stop, step)
    if dim >= 0:
        slices: List[slice] = [slice(None, None, None) for _ in range(dim)]
        slices.append(slc)
    else:
        slices = [Ellipsis, slc]  # type: ignore[list-item]
        slices.extend([slice(None, None, None) for _ in range(-dim - 1)])

    return input[tuple(slices)]


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.narrow),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("start", "start"),
        ("length", "length"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "narrow"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("start", "start"),
        ("length", "length"),
    ],
)
def custom_narrow_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    assert isinstance(node.kwargs["start"], int) and isinstance(
        node.kwargs["length"], int)

    dim = node.kwargs["dim"]
    start = node.kwargs["start"]
    stop = node.kwargs["start"] + node.kwargs["length"]

    slc = slice(start, stop, 1)

    if dim >= 0:
        slices: List[slice] = [slice(None, None, None) for _ in range(dim)]
        slices.append(slc)
    else:
        slices = [Ellipsis, slc]  # type: ignore[list-item]
        slices.extend([slice(None, None, None) for _ in range(-dim - 1)])

    kwargs = {"input": node.kwargs["input"], "idx": tuple(slices)}

    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(getitem, kwargs=kwargs)

    new_node.meta = node.meta.copy()
    return new_node


@register_acc_op
def tuple_construct(*, tensors):
    return tuple(tensors)


@register_custom_acc_mapper_fn(
    op_and_target=("call_module", torch.nn.LSTM),
    arg_replacement_tuples=[
        ("input", "input"),
        ('hidden_states', 'hx', this_arg_is_optional),
    ],
)
def lstm_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    hx = None if 'hx' not in node.kwargs else node.kwargs['hx']
    with node.graph.inserting_before(node):
        new_node = node.graph.call_module(node.target,
                                          kwargs={
                                              'input': node.kwargs['input'],
                                              'hx': hx,
                                          })
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_mapping(
    op_and_target=("call_function", torch.as_strided),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size"),
        ("stride", "stride"),
        ("storage_offset", "storage_offset", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "as_strided"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size"),
        ("stride", "stride"),
        ("storage_offset", "storage_offset", this_arg_is_optional),
    ],
)
@register_acc_op
def as_strided(*, input, size, stride, storage_offset=0):
    return torch.as_strided(input=input,
                            size=size,
                            stride=stride,
                            storage_offset=storage_offset)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.quantize_per_tensor),
    arg_replacement_tuples=[
        ("input", "input"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
        ("dtype", "dtype"),
    ],
)
@register_acc_op
def quantize_per_tensor(*, input, scale, zero_point, dtype):
    return torch.quantize_per_tensor(input, scale, zero_point, dtype)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "dequantize"))
@register_acc_op_mapping(op_and_target=("call_function", torch.dequantize))
@register_acc_op
def dequantize(*, input):
    return torch.dequantize(input)


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(op_and_target=("call_function",
                                        torch.ops.quantized.add),
                         arg_replacement_tuples=[
                             ("qa", "input"),
                             ("qb", "other"),
                             ("scale", "scale"),
                             ("zero_point", "zero_point"),
                         ])
@register_acc_op
def quantized_add(*, input, other, scale, zero_point):
    return torch.ops.quantized.add(input, other, scale, zero_point)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.ops.quantized.add_relu),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
    ],
)
def add_relu_unfuse_mapper(node: torch.fx.Node,
                           mod: torch.fx.GraphModule) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        add_kwargs = {
            "input": node.kwargs["input"],
            "other": node.kwargs["other"],
            "scale": node.kwargs["scale"],
            "zero_point": node.kwargs["zero_point"],
        }
        add_node = node.graph.call_function(quantized_add, kwargs=add_kwargs)
        add_node.meta = node.meta.copy()

        relu_node = node.graph.call_function(relu,
                                             kwargs={
                                                 "input": add_node,
                                                 "inplace": False
                                             })
        relu_node.meta = node.meta
        return relu_node


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.ne))
@register_acc_op_mapping(op_and_target=("call_function", operator.ne))
@register_acc_op_mapping(op_and_target=("call_method", "ne"))
@register_acc_op
def ne(*, input, other):
    return operator.ne(input, other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.eq))
@register_acc_op_mapping(op_and_target=("call_function", operator.eq))
@register_acc_op_mapping(op_and_target=("call_method", "eq"))
@register_acc_op
def eq(*, input, other):
    return operator.eq(input, other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.gt))
@register_acc_op_mapping(op_and_target=("call_function", operator.gt))
@register_acc_op_mapping(op_and_target=("call_method", "gt"))
@register_acc_op
def gt(*, input, other):
    return operator.gt(input, other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.lt))
@register_acc_op_mapping(op_and_target=("call_function", operator.lt))
@register_acc_op_mapping(op_and_target=("call_method", "lt"))
@register_acc_op
def lt(*, input, other):
    return operator.lt(input, other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.ge))
@register_acc_op_mapping(op_and_target=("call_function", operator.ge))
@register_acc_op_mapping(op_and_target=("call_method", "ge"))
@register_acc_op
def ge(*, input, other):
    return operator.ge(input, other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.le))
@register_acc_op_mapping(op_and_target=("call_function", operator.le))
@register_acc_op_mapping(op_and_target=("call_method", "le"))
@register_acc_op
def le(*, input, other):
    return operator.le(input, other)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.isinf))
@register_acc_op
def isinf(*, input):
    return torch.isinf(input=input)


@register_acc_op_mapping(
    op_and_target=("call_method", "any"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.any),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_acc_op
def any(*, input, dim=None, keepdim=False):
    if dim is not None:
        return torch.any(input, dim=dim, keepdim=keepdim)
    return input.any()


@register_acc_op_mapping(
    op_and_target=("call_method", "all"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.all),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_acc_op
def all(*, input, dim=None, keepdim=False):
    if dim is not None:
        return torch.all(input, dim=dim, keepdim=keepdim)
    return input.all()


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.isnan))
@register_acc_op
def isnan(*, input):
    return torch.isnan(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.nan_to_num))
@register_acc_op
def nan_to_num(*, input, nan=0.0, posinf=None, neginf=None):
    return torch.nan_to_num(input=input, nan=nan, posinf=posinf, neginf=neginf)


# @register_acc_op_mapping(op_and_target=("call_function", nn.functional.scaled_dot_product_attention))
@register_acc_op_mapping(
    op_and_target=("call_function", nn.functional.scaled_dot_product_attention),
    arg_replacement_tuples=[
        ("query", "query"),
        ("key", "key"),
        ("value", "value"),
        ("attn_mask", "attn_mask", this_arg_is_optional),
        ("dropout_p", "dropout_p", this_arg_is_optional),
        ("is_causal", "is_causal", this_arg_is_optional),
        ("scale", "scale", this_arg_is_optional),
    ],
)
@register_acc_op
def scaled_dot_product_attention(*, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    return nn.functional.scaled_dot_product_attention(query=query,
                                                      key=key,
                                                      value=value,
                                                      attn_mask=attn_mask,
                                                      dropout_p=dropout_p,
                                                      is_causal=is_causal,
                                                      scale=scale)


@register_acc_op_mapping(op_and_target=("call_function", torch.erf))
@register_acc_op_mapping(op_and_target=("call_function", torch.special.erf))
@register_acc_op
def erf(*, input):
    return torch.erf(input=input)

