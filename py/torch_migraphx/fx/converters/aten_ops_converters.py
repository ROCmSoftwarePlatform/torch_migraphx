#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
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

from typing import cast, Iterable, List, Sequence
import logging

import migraphx
import torch
from ..converter_registry import migraphx_converter
from torch_migraphx.fx.converters import acc_ops_converters
from .utils import *
from ..utils import torch_dtype_to_mgx_enum
from ..mgx_module import MGXInstruction

_LOGGER = logging.getLogger(__name__)


@migraphx_converter(torch.ops.aten.lift_fresh_copy.default)
@migraphx_converter(torch.ops.aten.lift_fresh.default)
@migraphx_converter(torch.ops.aten.lift.default)
@migraphx_converter(torch.ops.aten._to_copy.default)
@migraphx_converter(torch.ops.aten.clone.default)
@migraphx_converter(torch.ops.aten.copy.default)
@migraphx_converter(torch.ops.aten.detach.default)
def aten_ops_to_copy(mgx_module, node, args, kwargs):
    if node.target == torch.ops.aten.copy.default:
        assert len(args) == 2
        out = args[1]
    else:
        assert len(args) == 1
        out = args[0]
    if "dtype" in kwargs:
        assert not out.is_quantized()
        out = mgx_module.add_instruction(
            migraphx.op("convert",
                        target_type=torch_dtype_to_mgx_enum(kwargs["dtype"])),
            [out.instr_ref])
        out = MGXInstruction(out)

    return out


@migraphx_converter(torch.ops.aten.view.default)
@migraphx_converter(torch.ops.aten._unsafe_view.default)
@migraphx_converter(torch.ops.aten.reshape)
@migraphx_converter(torch.ops.aten.reshape.default)
def aten_ops_view(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, shape = args[0], args[1]
    inp_size = inp.shape().lens()

    if len(inp_size) == 1 and inp_size[0] == 1 and not shape:
        return inp

    acc_kwargs = {"input": inp, "shape": shape}
    return acc_ops_converters.acc_ops_reshape(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.unsqueeze.default)
def aten_ops_unsqueeze(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "dim": args[1]}
    return acc_ops_converters.acc_ops_unsqueeze(mgx_module, node, (),
                                                acc_kwargs)


@migraphx_converter(torch.ops.aten.squeeze.dim)
@migraphx_converter(torch.ops.aten.squeeze.default)
def aten_ops_squeeze(mgx_module, node, args, kwargs):
    assert len(args) >= 1
    acc_kwargs = {"input": args[0]}

    if len(args) == 2:
        acc_kwargs["dim"] = args[1]
        
    return acc_ops_converters.acc_ops_squeeze(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.repeat.default)
def aten_ops_repeat(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "repeats": args[1]}
    return acc_ops_converters.acc_ops_repeat(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.log2.default)
def aten_ops_log2(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_log2(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.log.default)
def aten_ops_log(mgx_module, node, args, _kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_log(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.topk.default)
def aten_ops_topk(mgx_module, node, args, kwargs):
    assert len(args) >= 2
    acc_kwargs = {
        "input": args[0], 
        "k": args[1], 
        "dim": args[2] if len(args) > 2 else -1,
        "largest": args[3] if len(args) > 3 else True, 
        "sorted": args[4] if len(args) > 4 else True,
    }
    return acc_ops_converters.acc_ops_topk(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.expand.default)
def aten_ops_expand(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "sizes": args[1]}
    return acc_ops_converters.acc_ops_expand_tensor(mgx_module, node, (),
                                                    acc_kwargs)


@migraphx_converter(torch.ops.aten.where.self)
def aten_ops_where(mgx_module, node, args, kwargs):
    assert len(args) == 3
    acc_kwargs = {"condition": args[0], "input": args[1], "other": args[2]}
    return acc_ops_converters.acc_ops_where(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.masked_fill.Scalar)
@migraphx_converter(torch.ops.aten.masked_fill.Tensor)
def aten_ops_masked_fill(mgx_module, node, args, kwargs):
    assert len(args) == 3

    acc_kwargs = {"input": args[0], "mask": args[1], "value": args[2]}
    return acc_ops_converters.acc_ops_masked_fill(mgx_module, node, (),
                                                  acc_kwargs)


@migraphx_converter(torch.ops.aten.fill.Scalar)
@migraphx_converter(torch.ops.aten.fill.Tensor)
def aten_ops_fill(mgx_module, node, args, kwargs):
    assert len(args) == 2

    mask_mgx = MGXInstruction(mgx_module.add_literal(torch.tensor(True).numpy()))
    acc_kwargs = {"input": args[0], "mask": mask_mgx, "value": args[1]}
    return acc_ops_converters.acc_ops_masked_fill(mgx_module, node, (),
                                                  acc_kwargs)


@migraphx_converter(torch.ops.aten.slice_scatter.default)
def aten_ops_slice_scatter(mgx_module, node, args, kwargs):
    assert len(args) >= 2
    acc_kwargs = {
        "input": args[0],
        "src": args[1],
        "dim": args[2] if len(args) >= 3 else 0,
        "start": args[3] if len(args) >= 4 else None,
        "end": args[4] if len(args) >= 5 else None,
        "step": args[5] if len(args) >= 6 else 1,
    }

    return acc_ops_converters.acc_ops_slice_scatter(mgx_module, node, (),
                                                    acc_kwargs)


@migraphx_converter(torch.ops.aten.select_scatter.default)
def aten_ops_select_scatter(mgx_module, node, args, kwargs):
    assert len(args) == 4
    acc_kwargs = {
        "input": args[0],
        "src": args[1],
        "dim": args[2],
        "index": args[3]
    }

    return acc_ops_converters.acc_ops_select_scatter(mgx_module, node, (),
                                                     acc_kwargs)


@migraphx_converter(torch.ops.aten.index_select.default)
def aten_ops_index_select(mgx_module, node, args, kwargs):
    assert len(args) == 3
    acc_kwargs = {"input": args[0], "dim": args[1], "index": args[2]}

    return acc_ops_converters.acc_ops_index_select(mgx_module, node, (),
                                                   acc_kwargs)

  
@migraphx_converter(torch.ops.aten.scatter_add.default)
def aten_ops_scatter_add(mgx_module, node, args, kwargs):
    assert len(args) == 4
    acc_kwargs = {
        "input": args[0],
        "dim": args[1],
        "index": args[2],
        "src": args[3],
        "reduce": "sum",
        "include_self": True
    }

    return acc_ops_converters.acc_ops_scatter_reduce(mgx_module, node, (),
                                                     acc_kwargs)

@migraphx_converter(torch.ops.aten.scatter_reduce.two)
def aten_ops_scatter_reduce(mgx_module, node, args, kwargs):
    assert len(args) == 5
    acc_kwargs = {
        "input": args[0],
        "dim": args[1],
        "index": args[2],
        "src": args[3],
        "reduce": args[4],
        "include_self": True
    }
    
    if "include_self" in kwargs:
        acc_kwargs["include_self"] = kwargs["include_self"]

    return acc_ops_converters.acc_ops_scatter_reduce(mgx_module, node, (),
                                                     acc_kwargs)


@migraphx_converter(torch.ops.aten.maximum.default)
def aten_ops_maximum(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "other": args[1]}
    return acc_ops_converters.acc_ops_maximum(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.minimum.default)
def aten_ops_minimum(mgx_module, node, args, _kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "other": args[1]}
    return acc_ops_converters.acc_ops_minimum(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.permute.default)
def aten_ops_permute(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "permutation": args[1]}
    return acc_ops_converters.acc_ops_permute(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.select.int)
def aten_ops_select(mgx_module, node, args, kwargs):
    assert len(args) == 3
    inp = args[0]
    dim = args[1]
    index = args[2]
    inp_size = inp.shape().lens()
    slices = [slice(None, None, None) for _ in inp_size]
    slices[dim] = index

    acc_kwargs = {"input": inp, "idx": slices}
    return acc_ops_converters.acc_ops_getitem(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.slice.Tensor)
def aten_ops_slice(mgx_module, node, args, kwargs):
    assert len(args) >= 1
    inp = args[0]
    dim = args[1] if len(args) >= 2 else 0
    start = args[2] if len(args) >= 3 else None
    end = args[3] if len(args) >= 4 else None
    step = args[4] if len(args) == 5 else None

    inp_size = inp.shape().lens()
    end = min(inp_size[dim], end) if end else end
    step = 1 if not step else step

    if (start == 0 and end == inp_size[dim] and step == 1):
        return inp

    slices = [slice(None, None, None) for _ in inp_size]
    slices[dim] = start if not end and (step == 1) else slice(start, end, step)

    acc_kwargs = {"input": inp, "idx": slices}
    return acc_ops_converters.acc_ops_getitem(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.index.Tensor)
@migraphx_converter(torch.ops.aten._unsafe_index.Tensor)
def aten_ops_index(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp = args[0]
    # None arguments in this op are treated differently than in __getitem__
    idx = [i if i is not None else slice(None, None, None) for i in args[1]]
    acc_kwargs = {"input": inp, "idx": idx}
    return acc_ops_converters.acc_ops_getitem(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.cat.default)
def aten_ops_cat(mgx_module, node, args, kwargs):
    assert len(args) >= 1
    acc_kwargs = {"tensors": args[0], "dim": args[1] if len(args) == 2 else 0}
    return acc_ops_converters.acc_ops_cat(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.split.Tensor)
def aten_ops_split(mgx_module, node, args, kwargs):
    assert len(args) >= 2
    inp = args[0]
    split_size_or_sections = args[1]
    dim = args[2] if len(args) == 3 else 0

    if isinstance(split_size_or_sections, int):
        acc_kwargs = {
            "input": inp,
            "split_size": split_size_or_sections,
            "dim": dim
        }
        return acc_ops_converters.acc_ops_split(mgx_module, node, (),
                                                acc_kwargs)

    assert isinstance(split_size_or_sections, (list, tuple))
    start = 0
    slice_nodes = []
    for i in split_size_or_sections:
        assert isinstance(i, int)
        stop = start + i
        slc = slice(start, stop, 1)

        if dim >= 0:
            slices = [slice(None, None, None) for _ in range(dim)]
            slices.append(slc)
        else:
            slices = [Ellipsis, slc]
            slices.extend([slice(None, None, None) for _ in range(-dim - 1)])

        acc_kwargs = {"input": inp, "idx": slices}
        slice_nodes.append(
            acc_ops_converters.acc_ops_getitem(mgx_module, node, (),
                                               acc_kwargs))
        start += i

    return slice_nodes


@migraphx_converter(torch.ops.aten.hardtanh.default)
@migraphx_converter(torch.ops.aten.clamp.default)
@migraphx_converter(torch.ops.aten.clamp.Tensor)
@migraphx_converter(torch.ops.aten.clamp_min.default)
@migraphx_converter(torch.ops.aten.clamp_min.Tensor)
@migraphx_converter(torch.ops.aten.clamp_max.default)
@migraphx_converter(torch.ops.aten.clamp_max.Tensor)
def aten_ops_clamp(mgx_module, node, args, _kwargs):
    assert len(args) >= 1
    acc_kwargs = {
        "input": args[0],
        "min": None,
        "max": None
    }
    if node.target == torch.ops.aten.hardtanh.default:
        acc_kwargs["min"] = args[1] if len(args) >= 2 else -1
        acc_kwargs["max"] = args[2] if len(args) == 3 else 1
    elif node.target in {torch.ops.aten.clamp_min.default, torch.ops.aten.clamp_min.Tensor}:
        acc_kwargs["min"] = args[1]
        acc_kwargs["max"] = None
    elif node.target in {torch.ops.aten.clamp_max.default, torch.ops.aten.clamp_max.Tensor}:
        acc_kwargs["min"] = None
        acc_kwargs["max"] = args[1]
    else:
        acc_kwargs["min"] = args[1] if len(args) >= 2 else None
        acc_kwargs["max"] = args[2] if len(args) == 3 else None
    return acc_ops_converters.acc_ops_clamp(mgx_module, node, (), acc_kwargs)

@migraphx_converter(torch.ops.torchvision.roi_align.default)
def aten_ops_roi_align(mgx_module, node, args, _kwargs):
    assert len(args) == 7
    acc_kwargs = {
	        "input": args[0],
	        "boxes": args[1],
	        "spatial_scale": args[2],
	        "output_size": [args[3], args[4]],
	        "sampling_ratio": args[5],
	        "aligned": args[6]
	    }          
    return acc_ops_converters.acc_ops_roi_align(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.nll_loss_forward.default)
def aten_ops_nll_loss_forward(mgx_module, node, args, _kwargs):
    assert len(args) == 5

    reduction_dict = {torch.nn.functional._Reduction.get_enum(k):k for k in ["none", "sum", "mean"]}
    reduction = reduction_dict[args[3]]

    target_lens = args[1].instr_ref.shape().lens()
    ignore_index = args[4]

    acc_kwargs = {
        "input": args[0],
        "target": args[1],
        "weight": args[2],
        "reduction": reduction,
        "ignore_index": ignore_index,
        "weight_sum": True
    }
    # Use "weight_sum" parameter to flag acc converter to return a tuple. Ideally, it would
    # be nice to have the computation for that here, but that would mean a lot of repeated
    # computations
    
    # The acc op returns one tensor, while the aten op returns two tensors
    return acc_ops_converters.acc_ops_nll_loss(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.relu.default)
def aten_ops_relu(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_relu(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.tanh.default)
def aten_ops_tanh(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_tanh(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.elu.default)
def aten_ops_elu(mgx_module, node, args, kwargs):
    assert len(args) >= 1
    inp = args[0]
    alpha = 1.0 if len(args) < 2 else args[1]

    acc_kwargs = {'input': inp, 'alpha': alpha}
    return acc_ops_converters.acc_ops_elu(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.leaky_relu.default)
def aten_ops_leaky_relu(mgx_module, node, args, kwargs):
    assert len(args) >= 1
    inp = args[0]
    neg_slope = 0.01 if len(args) < 2 else args[1]

    acc_kwargs = {'input': inp, 'negative_slope': neg_slope}
    return acc_ops_converters.acc_ops_leaky_relu(mgx_module, node, (),
                                                 acc_kwargs)


@migraphx_converter(torch.ops.aten.hardswish.default)
def aten_ops_hardswish(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    hard_sig = acc_ops_converters.acc_ops_hard_sigmoid(mgx_module, node, (),
                                                       acc_kwargs)

    mul_kwargs = {"input": args[0], "other": hard_sig}
    return acc_ops_converters.acc_ops_mul(mgx_module, node, (), mul_kwargs)


@migraphx_converter(torch.ops.aten.hardsigmoid.default)
def aten_ops_hardsigmoid(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_hard_sigmoid(mgx_module, node, (),
                                                   acc_kwargs)


@migraphx_converter(torch.ops.aten.sigmoid.default)
def aten_ops_sigmoid(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_sigmoid(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.glu.default)
def aten_ops_glu(mgx_module, node, args, kwargs):
    assert len(args) >= 1
    acc_kwargs = {"input": args[0], "dim": args[1] if len(args) >= 2 else -1}

    return acc_ops_converters.acc_ops_glu(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.gelu.default)
def aten_ops_gelu(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_gelu(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.silu.default)
def aten_ops_silu(mgx_module, node, args, kwargs):
    assert len(args) == 1
    inp = args[0]
    sig_kwargs = {"input": inp}
    sig = acc_ops_converters.acc_ops_sigmoid(mgx_module, node, (), sig_kwargs)

    mul_kwargs = {"input": inp, "other": sig}
    return acc_ops_converters.acc_ops_mul(mgx_module, node, (), mul_kwargs)


# Ignores half_to_float input
@migraphx_converter(torch.ops.aten._softmax.default)
def aten_ops_softmax(mgx_module, node, args, kwargs):
    assert len(args) == 3
    acc_kwargs = {"input": args[0], "dim": args[1]}

    return acc_ops_converters.acc_ops_softmax(mgx_module, node, (), acc_kwargs)


# Ignores half_to_float input
@migraphx_converter(torch.ops.aten._log_softmax.default)
def aten_ops_log_softmax(mgx_module, node, args, _kwargs):
    assert len(args) == 3
    acc_kwargs = {"input": args[0], "dim": args[1]}

    return acc_ops_converters.acc_ops_log_softmax(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.logical_not.default)
def aten_ops_logical_not(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.logical_not(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.reciprocal.default)
def aten_ops_reciprocal(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_reciprocal(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.sqrt.default)
def aten_ops_sqrt(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_sqrt(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.rsqrt.default)
def aten_ops_rsqrt(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_rsqrt(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.sin.default)
def aten_ops_sin(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_sin(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.cos.default)
def aten_ops_cos(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_cos(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.exp.default)
def aten_ops_exp(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_exp(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.bmm.default)
@migraphx_converter(torch.ops.aten.mm.default)
def aten_ops_bmm(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "other": args[1]}
    return acc_ops_converters.acc_ops_matmul(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.addmm.default)
def aten_ops_addmm(mgx_module, node, args, kwargs):
    assert len(args) == 3
    inp, mat1, mat2 = args
    beta = kwargs["beta"] if "beta" in kwargs else 1
    alpha = kwargs["alpha"] if "alpha" in kwargs else 1

    mm_kwargs = {"input": mat1, "other": mat2}
    mm_out = acc_ops_converters.acc_ops_matmul(mgx_module, node, (), mm_kwargs)

    if alpha != 1:
        mul_kwargs = {"input": mm_out, "other": alpha}
        mm_out = acc_ops_converters.acc_ops_mul(mgx_module, node, (),
                                                mul_kwargs)

    if beta != 1:
        mul_kwargs = {"input": inp, "other": beta}
        inp = acc_ops_converters.acc_ops_mul(mgx_module, node, (), mul_kwargs)

    add_kwargs = {"input": inp, "other": mm_out}
    return acc_ops_converters.acc_ops_add(mgx_module, node, (), add_kwargs)


@migraphx_converter(torch.ops.aten.addcmul.default)
def aten_ops_addcmul(mgx_module, node, args, kwargs):
    assert len(args) == 3
    inp, mat1, mat2 = args
    value = kwargs["value"] if "value" in kwargs else 1

    mul_kwargs = {"input": mat1, "other": mat2}
    mul_out = acc_ops_converters.acc_ops_mul(mgx_module, node, (), mul_kwargs)

    if value != 1:
        mul_kwargs = {"input": mul_out, "other": value}
        mul_out = acc_ops_converters.acc_ops_mul(mgx_module, node, (),
                                                 mul_kwargs)

    add_kwargs = {"input": inp, "other": mul_out}
    return acc_ops_converters.acc_ops_add(mgx_module, node, (), add_kwargs)


@migraphx_converter(torch.ops.aten.add.Scalar)
@migraphx_converter(torch.ops.aten.add.Tensor)
def aten_ops_add(mgx_module, node, args, kwargs):
    assert len(args) >= 2
    inp, other = args[0], args[1]

    if node.target == torch.ops.aten.add.Scalar:
        alpha = args[2] if len(args) == 3 else 1
    else:
        alpha = kwargs["alpha"] if "alpha" in kwargs else 1

    if alpha != 1:
        mul_kwargs = mul_kwargs = {"input": other, "other": alpha}
        other = acc_ops_converters.acc_ops_mul(mgx_module, node, (),
                                               mul_kwargs)

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_add(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.sub.Scalar)
@migraphx_converter(torch.ops.aten.sub.Tensor)
def aten_ops_sub(mgx_module, node, args, kwargs):
    assert len(args) >= 2
    inp, other = args[0], args[1]

    if node.target == torch.ops.aten.sub.Scalar:
        alpha = args[2] if len(args) == 3 else 1
    else:
        alpha = kwargs["alpha"] if "alpha" in kwargs else 1

    if alpha != 1:
        mul_kwargs = mul_kwargs = {"input": other, "other": alpha}
        other = acc_ops_converters.acc_ops_mul(mgx_module, node, (),
                                               mul_kwargs)

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_sub(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.rsub.Scalar)
@migraphx_converter(torch.ops.aten.rsub.Tensor)
def aten_ops_rsub(mgx_module, node, args, kwargs):
    args = list(args)
    args[0], args[1] = args[1], args[0]
    return aten_ops_sub(mgx_module, node, tuple(args), kwargs)


@migraphx_converter(torch.ops.aten.mul.Scalar)
@migraphx_converter(torch.ops.aten.mul.Tensor)
def aten_ops_mul(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, other = args[0], args[1]

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_mul(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.div.Scalar)
@migraphx_converter(torch.ops.aten.div.Tensor)
@migraphx_converter(torch.ops.aten.div.Tensor_mode)
def aten_ops_div(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, other = args[0], args[1]

    acc_kwargs = {"input": inp, "other": other}
    
    if "rounding_mode" in kwargs.keys():
        acc_kwargs["rounding_mode"] = kwargs["rounding_mode"]

        if acc_kwargs["rounding_mode"] == "floor":
            return acc_ops_converters.acc_ops_floor_div(mgx_module, node, (), acc_kwargs)
        elif acc_kwargs["rounding_mode"] == "trunc":
            return acc_ops_converters.acc_ops_trunc_div(mgx_module, node, (), acc_kwargs)
        elif acc_kwargs["rounding_mode"] is None:
            pass
        else:
            raise ValueError("Rounding Mode must in [floor, trunc]")
        
    return acc_ops_converters.acc_ops_div(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.pow.Tensor_Scalar)
@migraphx_converter(torch.ops.aten.pow.Tensor_Tensor)
def aten_ops_pow(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, exp = args[0], args[1]

    acc_kwargs = {"input": inp, "exponent": exp}
    return acc_ops_converters.acc_ops_pow(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.batch_norm.default)
@migraphx_converter(torch.ops.aten.miopen_batch_norm.default)
@migraphx_converter(
    torch.ops.aten._native_batch_norm_legit_no_training.default)
def aten_ops_batch_norm(mgx_module, node, args, kwargs):
    assert len(args) >= 7

    acc_kwargs = {
        "input": args[0],
        "weight": args[1],
        "bias": args[2],
        "running_mean": args[3],
        "running_var": args[4],
    }
    if node.target == torch.ops.aten._native_batch_norm_legit_no_training.default:
        acc_kwargs["momentum"], acc_kwargs["eps"] = args[5], args[6]
    else:
        acc_kwargs["momentum"], acc_kwargs["eps"] = args[6], args[7]

    bn = acc_ops_converters.acc_ops_batch_norm(mgx_module, node, (),
                                               acc_kwargs)

    return bn, acc_kwargs["running_mean"], acc_kwargs["running_var"]


@migraphx_converter(torch.ops.aten.native_group_norm.default)
def aten_ops_group_norm(mgx_module, node, args, kwargs):
    assert len(args) == 8

    acc_kwargs = {
        "input": args[0],
        "weight": args[1],
        "bias": args[2],
        "num_groups": args[6],
        "eps": args[7],
    }

    return acc_ops_converters.acc_ops_group_norm(mgx_module, node, (),
                                                 acc_kwargs), None, None
    

@migraphx_converter(torch.ops.aten.linalg_vector_norm.default)
def aten_ops_linalg_vector_norm(mgx_module, node, args, kwargs):
    assert len(args) >= 1

    acc_kwargs = {
        "input": args[0],
        "ord": args[1] if len(args) >= 2 else 2,
        "dim": args[2] if len(args) >= 3 else None,
        "keepdim": args[3] if len(args) >= 4 else False,
    }
    
    # Aten op decelaration:
    # aten::linalg_vector_norm(Tensor self, 
    #                          Scalar ord=2, 
    #                          int[1]? dim=None, 
    #                          bool keepdim=False, 
    #                          *, 
    #                          ScalarType? dtype=None) -> Tensor
    # For dim the aten type is: int[1]? dim=None. Unfold int[1] to scalar
    if isinstance(acc_kwargs["dim"], (list, tuple)):
        assert len(acc_kwargs["dim"]) == 1
        acc_kwargs["dim"] = acc_kwargs["dim"][0]

    return acc_ops_converters.acc_ops_linalg_vector_norm(
        mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.convolution.default)
def aten_ops_convolution(mgx_module, node, args, kwargs):
    assert len(args) == 9

    acc_kwargs = {
        "input": args[0],
        "weight": args[1],
        "bias": args[2],
        "stride": args[3],
        "padding": args[4],
        "dilation": args[5],
        "transposed": args[6],
        "output_padding": args[7],
        "groups": args[8],
    }

    if acc_kwargs["transposed"]:
        raise RuntimeError("'transposed' parameter not supported.")

    if not all(i == 0 for i in acc_kwargs["output_padding"]):
        raise RuntimeError(
            "non-zero values for 'output_padding' not supported")

    return acc_ops_converters.acc_ops_convnd(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.t.default)
def aten_ops_t(mgx_module, node, args, kwargs):
    assert len(args) == 1

    in_shape = args[0].shape().lens()
    if len(in_shape) != 2:
        raise RuntimeError(f"aten.t expects a 2D input shape, got {in_shape}")

    acc_kwargs = {"input": args[0], "permutation": [1, 0]}
    return acc_ops_converters.acc_ops_permute(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.transpose.int)
def aten_ops_transpose(mgx_module, node, args, kwargs):
    assert len(args) == 3
    inp, dim1, dim2 = args[0], args[1], args[2]
    perm = list(range(len(inp.shape().lens())))
    perm[dim1] = dim2
    perm[dim2] = dim1

    acc_kwargs = acc_kwargs = {"input": inp, "permutation": perm}
    return acc_ops_converters.acc_ops_permute(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.constant_pad_nd.default)
def aten_ops_constant_pad(mgx_module, node, args, kwargs):
    assert len(args) >= 2
    inp, pad = args[0], args[1]
    value = 0 if len(args) < 3 else args[2]

    acc_kwargs = {"input": inp, "pad": pad, "mode": "constant", "value": value}
    return acc_ops_converters.acc_ops_pad(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.unbind.int)
def aten_ops_unbind(mgx_module, node, args, kwargs):
    assert len(args) >= 1
    inp = args[0]
    dim = 0 if len(args) < 2 else args[1]

    acc_kwargs = {"input": inp, "dim": dim}
    return acc_ops_converters.acc_ops_unbind(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.split_with_sizes.default)
def aten_ops_split_with_sizes(mgx_module, node, args, kwargs):
    assert len(args) >= 2
    inp, split_sizes = args[0], args[1]
    dim = 0 if len(args) < 3 else args[2]
    in_shape = inp.shape().lens()

    start = 0
    outs = []
    for i in split_sizes:
        slices = [slice(None, None, None) for _ in in_shape]
        end = start + i if start + i <= in_shape[dim] else in_shape[dim]
        slices[dim] = slice(start, end, 1)
        outs.append(
            acc_ops_converters.acc_ops_getitem(mgx_module,
                                               node, (),
                                               kwargs={
                                                   'input': inp,
                                                   'idx': slices
                                               }))

        start += i

    return tuple(outs)


@migraphx_converter(torch.ops.aten.sum.dim_IntList)
def aten_ops_sum(mgx_module, node, args, kwargs):
    assert len(args) >= 2

    acc_kwargs = {
        "input": args[0],
        "dim": args[1],
        "keepdim": args[2] if len(args) == 3 else False
    }

    return acc_ops_converters.acc_ops_sum(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.cumsum.default)
def aten_ops_cumsum(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "dim": args[1]}

    return acc_ops_converters.acc_ops_cumsum(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.mean.dim)
def aten_ops_mean(mgx_module, node, args, kwargs):
    assert len(args) >= 2

    acc_kwargs = {
        "input": args[0],
        "dim": args[1],
        "keepdim": args[2] if len(args) == 3 else False
    }

    # TODO: Remove Work around for fused_reduce bug in migraphx
    if acc_kwargs["keepdim"] and sorted(acc_kwargs["dim"]) == [-2, -1]:
        return aten_ops_adaptive_avg_pool2d(mgx_module, node,
                                            (args[0], [1, 1]), {})

    return acc_ops_converters.acc_ops_mean(mgx_module, node, (), acc_kwargs)

@migraphx_converter(torch.ops.aten.std.correction)
def aten_ops_std(mgx_module, node, args, kwargs):
    assert len(args) >= 2

    acc_kwargs = {
        "input": args[0],
        "dim": args[1],
        "correction": kwargs["correction"] if kwargs.get("correction") is not None else 1,
        "keepdim": kwargs["keepdim"] if kwargs.get("keepdim") is not None else False
    }

    return acc_ops_converters.acc_ops_std(mgx_module, node, (), acc_kwargs)

@migraphx_converter(torch.ops.aten._adaptive_avg_pool2d.default)
def aten_ops_adaptive_avg_pool2d(mgx_module, node, args, kwargs):
    assert len(args) == 2

    acc_kwargs = {"input": args[0], "output_size": args[1]}
    return acc_ops_converters.acc_ops_adaptive_avg_pool2d(
        mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.avg_pool2d.default)
def aten_ops_avg_pool2d(mgx_module, node, args, kwargs):
    assert len(args) >= 2

    acc_kwargs = {
        "input": args[0],
        "kernel_size": args[1],
        "stride": args[2] if len(args) >= 3 and args[2] else args[1],
        "padding": args[3] if len(args) >= 4 else 0,
        "ceil_mode": args[4] if len(args) >= 5 else False,
        "count_include_pad": args[5] if len(args) == 6 else True
    }

    return acc_ops_converters.acc_ops_avg_pool2d(mgx_module, node, (),
                                                 acc_kwargs)


@migraphx_converter(torch.ops.aten.max_pool2d_with_indices.default)
def aten_ops_max_pool2d(mgx_module, node, args, kwargs):
    assert len(args) >= 2

    acc_kwargs = {
        "input": args[0],
        "kernel_size": args[1],
        "stride": args[2] if len(args) >= 3 and args[2] else args[1],
        "padding": args[3] if len(args) >= 4 else 0,
        "dilation": args[4] if len(args) >= 5 else 1,
        "ceil_mode": args[5] if len(args) == 6 else False
    }

    return acc_ops_converters.acc_ops_max_pool2d(mgx_module, node, (),
                                                 acc_kwargs), None


@migraphx_converter(torch.ops.aten.native_layer_norm.default)
def aten_ops_layer_norm(mgx_module, node, args, kwargs):
    assert len(args) == 5

    acc_kwargs = {
        "input": args[0],
        "normalized_shape": args[1],
        "weight": args[2],
        "bias": args[3],
        "eps": args[4]
    }

    return acc_ops_converters.acc_ops_layer_norm(mgx_module, node, (),
                                                 acc_kwargs), None, None


@migraphx_converter(torch.ops.aten.embedding.default)
def aten_ops_embedding(mgx_module, node, args, kwargs):
    assert len(args) >= 2

    acc_kwargs = {
        "weight": args[0],
        "input": args[1],
    }

    return acc_ops_converters.acc_ops_embedding(mgx_module, node, (),
                                                acc_kwargs)


@migraphx_converter(torch.ops.aten.argmax.default)
def aten_ops_argmax(mgx_module, node, args, kwargs):
    assert len(args) >= 1

    acc_kwargs = {
        "input": args[0],
        "dim": args[1] if len(args) >= 2 else None,
        "keepdim": args[2] if len(args) >= 3 else False
    }

    return acc_ops_converters.acc_ops_argmax(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.argmin.default)
def aten_ops_argmin(mgx_module, node, args, kwargs):
    assert len(args) >= 1

    acc_kwargs = {
        "input": args[0],
        "dim": args[1] if len(args) >= 2 else None,
        "keepdim": args[2] if len(args) >= 3 else False
    }

    return acc_ops_converters.acc_ops_argmin(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.max.default)
@migraphx_converter(torch.ops.aten.max.dim)
def aten_ops_max(mgx_module, node, args, kwargs):
    assert len(args) >= 1

    acc_kwargs = {
        "input": args[0],
        "keepdim": False,
    }

    if len(args) >= 2:
        acc_kwargs["dim"] = args[1]

    if len(args) >= 3:
        acc_kwargs["keepdim"] = args[2]

    return acc_ops_converters.acc_ops_max(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.min.default)
@migraphx_converter(torch.ops.aten.min.dim)
def aten_ops_min(mgx_module, node, args, kwargs):
    assert len(args) >= 1

    acc_kwargs = {"input": args[0], "keepdim": False}

    if len(args) >= 2:
        acc_kwargs["dim"] = args[1]

    if len(args) >= 3:
        acc_kwargs["keepdim"] = args[2]

    return acc_ops_converters.acc_ops_min(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.stack.default)
def aten_ops_stack(mgx_module, node, args, kwargs):
    assert len(args) >= 1
    """
    Map aten.stack to unsqueeze + cat acc ops.
    """
    inputs = args[0]
    assert isinstance(inputs, Sequence)

    dims = args[1] if len(args) > 1 else 0

    unsqueeze_kwargs = {"dim": dims}
    cat_kwargs = {"dim": dims}

    unsqueeze_nodes = []
    for i, t in enumerate(inputs):
        unsqueeze_kwargs["input"] = t
        unsq_res = acc_ops_converters.acc_ops_unsqueeze(
            mgx_module, node, (), unsqueeze_kwargs)
        unsqueeze_nodes.append(unsq_res)

    cat_kwargs["tensors"] = unsqueeze_nodes
    return acc_ops_converters.acc_ops_cat(mgx_module, node, (), cat_kwargs)


@migraphx_converter(torch.ops.aten.as_strided.default)
def aten_ops_as_strided(mgx_module, node, args, kwargs):
    assert len(args) >= 3

    acc_kwargs = {
        "input": args[0],
        "size": args[1],
        "stride": args[2],
        "storage_offset": args[3] if len(args) == 4 else 0
    }

    return acc_ops_converters.acc_ops_as_strided(mgx_module, node, (),
                                                 acc_kwargs)


@migraphx_converter(torch.ops.aten.eq.Scalar)
@migraphx_converter(torch.ops.aten.eq.Tensor)
def aten_ops_eq(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, other = args[0], args[1]

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_eq(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.ne.Scalar)
@migraphx_converter(torch.ops.aten.ne.Tensor)
def aten_ops_ne(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, other = args[0], args[1]

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_ne(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.gt.Scalar)
@migraphx_converter(torch.ops.aten.gt.Tensor)
def aten_ops_gt(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, other = args[0], args[1]

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_gt(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.lt.Scalar)
@migraphx_converter(torch.ops.aten.lt.Tensor)
def aten_ops_lt(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, other = args[0], args[1]

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_lt(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.ge.Scalar)
@migraphx_converter(torch.ops.aten.ge.Tensor)
def aten_ops_ge(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, other = args[0], args[1]

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_ge(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.le.Scalar)
@migraphx_converter(torch.ops.aten.le.Tensor)
def aten_ops_le(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, other = args[0], args[1]

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_le(mgx_module, node, (), acc_kwargs)

  
@migraphx_converter(torch.ops.aten.floor.default)
def aten_ops_floor(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_floor(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.logical_not.default)
def aten_ops_logical_not(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_logical_not(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.neg.default)
def aten_ops_neg(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_neg(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.abs.default)
def aten_ops_abs(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_abs(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.isinf.default)
def aten_ops_isinf(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_isinf(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.any.default, min_migraphx_ver="2.11.0")
@migraphx_converter(torch.ops.aten.any.dim, min_migraphx_ver="2.11.0")
def aten_ops_any(mgx_module, node, args, _kwargs):
    assert len(args) >= 1

    acc_kwargs = {
        "input": args[0],
        "dim": None,
        "keepdim": False,
    }

    if len(args) >= 2:
        acc_kwargs["dim"] = args[1]

    if len(args) >= 3:
        acc_kwargs["keepdim"] = args[2]

    return acc_ops_converters.acc_ops_any(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.all.default, min_migraphx_ver="2.11.0")
@migraphx_converter(torch.ops.aten.all.dim, min_migraphx_ver="2.11.0")
def aten_ops_all(mgx_module, node, args, _kwargs):
    assert len(args) >= 1

    acc_kwargs = {
        "input": args[0],
        "dim": None,
        "keepdim": False,
    }

    if len(args) >= 2:
        acc_kwargs["dim"] = args[1]

    if len(args) >= 3:
        acc_kwargs["keepdim"] = args[2]

    return acc_ops_converters.acc_ops_all(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.isnan.default)
def aten_ops_isnan(mgx_module, node, args, _kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_isnan(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.nan_to_num.default)
def aten_ops_nan_to_num(mgx_module, node, args, _kwargs):
    assert len(args) >= 1
    acc_kwargs = {"input": args[0]}
    if len(args) >= 2:
        acc_kwargs["nan"] = args[1]
    if len(args) >= 3:
        acc_kwargs["posinf"] = args[2]
    if len(args) >= 4:
        acc_kwargs["neginf"] = args[3]
    return acc_ops_converters.acc_ops_nan_to_num(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.bitwise_and.Scalar, min_migraphx_ver="2.11.0")
@migraphx_converter(torch.ops.aten.bitwise_and.Tensor, min_migraphx_ver="2.11.0")
def aten_ops_bitwise_and(mgx_module, node, args, _kwargs):
    assert len(args) >= 2
    inp, other = args[0], args[1]
    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_bitwise_and(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten._scaled_dot_product_flash_attention.default)
def aten_ops_scaled_dot_product_attention(mgx_module, node, args, kwargs):
    assert len(args) >= 3
    query, key, value = args[0], args[1], args[2]
    acc_kwargs = {"query": query, "key": key, "value": value}

    if len(args) >= 4:
        acc_kwargs["dropout_p"] = args[3]
    if len(args) >= 5:
        acc_kwargs["is_causal"] = args[4]

    if "scale" in kwargs:
        acc_kwargs["scale"] = kwargs["scale"]

    node.meta['tensor_meta'] = node.meta['tensor_meta'][0]
    
    return acc_ops_converters.acc_ops_scaled_dot_product_attention(mgx_module, node, (), acc_kwargs), None, None, None, None, None, None, None



@migraphx_converter(torch.ops.aten._scaled_dot_product_efficient_attention.default)
def aten_ops_scaled_dot_product_attention(mgx_module, node, args, kwargs):
    assert len(args) >= 3
    query, key, value = args[0], args[1], args[2]
    acc_kwargs = {"query": query, "key": key, "value": value}

    if len(args) >= 4:
        acc_kwargs["attn_bias"] = args[3]
    if len(args) >= 7:
        acc_kwargs["is_causal"] = args[6]

    if "scale" in kwargs:
        acc_kwargs["scale"] = kwargs["scale"]

    node.meta['tensor_meta'] = node.meta['tensor_meta'][0]
    
    return acc_ops_converters.acc_ops_scaled_dot_product_attention(mgx_module, node, (), acc_kwargs), None, None, None

  
@migraphx_converter(torch.ops.aten.erf.default)
def aten_ops_erf(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_erf(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.gather.default)
def aten_ops_gather(mgx_module, node, args, kwargs):
    assert len(args) == 3
    acc_kwargs = {"input": args[0], "dim": args[1], "index": args[2]}
    return acc_ops_converters.acc_ops_gather(mgx_module, node, (), acc_kwargs)

