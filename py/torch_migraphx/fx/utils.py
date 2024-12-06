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
import os
import sys
from enum import Enum
from typing import List, Callable
from packaging import version
import random
import torch
import migraphx
from .. import _C
from torch.fx.passes.shape_prop import TensorMetadata

from typing import List

HIPSTREAMTYPE = 'ihipStream_t'


class SuppressPrints:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class SetLogLevel:

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


class LowerPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


TYPE_MAP = {
    torch.bool: 'bool_type',
    torch.half: 'half_type',
    torch.float: 'float_type',
    torch.double: 'double_type',
    torch.uint8: 'uint8_type',
    torch.int8: 'int8_type',
    torch.int16: 'int16_type',
    torch.int32: 'int32_type',
    torch.long: 'int64_type'
}

INV_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}

QTYPE_MAP = {
    torch.quint8: 'uint8_type',
    torch.qint8: 'int8_type',
    torch.qint32: 'int32_type',
}

INV_QTYPE_MAP = {v: k for k, v in QTYPE_MAP.items()}


def torch_dtype_to_mgx(dtype: torch.dtype) -> str:
    return TYPE_MAP[dtype]


def torch_dtype_from_mgx(type_string: str) -> torch.dtype:
    return INV_TYPE_MAP[type_string]


def torch_qdtype_to_mgx(dtype: torch.dtype) -> str:
    return QTYPE_MAP[dtype]


def torch_qdtype_from_mgx(type_string: str) -> torch.dtype:
    return INV_QTYPE_MAP[type_string]


def mgx_type_str_to_enum(type_string: str) -> migraphx.shape.type_t:
    return getattr(migraphx.shape.type_t, type_string)


def torch_dtype_to_mgx_enum(dtype: torch.dtype) -> migraphx.shape.type_t:
    return mgx_type_str_to_enum(torch_dtype_to_mgx(dtype))


def torch_qdtype_to_mgx_enum(dtype: torch.dtype) -> migraphx.shape.type_t:
    return mgx_type_str_to_enum(torch_qdtype_to_mgx(dtype))


def mgx_argument_from_tensor(tensor: torch.tensor) -> migraphx.argument:
    return _C.tensor_to_arg(tensor)


def tensor_from_mgx_argument(
    arg: migraphx.argument, device: torch.device = torch.device('cuda')
) -> torch.tensor:
    return _C.arg_to_tensor(arg, device)


def mgx_argument_from_ptr(ptr: int,
                          shape: migraphx.shape) -> migraphx.argument:
    return migraphx.argument_from_pointer(shape, ptr)


def tensors_from_mgx_arguments_par(args: List[migraphx.argument],
                                   lens: List[List[int]],
                                   strides: List[List[int]],
                                   type_strs: List[str],
                                   device: torch.device = torch.device('cuda'),
                                   thread_size: int = 1) -> List[torch.tensor]:

    ptrs = [a.data_ptr() for a in args]
    scalars = [a.shape().scalar() for a in args]
    return _C.args_to_tensors_par(ptrs, lens, strides, type_strs, scalars,
                                  device, thread_size)


def tensors_from_mgx_arguments(
    args: List[migraphx.argument],
    mgx_shapes: List[migraphx.shape],
    device: torch.device = torch.device('cuda')
) -> List[torch.tensor]:
    return [
        _C.tensor_from_ptr(a.data_ptr(), s.lens(), s.strides(),
                           s.type_string(), s.scalar(), device)
        for a, s in zip(args, mgx_shapes)
    ]


def get_qparams(tensor: torch.Tensor) -> (torch.Tensor, dict):
    if not tensor.is_quantized:
        return tensor, None

    if tensor.qscheme() in (torch.per_tensor_affine,
                            torch.per_tensor_symmetric):
        q_scale = tensor.q_scale()
        q_zero_point = tensor.q_zero_point()
        q_axis = None
    else:
        q_scale = tensor.q_per_channel_scales()
        q_zero_point = tensor.q_per_channel_zero_points()
        q_axis = tensor.q_per_channel_axis()

    q_params = {"scale": q_scale, "zero_point": q_zero_point, "axis": q_axis}
    return tensor.int_repr(), q_params


# TODO: currently the migraphx api does not support directly interacting
# with program bytes. To work around this, we let it create a dummy file
# and read from it. Update this once it is supported by the api
def mgx_program_to_bytearray(program: migraphx.program) -> bytearray:
    dummy_file_name = f'__temp_prog_{random.getrandbits(32)}.mxr'
    migraphx.save(program, dummy_file_name)

    with open(dummy_file_name, 'rb') as f:
        prog_bytes = bytearray(f.read())

    os.remove(dummy_file_name)

    return prog_bytes


def mgx_program_from_bytearray(barray: bytearray) -> migraphx.program:
    dummy_file_name = f'__temp_prog_{random.getrandbits(32)}.mxr'

    with open(dummy_file_name, 'wb') as f:
        f.write(barray)

    prog = migraphx.load(dummy_file_name)
    os.remove(dummy_file_name)

    return prog


def get_node_info(node: torch.fx.Node) -> str:
    node_info = 'Return' if node.op == 'output' else node.format_node()
    out_str = f"{node_info}, args: {node.args}, kwargs: {node.kwargs}"
    if 'tensor_meta' in node.meta:
        out_str += tensor_meta_str(node.meta['tensor_meta'])
    return out_str


def get_graph_info(graph: torch.fx.Graph) -> str:
    out_str = ""
    for node in graph.nodes:
        out_str += f"\n\t{get_node_info(node)}\n"
    return out_str


def print_graph(graph: torch.fx.Graph) -> None:
    print(get_graph_info(graph))


def tensor_meta_str(tm) -> str:
    if isinstance(tm, TensorMetadata):
        return f", shape: [{tm.dtype}, {tm.shape}, {tm.stride}]"
    elif isinstance(tm, (tuple, list)):
        return ' '.join([tensor_meta_str(i) for i in tm])
    else:
        return ''


def req_torch_version(min_torch_version: str = "2.dev"):
    """
    Create a decorator which verifies the Torch version installed
    against a specified version range
    Args:
        min_torch_version (str): The minimum required Torch version
        for the decorated function to work properly
    Returns:
        A decorator which raises a descriptive error message if
        an unsupported Torch version is used
    """

    def nested_decorator(f: Callable):

        def function_wrapper(*args, **kwargs):
            # Parse minimum and current Torch versions
            min_version = version.parse(min_torch_version)
            current_version = version.parse(torch.__version__)

            if current_version < min_version:
                raise AssertionError(
                    f"Expected Torch version {min_torch_version} or greater, "
                    +
                    f"when calling {f}. Detected version {torch.__version__}")
            else:
                return f(*args, **kwargs)

        return function_wrapper

    return nested_decorator
