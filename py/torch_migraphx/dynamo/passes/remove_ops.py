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
import logging
import os
import torch
import torch.fx
from .utils import log_pass

_LOGGER = logging.getLogger(__name__)
DYNAMO_LOGLEVEL = os.environ.get('TORCH_MIGRAPHX_LOG_DYNAMO_PASSES', None)
if DYNAMO_LOGLEVEL:
    _LOGGER.setLevel(DYNAMO_LOGLEVEL)


@log_pass(_LOGGER, logging.DEBUG)
def remove_clone_ops(gm: torch.fx.GraphModule):
    clone_ops = [
        torch.ops.aten.clone.default,
        torch.ops.aten.copy.default,
    ]
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in clone_ops:
            og_node = node
            in_node = node.all_input_nodes[0]
            og_node.replace_all_uses_with(in_node)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


@log_pass(_LOGGER, logging.DEBUG)
def remove_view_ops(gm: torch.fx.GraphModule):
    view_ops = [
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.view.default,
    ]
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in view_ops:
            og_node = node
            with gm.graph.inserting_after(og_node):
                new_node = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape,
                    args=og_node.args,
                    kwargs=og_node.kwargs,
                )
                new_node.meta = og_node.meta
                og_node.replace_all_uses_with(new_node)
                gm.graph.erase_node(og_node)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


@log_pass(_LOGGER, logging.DEBUG)
def remove_const_ops(gm: torch.fx.GraphModule, device: str = "cuda"):

    @log_pass(_LOGGER, logging.DEBUG)
    def _remove_new_const_ops(gm: torch.fx.GraphModule):
        const_ops = {
            torch.ops.aten.new_zeros.default: torch.zeros,
            torch.ops.aten.new_ones.default: torch.ones,
        }
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in const_ops.keys():
                og_node = node
                size = node.args[1]
                dtype = og_node.meta['tensor_meta'].dtype
                const_tensor = const_ops[node.target](*size,
                                                      dtype=dtype,
                                                      device=device)
                const_name = f"{node.name}_const"
                gm.register_buffer(const_name, const_tensor)
                with gm.graph.inserting_after(og_node):
                    new_node = gm.graph.create_node(
                        "get_attr",
                        const_name,
                    )
                    new_node.meta = og_node.meta
                    og_node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(og_node)

        gm.graph.eliminate_dead_code()
        gm.recompile()

    @log_pass(_LOGGER, logging.DEBUG)
    def _remove_const_like_ops(gm: torch.fx.GraphModule):
        const_ops = {
            torch.ops.aten.full_like.default: torch.full,
            torch.ops.aten.full.default: torch.full,
            torch.ops.aten.zeros_like.default: torch.full,
        }
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in const_ops.keys():
                og_node = node
                size = og_node.meta['tensor_meta'].shape
                dtype = og_node.meta['tensor_meta'].dtype   
                value = 0 if node.target == torch.ops.aten.zeros_like.default else node.args[1]

                const_tensor = const_ops[node.target](size,
                                                      value,
                                                      dtype=dtype,
                                                      device=device)
                const_name = f"{node.name}_const"
                gm.register_buffer(const_name, const_tensor)
                with gm.graph.inserting_after(og_node):
                    new_node = gm.graph.create_node(
                        "get_attr",
                        const_name,
                    )
                    new_node.meta = og_node.meta
                    og_node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(og_node)

        gm.graph.eliminate_dead_code()
        gm.recompile()

    @log_pass(_LOGGER, logging.DEBUG)
    def _remove_range_ops(gm: torch.fx.GraphModule):
        const_ops = {
            torch.ops.aten.arange.start: torch.arange,
        }
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in const_ops.keys():
                og_node = node
                dtype = og_node.meta['tensor_meta'].dtype
                const_tensor = const_ops[node.target](*node.args,
                                                      dtype=dtype,
                                                      device=device)
                const_name = f"{node.name}_const"
                gm.register_buffer(const_name, const_tensor)
                with gm.graph.inserting_after(og_node):
                    new_node = gm.graph.create_node(
                        "get_attr",
                        const_name,
                    )
                    new_node.meta = og_node.meta
                    og_node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(og_node)

        gm.graph.eliminate_dead_code()
        gm.recompile()

    _remove_new_const_ops(gm)
    _remove_const_like_ops(gm)
    _remove_range_ops(gm)
    gm.recompile()
    return gm
