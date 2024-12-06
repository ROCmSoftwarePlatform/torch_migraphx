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

from typing import Dict, Optional, Sequence, Mapping
import logging
import os

import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport

from torch_migraphx.fx.converter_registry import CONVERTERS
from ..utils import get_graph_info, SetLogLevel
from ...fx.utils import TYPE_MAP

_LOGGER = logging.getLogger(__name__)
DYNAMO_LOGLEVEL = os.environ.get('TORCH_MIGRAPHX_LOG_DYNAMO_PASSES', None)
PARTITIONER_LOGLEVEL = os.environ.get('TORCH_MIGRAPHX_LOG_PARTITIONER',
                                      DYNAMO_LOGLEVEL)
if PARTITIONER_LOGLEVEL:
    _LOGGER.setLevel(PARTITIONER_LOGLEVEL)


class MGXOperatorSupport(OperatorSupport):
    '''Construct OperatorSupport object used for partitioning based on registered converters'''

    def __init__(self, support_dict=None):
        super().__init__(support_dict)
        self.supported_dtypes = TYPE_MAP.keys()
        # Keep track of visited ops for support summary
        self.supported = set()
        self.unsupported = set()

    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module],
                          node: torch.fx.Node) -> bool:
        node_meta = node.meta.get("tensor_meta", None)
        if node_meta and not node_meta.dtype in self.supported_dtypes:
            self.unsupported.add(f"{node.target} : {node_meta.dtype}")
            return False

        if node.op == "get_attr": return True

        if node.target in CONVERTERS.keys():
            if not node.is_impure():
                self.supported.add(node.target)
            return True
        else:
            if not node.is_impure():
                self.unsupported.add(node.target)
            return False

    def support_summary(self):
        summary = "Supported Nodes:\n"
        for n in self.supported:
            summary += f"\t{n}\n"

        summary += "\nUnsupported Nodes:\n"
        for n in self.unsupported:
            summary += f"\t{n}\n"

        return summary

    def print_support_summary(self):
        print(self.support_summary())


def partition(gm: torch.fx.GraphModule,
              max_partitions: int = 500,
              verbose: bool = True):
    """Partition the graph into supported and unsupported subgraphx for lowering

    Args:
        gm (torch.fx.GraphModule): Graph to be partitioned
        max_partitions (int, optional): Max allowed partitions. Defaults to 20.
        verbose (bool, optional): Print node suppory summary. Defaults to True.
    """

    op_support = MGXOperatorSupport()
    partitioner = CapabilityBasedPartitioner(gm, op_support)

    partitions = partitioner.propose_partitions()
    fused_gm = partitioner.fuse_partitions(partitions)
    fused_gm.graph.eliminate_dead_code()
    fused_gm.recompile()
    fused_gm.delete_all_unused_submodules()

    log_level = _LOGGER.level
    if verbose and _LOGGER.level > logging.INFO:
        log_level = logging.INFO

    with SetLogLevel(_LOGGER, log_level):
        _LOGGER.debug(f"Partitioned Module:\n{get_graph_info(fused_gm.graph)}")
        _LOGGER.info(f"Node support summary:\n{op_support.support_summary()}")
        _LOGGER.info(f"Number of partitions: {len(partitions)}")

    # TODO: Compute number of partitions after dead code elimination
    if len(partitions) > max_partitions:
        raise RuntimeError(
            f'Found {len(partitions)} partitions, max allowed: {max_partitions}.'
        )

    return fused_gm


def get_partition_inputs(
        mod: torch.fx.GraphModule, submod: torch.fx.GraphModule,
        example_input: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    """Returns the input to a specific submodule given initial model input. 
       Uses same technique as torch.fx.passes.splitter_base.

    Args:
        mod (torch.fx.GraphModule): Full model graph
        submod (torch.fx.GraphModule): Subgraph for which to capture input
        example_input (Sequence[torch.Tensor]): Input to the full model graph

    Returns:
        Sequence[torch.Tensor]: Inputs to the specified subgraph
    """
    sub_inputs = None

    def get_inputs(self, inputs):
        nonlocal sub_inputs
        sub_inputs = inputs

    handle = submod.register_forward_pre_hook(get_inputs)
    mod(*example_input)
    handle.remove()
    return sub_inputs
