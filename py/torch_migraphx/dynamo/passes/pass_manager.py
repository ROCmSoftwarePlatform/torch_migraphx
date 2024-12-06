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
from torch.fx.passes.pass_manager import PassManager

from .remove_ops import remove_const_ops, remove_view_ops
from .const_fold import const_fold
from .promote_types import promote_inputs
from .remove_empty_slice import remove_empty_slices
from .fix_tensor_meta import fix_tensor_meta
from ..utils import get_graph_info

_LOGGER = logging.getLogger(__name__)
DYNAMO_LOGLEVEL = os.environ.get('TORCH_MIGRAPHX_LOG_DYNAMO_PASSES', None)
if DYNAMO_LOGLEVEL:
    _LOGGER.setLevel(DYNAMO_LOGLEVEL)

class MGXPassManager(PassManager):

    def __init__(self, passes=None, constraints=None):
        super().__init__(passes, constraints)


def pre_partition_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    passes = [
        remove_const_ops,
        remove_view_ops,
        promote_inputs,
        remove_empty_slices,
        const_fold,
    ]
    pre_partition_pass_mgr = MGXPassManager(passes)
    _LOGGER.info(f"Pre Partition Pass In:\n{get_graph_info(gm.graph)}") 
    return pre_partition_pass_mgr(gm)


def post_partition_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    passes = [
        fix_tensor_meta,
    ]
    post_partition_pass_mgr = MGXPassManager(passes)
    _LOGGER.info(f"Post Partition Pass In:\n{get_graph_info(gm.graph)}") 
    return post_partition_pass_mgr(gm)
