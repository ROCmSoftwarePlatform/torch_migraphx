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
import dataclasses as dc
from typing import List, Optional, Set, Type

from torch import nn
from torch.fx.passes.pass_manager import PassManager
from torch.fx import Tracer

from .utils import LowerPrecision


@dc.dataclass
class LowerSettingBasic:
    """
    Basic class for lowering.
    max_batch_size: The maximum batch size for lowering job.
                    If run with TensorRT lowering, this is the maximum
                    batch size which can be used at execution time,
                    and also the batch size for which the ICudaEngine
                    will be optimized.
                    If run with AITemplate lowering, this the max batch_size
                    for the model.
    lower_precision: lower precision dtype during lowering.
    min_acc_module_size(int): minimal number of nodes for an accelerate submodule.
    ast_rewriter_allow_list (Optional[Set[nn.Module]]): Optional allow list of
    modules that need AST rewriting. This is aiming to eliminate input variable involve in
    exception checking control flow.
    leaf_module_list (Optional[Set[nn.Module]]): Optional leaf module list where
    modules will not be traced into.
    verbose_profile (bool): verbosity of profiler, default to False.
    """

    max_batch_size: int = 2048
    lower_precision: LowerPrecision = LowerPrecision.FP32
    min_acc_module_size: int = 10
    ast_rewriter_allow_list: Optional[Set[Type[nn.Module]]] = None
    leaf_module_list: Optional[Set[Type[nn.Module]]] = None
    verbose_profile: bool = False
    use_aten: bool = False

@dc.dataclass
class LowerSetting(LowerSettingBasic):
    """
    Basic configuration for lowering stack.
    Args:
    explicit_precision: Use explicit precision during lowering.
    preset_lowerer (str): when specified, use a preset logic to build the
    instance of Lowerer.
    """

    explicit_precision: bool = False
    preset_lowerer: str = ""
    suppress_accuracy_check: bool = False
    save_subgraph_programs: bool = False
    tracer_base_cls: Type = Tracer
    correctness_atol: float = 0.5
    correctness_rtol: float = 0.5
