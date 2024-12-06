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
import logging
import os
from typing import Any, Callable, Optional, Sequence

import migraphx
import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.splitter_base import SplitResult

from .fx2mgx import MGXInterpreter
from .lower_setting import LowerSetting
from .passes.lower_pass_manager_builder import LowerPassManagerBuilder
from .passes.pass_utils import decorate_method, PassFunc, validate_inference
from torch_migraphx.fx.tracer.acc_tracer.acc_shape_prop import AccShapeProp
from .tools.mgx_splitter import MGXSplitter, MGXSplitterSetting

from .tracer.acc_tracer import acc_tracer
from .tracer.aten_tracer import aten_tracer
from .mgx_module import MGXModule
from .utils import LowerPrecision, SuppressPrints, SetLogLevel, get_graph_info

_LOGGER = logging.getLogger(__name__)
LOWERER_LOGLEVEL = os.environ.get('TORCH_MIGRAPHX_LOG_FX_LOWER', None)
if LOWERER_LOGLEVEL:
    _LOGGER.setLevel(LOWERER_LOGLEVEL)
Input = Sequence[Any]


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (tuple, list)):
        return [to_device(y) for y in x]
    else:
        return x


def lower_to_mgx(module: nn.Module,
                 input,
                 lower_precision=LowerPrecision.FP32,
                 min_acc_module_size=10,
                 use_aten=False,
                 verbose_log=False,
                 suppress_accuracy_check=False,
                 save_subgraph_programs=False,
                 tracer_base_cls=torch.fx.Tracer,
                 leaf_modules=None) -> nn.Module:
    """
    Takes in original module, input and lowering setting, run lowering workflow to turn module
    into lowered module.
    Args:
        module: Original module for lowering.
        input: Input for module (list of tensors).
        lower_precision: Precision to use for lowered program (default FP32).
        use_aten: Export graph to aten ops rather than using acc converters (default False).
        verbose_log: Enable verbose log (default False).
        suppress_accuracy_check: Suppress accuracy errors detected on lowered modules (default False).
        tracer_base_cls: FX Tracer to use for tracing model (default torch.fx.Tracer).
        leaf_modules: List of modules to be treated as leaf nodes by FX tracer (default None).
    Returns:
        A SplitModule object containing MGXModule (lowered graphs) and torch.fx.GraphModule (unsupported graphs) objects.
    """
    module = module.cpu().eval()
    input = [to_device(x, "cpu") for x in input]

    if verbose_log and _LOGGER.level > logging.INFO:
        log_level = logging.INFO
    else:
        log_level = _LOGGER.level

    with SetLogLevel(_LOGGER, log_level):
        lower_setting = LowerSetting(
            lower_precision=lower_precision,
            min_acc_module_size=min_acc_module_size,
            suppress_accuracy_check=suppress_accuracy_check,
            save_subgraph_programs=save_subgraph_programs,
            tracer_base_cls=tracer_base_cls,
            leaf_module_list=leaf_modules,
            use_aten=use_aten,
        )
        lowerer = Lowerer.create(lower_setting=lower_setting)
        return lowerer(module, input)


@dc.dataclass
class LowerMgxInterpreter:
    lower_setting: LowerSetting

    @classmethod
    def create(cls, lower_setting):
        return LowerMgxInterpreter(lower_setting)

    def __call__(self, mod, input, split_name) -> MGXInterpreter:
        _LOGGER.info(f"Running MGXInterpreter for {split_name}")
        AccShapeProp(mod).propagate(*input)
        input_shapes = [(i.shape, i.dtype) for i in input]
        _LOGGER.debug(f"Input Shapes: {input_shapes}")
        _LOGGER.debug(f"{split_name} Graph:\n{get_graph_info(mod.graph)}")
        interpreter = MGXInterpreter(mod, input)
        interpreter.run()

        return interpreter


def default_split_function(model: fx.GraphModule, inputs: Input,
                           lower_setting: LowerSetting) -> SplitResult:
    splitter_setting = MGXSplitterSetting()
    splitter_setting.min_acc_module_size = lower_setting.min_acc_module_size
    splitter = MGXSplitter(model, inputs, settings=splitter_setting)
    with SuppressPrints():
        node_preview = splitter.node_support_preview()
    _LOGGER.info(f"\n{node_preview}")
    return splitter.generate_split_results()


def create_lower_mgx_interpreter(
        lower_setting: LowerSetting) -> LowerMgxInterpreter:
    return LowerMgxInterpreter.create(lower_setting)


def default_lower_pass(
    create_mgx_interpreter: Callable[[LowerSetting], LowerMgxInterpreter],
) -> PassFunc:

    def lower_pass(mod: nn.Module, input: Input, lower_setting: LowerSetting,
                   module_name: str) -> nn.Module:
        """
        Create a module transformation pass which lowers an `fx.GraphModule` into an
        accelerated module
        """
        interpreter = create_mgx_interpreter(lower_setting)
        interp_res: MGXInterpreter = interpreter(mod, input, module_name)
        
        _LOGGER.debug(f"Interpreted MIGraphX Program:\n{interp_res.program}")

        if lower_setting.save_subgraph_programs:
            migraphx.save(interp_res.program, f'{module_name}.mxr')

        fp16_mode = lower_setting.lower_precision == LowerPrecision.FP16

        mgx_module = MGXModule(
            program=interp_res.program,
            input_names=interp_res.get_input_names(),
            quantize_fp16=fp16_mode,
        )
        
        _LOGGER.debug(f"Compiled MIGraphX Program:\n{mgx_module.program}")
        return mgx_module

    return lower_pass


@dc.dataclass(frozen=True)
class Lowerer:
    """Lowers a module using fx2mgx.
        1. Trace torch module to get graph representation (fx or aten export)
        2. Split the generated graph into subgraphs based on ops supported by the accelerator
    
    For each subgraph that can be lowered to accelerator:
        1. Use MGXInterpreter to create a MIGraphX program
        2. Wrap this program into MGXModule so that it can be treated as nn.Module
        3. Attach submodule to SplitModule object that contains the full program execution path
    """

    lower_pass_manager_builder: LowerPassManagerBuilder

    @classmethod
    def create(
        cls,
        lower_setting: LowerSetting,
        interpreter_builder: Callable = create_lower_mgx_interpreter,
        split_func: Callable = default_split_function,
    ) -> "Lowerer":
        """Instantiate a `Lowerer` instance."""

        if not lower_setting.use_aten:
            return cls(lower_pass_manager_builder=LowerPassManagerBuilder(
                lower_setting=lower_setting,
                trace_func=lambda module, inputs: acc_tracer.trace(
                    module,
                    inputs,
                    ast_rewriter_allow_list=lower_setting.
                    ast_rewriter_allow_list,
                    leaf_module_list=lower_setting.leaf_module_list,
                    tracer_cls=lower_setting.tracer_base_cls,
                ),
                split_func=split_func,
                lower_func=default_lower_pass(interpreter_builder),
            ))
        else:
            return cls(lower_pass_manager_builder=LowerPassManagerBuilder(
                lower_setting=lower_setting,
                trace_func=lambda module, inputs: aten_tracer.trace(
                    module,
                    inputs,
                ),
                split_func=split_func,
                lower_func=default_lower_pass(interpreter_builder),
            ))

    def __call__(
        self,
        module: nn.Module,
        inputs: Input,
        additional_inputs: Optional[Input] = None,
    ) -> nn.Module:
        lower_setting = self.lower_pass_manager_builder.lower_setting
        atol = lower_setting.correctness_atol
        rtol = lower_setting.correctness_rtol

        @validate_inference(atol=atol,
                            rtol=rtol,
                            suppress_accuracy_check_failure=lower_setting.
                            suppress_accuracy_check)
        def lower_mod(module: nn.Module, inputs: Input) -> nn.Module:
            module.eval()

            if lower_setting.use_aten:
                pm = self.lower_pass_manager_builder.build_mgx_aten_lower_pipeline(
                    inputs, additional_inputs)
            else:
                pm = self.lower_pass_manager_builder.build_mgx_lower_pipeline(
                    inputs, additional_inputs)

            lower_result = pm(module)

            return lower_result

        return lower_mod(module, inputs)
