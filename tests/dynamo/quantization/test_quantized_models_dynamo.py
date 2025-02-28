import sys
import pytest
import copy
from packaging import version
import torch_migraphx
import torch
from torchvision import models
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch_migraphx.dynamo.quantization import MGXQuantizer
from quantization_utils_dynamo import (stable_convert_pt2e,
                                       stable_pre_aot_export,
                                       move_q_gm_to_device, 
                                       verify_outputs,
                                       compute_quantized_outputs,
                                       verify_quantized_outputs)

try:
    import transformers
    from transformers import GPT2Tokenizer, GPT2Model
except ImportError:
    pass

transformers_skip_mark = pytest.mark.skipif(
    'transformers' not in sys.modules,
    reason="requires the transformers library")


@pytest.mark.parametrize("model_name, model_weights, rtol, atol", [
    ("resnet50", models.ResNet50_Weights.IMAGENET1K_V1, 1e-1, 1e-1),
])
@pytest.mark.parametrize("asymm", [False, True])
def test_quant_vision_model(model_name, model_weights, rtol, atol, asymm,
                            default_torch_seed):
    model = getattr(models, model_name)(weights=model_weights).eval()
    torch_fp32_mod = copy.deepcopy(model)

    sample_inputs = [torch.randn(4, 3, 244, 244)]

    with torch.no_grad():
        model_export = stable_pre_aot_export(model, sample_inputs)

        quantizer = MGXQuantizer(asymmetric_activations=asymm)
        m = prepare_pt2e(model_export, quantizer)

        # psudo calibrate
        m(*sample_inputs)

        q_m = stable_convert_pt2e(m)
        torch_q_mod = copy.deepcopy(q_m)

        mgx_mod = torch.compile(q_m.cuda(), backend='migraphx')

        verify_outputs(torch_fp32_mod, torch_q_mod, mgx_mod, sample_inputs, rtol,
                    atol)

    del mgx_mod, model, torch_fp32_mod, torch_q_mod, model_export


@pytest.mark.skipif(version.parse(torch.__version__) > version.parse("2.4"), 
                    reason="pt2e pipeline interaction with torch.compile API has changed in PyTorch 2.5, needs refactoring")
@pytest.mark.skipif('transformers' not in sys.modules,
                    reason="requires the transformers library")
@pytest.mark.parametrize(
    "model_class, tokenizer_class, model_name, rtol, atol", [
        ('GPT2Model', 'GPT2Tokenizer', 'distilgpt2', 1e-1, 1e-1),
    ])
@pytest.mark.parametrize("asymm", [False, True])
def test_quant_LLM(model_class, tokenizer_class, model_name, rtol, atol, asymm,
                   default_torch_seed):
    model = getattr(transformers, model_class).from_pretrained(model_name)
    torch_fp32_mod = copy.deepcopy(model)
    tokenizer = getattr(transformers,
                        tokenizer_class).from_pretrained(model_name)
    text = "Just some example text to be tokenized."
    encoded_input = tokenizer(text, return_tensors='pt')
    inputs = [encoded_input["input_ids"]]
    gold_output = model(*inputs)

    with torch.no_grad():
        model_export = stable_pre_aot_export(model, inputs)

        quantizer = MGXQuantizer(asymmetric_activations=asymm)
        m = prepare_pt2e(model_export, quantizer)
        m(*inputs)
        q_m = stable_convert_pt2e(m)
        torch_q_mod = copy.deepcopy(q_m)

        q_m = move_q_gm_to_device(q_m)

        mgx_mod = torch.compile(q_m, backend='migraphx').cuda()

        torch_fp32_out, torch_q_out, mgx_out = compute_quantized_outputs(
            torch_fp32_mod, torch_q_mod, mgx_mod, inputs)
        verify_quantized_outputs(torch_fp32_out.last_hidden_state,
                                torch_q_out.last_hidden_state,
                                mgx_out["last_hidden_state"],
                                rtol=rtol,
                                atol=atol)

    del mgx_mod, model, model_export, q_m
