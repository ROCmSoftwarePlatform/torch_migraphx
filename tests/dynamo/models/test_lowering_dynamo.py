import pytest
import torch
import torch_migraphx
import torchvision.models as models
import os

DEFAULT_RTOL, DEFAULT_ATOL = 3e-3, 1e-2

os.environ["MIGRAPHX_DISABLE_HIPBLASLT_GEMM"] = "1"

@pytest.mark.parametrize("model, rtol, atol", [
    (models.wide_resnet50_2(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.vit_b_16(), DEFAULT_RTOL, DEFAULT_ATOL),
])
def test_vision_model_dynamo(model, rtol, atol, default_torch_seed):
    model = model.eval()
    sample_inputs = [torch.randn(4, 3, 224, 224)]
    torch_out = model(*sample_inputs)

    mgx_model = torch.compile(model.cuda(),
                              backend="migraphx",
                              options={"verbose": True})
    mgx_inputs = [i.cuda() for i in sample_inputs]
    mgx_out = mgx_model(*mgx_inputs)

    assert torch.allclose(mgx_out.cpu(), torch_out, rtol=rtol, atol=atol)

    del mgx_model
    del model
