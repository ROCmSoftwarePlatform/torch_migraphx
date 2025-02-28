
import pytest
import torch
from dynamo_test_utils import  MultiInFuncModule, convert_to_mgx, verify_outputs
import sys

@pytest.mark.skipif('torchvision' not in sys.modules,
                    reason="requires the torchvision library")
@pytest.mark.parametrize('op_alias', [torch.ops.torchvision.roi_align.default])
@pytest.mark.parametrize("spatial_scale, sampling_ratio", [(1.5, 3), (0.5, 2)])
@pytest.mark.parametrize("aligned", [(True), (False)])
@pytest.mark.parametrize(
    "input, boxes, output_size", [((1, 2, 3, 4),  ([[0, 1.1, 1.2,  0.6, 2.6]]), [2, 3]),
                                  #   boxes as List[Tensor[L, 4]] is not supported
                                  #############  ((2, 2, 3, 3),  ([(1.1, 1.2,  0.6, 2.6), (1.13, 1.23,  0.63, 2.63)]), [2, 2]),
                                  ((2, 2, 3, 4),  ([[1, 1.1, 1.2,  0.6, 2.6], [0, 1.13, 1.23,  0.63, 2.63]]), [5, 2]),
                                  ((4, 5, 256, 256),
                                   ([[0, 10.6, 11.2,  21.1, 22.6],
                                     [2, 10.63, 11.23,  21.63, 22.63],
                                     [3, 10.64, 11.23,  21.67, 22.63],
                                     [1, 10.65, 11.23,  21.68, 22.63],
                                     ]),
                                   [7, 6])
                                  ]
    )
def test_roialign(op_alias, input, boxes, output_size, spatial_scale, sampling_ratio, aligned):
    assert(input[0] == len(boxes))
    inp = torch.randn(input).cuda()
    roi = torch.tensor(boxes).cuda()
    
    mod = MultiInFuncModule(op_alias, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned)
    mgx_mod = convert_to_mgx(mod, [inp, roi])
    verify_outputs(mod, mgx_mod, [inp, roi])


