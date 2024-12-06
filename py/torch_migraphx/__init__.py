from packaging import version
from torch import __version__ as _torch_version
import migraphx
from torch_migraphx import fx, _C

if version.parse(_torch_version) >= version.parse("2.1.0"):
    from torch_migraphx import dynamo

import logging
import os
import sys

LOGLEVEL = os.environ.get('TORCH_MIGRAPHX_LOGLEVEL', 'WARNING').upper()
logging.basicConfig(
    level=LOGLEVEL,
    stream=sys.stderr,
    format=
    '%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s\n',
    datefmt='%Y-%m-%d:%H:%M:%S')
