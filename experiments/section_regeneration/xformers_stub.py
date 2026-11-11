"""Local-only xformers stub so MusicGen can load on machines without xformers (e.g. Apple Silicon).

audiocraft checks that xformers is importable whenever the checkpoint config sets
memory_efficient=True, but with the default 'torch' attention backend it never calls
xformers kernels: the mask object is only tested for `is not None` (causal flag).
Import this module before audiocraft. Do not use it in the server image.
"""
import sys
import types

import torch

try:
    import xformers  # noqa: F401
except ImportError:
    _xformers = types.ModuleType('xformers')
    _ops = types.ModuleType('xformers.ops')

    class LowerTriangularMask:
        pass

    def _unavailable(*args, **kwargs):
        raise RuntimeError("xformers stub: only the 'torch' attention backend is supported")

    _ops.LowerTriangularMask = LowerTriangularMask
    _ops.memory_efficient_attention = _unavailable
    _ops.unbind = torch.unbind
    _xformers.ops = _ops
    sys.modules['xformers'] = _xformers
    sys.modules['xformers.ops'] = _ops
