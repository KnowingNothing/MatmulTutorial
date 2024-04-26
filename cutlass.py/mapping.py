from typing import List, Optional
from dataclasses import dataclass
from tiling import HyperCube, HyperPoint
from functools import reduce

class Function:
    def forward(self, *args):
        raise NotImplementedError()
    
    def backward(self, *args):
        raise NotImplementedError()
    
class Mapping(Function):
    def __init__(self, functions: Optional[List[Function]] = None) -> None:
        self.functions = functions if functions is not None else []
        for func in self.functions:
            assert isinstance(func, Function), "Should put Function type in Mapping."

@dataclass
class Layout(Function):
    shape: HyperCube
    stride: Optional[HyperPoint] = None
    
    def __post_init__(self):
        self.ndim = len(self.shape)
        if self.stride is None:
            self.stride = Hyperreduce(lambda a, b: a + [a[-1] * b], reversed(self.shape[:-1]), [1])