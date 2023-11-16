from dataclasses import dataclass
from typing import Any
from data_math import is_py_value
from enum import Enum
from functools import reduce

class Arange:
    class CombineOp(Enum):
        Nop = -1
        Add = 0
        Sub = 1
        Mul = 2
        Div = 3
        Mod = 4

    class Range:
        def __init__(self, start, end, step=1) -> None:
            self.start = start
            self.end = end
            self.step = step
            
    def __init__(self, start, end, step : Any = 1) -> None:
        self.dims = []
        self.combine_op = self.CombineOp.Nop
        self.sub_arranges = []
        if isinstance(start, (list, tuple)):
            assert isinstance(end, (list, tuple)) and len(start) == len(end)
            step = step if isinstance(step, (list, tuple)) else [step] * len(start)
            assert len(start) == len(step)
            for s, e, st in zip(start, end, step):
                self.dims.append(self.Range(s, e, st))
        else:
            self.dims.append(self.Range(start, end, step))
            
    def dim(self):
        assert len(self.dims) == 0 or len(self.sub_arranges) == 0
        if len(self.dims):
            return len(self.dims)
        else:
            return reduce(lambda x, y: x + y, [x.dim() if isinstance(x, self.__class__) else 0 for x in self.sub_arranges], 0)
    
    @classmethod
    def combine(cls, op, *args):
        assert op in cls.CombineOp
        assert len(args) > 0
        ret = Arange([], [], []) # empty
        ret.combine_op = op
        ret.sub_arranges = list(args)
        return ret
            
    def __add__(self, other):
        return self.combine(self.CombineOp.Add, self, other)
        
    def __sub__(self, other):
        return self.combine(self.CombineOp.Sub, self, other)
        
    def __mul__(self, other):
        return self.combine(self.CombineOp.Mul, self, other)
    
    def __div__(self, other):
        return self.combine(self.CombineOp.Div, self, other)
    
    def __mod__(self, other):
        return self.combine(self.CombineOp.Mod, self, other)
    
    def __radd__(self, other):
        return self.combine(self.CombineOp.Add, other, self)
    
    def __rsub__(self, other):
        return self.combine(self.CombineOp.Sub, other, self)
    
    def __rmul__(self, other):
        return self.combine(self.CombineOp.Mul, other, self)
    
    def __rdiv__(self, other):
        return self.combine(self.CombineOp.Div, other, self)
    
    def __rmod__(self, other):
        return self.combine(self.CombineOp.Mod, other, self)
        
    


@dataclass
class Dim3:
    x: int
    y: int
    z: int


class GridDim(Dim3):
    pass


class ClusterIdx(Dim3):
    pass


class ClusterDim(Dim3):
    pass


class BlockIdx(Dim3):
    pass


class BlockDim(Dim3):
    pass


class BlockIdxInCluster(Dim3):
    pass


class ThreadIdx(Dim3):
    pass


@dataclass
class Range3:
    x: Arange
    y: Arange
    z: Arange


class GPUKernel:
    def __init__(self, grid: Dim3, cluster: Dim3, block: Dim3) -> None:
        """
        grid: grid config (how many blocks in grid)
        cluster: cluster config (how many blocks in cluster)
        block: block config (how many threads in block)
        """
        assert type(grid) == Dim3
        assert type(cluster) == Dim3
        assert type(block) == Dim3
        self.grid = grid
        self.cluster = cluster
        self.block = block

    def blockRange(self):
        return Range3(
            Arange(0, self.grid.x), Arange(0, self.grid.y), Arange(0, self.grid.z)
        )

    def threadRange(self):
        return Range3(
            Arange(0, self.block.x), Arange(0, self.block.y), Arange(0, self.block.z)
        )
