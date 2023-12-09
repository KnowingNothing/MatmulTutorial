from dataclasses import dataclass
from typing import Any
from data_math import is_py_value
from enum import Enum
from functools import reduce
import numpy as np


class NameGenerator:
    def __init__(self) -> None:
        self._cache = {}

    def get(self, name):
        lastname = name
        ret = name
        while ret in self._cache:
            lastname = ret
            ret = ret + str(self._cache[ret])
            self._cache[lastname] += 1
        self._cache[ret] = 0
        return ret


class Arange:
    name_generator = NameGenerator()

    def __init__(self, start, end, step: int = 1, binding_name=None) -> None:
        self._idx = np.arange(start, end, step)
        binding_name = (
            self.name_generator.get("_") if binding_name is None else binding_name
        )
        self._dim_map = {binding_name: self._idx}
        self._dim_order = [binding_name]

    @classmethod
    def make(cls, idx, dim_map, dim_order):
        ret = Arange(0, 0)
        ret._idx = idx
        ret._dim_map = dim_map
        ret._dim_order = dim_order
        return ret

    @property
    def dim(self):
        return len(self._idx.shape)

    def __handle_binary__(self, func, other):
        if is_py_value(other):
            return Arange.make(func(self._idx, other), self._dim_map, self._dim_order)
        else:
            assert isinstance(other, Arange)
            left = self._idx
            for i in range(other.dim):
                left = np.expand_dims(left, axis=-1)
            right = other._idx
            for i in range(self.dim):
                right = np.expand_dims(right, axis=0)
            idx = func(left, right)
            for k in self._dim_map.keys():
                assert k not in other._dim_map.keys(), "Unexpected dim name collision"
            dim_map = {k: v for k, v in self._dim_map.items()}
            dim_map.update(other._dim_map)
            dim_order = self._dim_order + other._dim_order
            return Arange.make(idx, dim_map, dim_order)

    def __handle_rbinary__(self, func, other):
        if is_py_value(other):
            return Arange.make(func(other, self._idx), self._dim_map, self._dim_order)
        else:
            assert isinstance(other, Arange)
            left = self._idx
            for i in range(other.dim):
                left = np.expand_dims(left, axis=-1)
            right = other._idx
            for i in range(self.dim):
                right = np.expand_dims(right, axis=0)
            idx = func(right, left)
            for k in self._dim_map.keys():
                assert k not in other._dim_map.keys(), "Unexpected dim name collision"
            dim_map = {k: v for k, v in self._dim_map.items()}
            dim_map.update(other._dim_map)
            dim_order = other._dim_order + self._dim_order
            return Arange.make(idx, dim_map, dim_order)

    def __add__(self, other):
        return self.__handle_binary__(lambda x, y: x + y, other)

    def __sub__(self, other):
        return self.__handle_binary__(lambda x, y: x - y, other)

    def __mul__(self, other):
        return self.__handle_binary__(lambda x, y: x * y, other)

    def __div__(self, other):
        return self.__handle_binary__(lambda x, y: x / y, other)

    def __mod__(self, other):
        return self.__handle_binary__(lambda x, y: x % y, other)

    def __radd__(self, other):
        return self.__handle_rbinary__(lambda x, y: x + y, other)

    def __rsub__(self, other):
        return self.__handle_rbinary__(lambda x, y: x - y, other)

    def __rmul__(self, other):
        return self.__handle_rbinary__(lambda x, y: x * y, other)

    def __rdiv__(self, other):
        return self.__handle_rbinary__(lambda x, y: x / y, other)

    def __rmod__(self, other):
        return self.__handle_rbinary__(lambda x, y: x % y, other)


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
            Arange(0, self.grid.x, 1, "blockx"), Arange(0, self.grid.y, 1, "blocky"), Arange(0, self.grid.z, 1, "blockz")
        )

    def threadRange(self):
        return Range3(
            Arange(0, self.block.x, 1, "threadx"), Arange(0, self.block.y, 1, "thready"), Arange(0, self.block.z, 1, "threadz")
        )
