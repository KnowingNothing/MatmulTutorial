import math
from typing import List, Optional, Union
from dtype import IntegerType

class HyperCube:
    def __init__(self, ndim: int, dimensions: Optional[List[Union[int, IntegerType]]] = None) -> None:
        """
        dimensions:
        each item: if item is IntegerType, it's dynamic, else it's static int value.
        """
        self.ndim = ndim
        self.dimensions = ([IntegerType(unsigned=False, bits=32)] for _ in range(ndim)) if dimensions is None else dimensions
        assert isinstance(dimensions, (list, tuple))
        assert len(self.dimensions) == self.ndim
        assert all([self.dim_is_dynamic(i) or self.dimensions[i] > 0 for i in range(self.ndim)]), "Static dim value should be > 0."

    def __getitem__(self, key: int):
        assert key >= 0 and key < self.ndim
        return self.dimensions[key]
    
    def __setitem__(self, key: int, value: Union[int, IntegerType]):
        assert key >= 0 and key < self.ndim
        assert isinstance(value, IntegerType) or value > 0, "Static dim value should be > 0."
        self.dimensions[key] = value

    def dim_is_dynamic(self, key: int):
        return isinstance(self[key], IntegerType)

    def dim_is_static(self, key: int):
        return not self.dim_is_dynamic(key)

    def has_dynamic(self):
        return any([self.dim_is_dynamic(i) for i in range(self.ndim)])
    
    def get_static_dims(self):
        return [self.dimensions[i] for i in range(self.ndim) if self.dim_is_static(i)]
    
    def get_static_dims_with_keys(self):
        return [(i, self.dimensions[i]) for i in range(self.ndim) if self.dim_is_static(i)]

    def num_elements(self):
        assert not self.has_dynamic(), "Can't get number of elements for dynamic HyperCube."
        return math.prod(self.dimensions)
    
    def num_elements_gt(self, number: int):
        """
        We can conclude if number of elements larger than a given value.
        Because all dimensions should be > 0.
        """
        return math.prod(self.get_static_dims()) > number
    
    def num_elements_ge(self, number: int):
        """
        We can conclude if number of elements larger than a given value.
        Because all dimensions should be > 0.
        """
        return math.prod(self.get_static_dims()) >= number
    
    def __repr__(self) -> str:
        numbers = ",".join(map(str, self.dimensions))
        return f"HyperCube({numbers})"


class HyperPoint:
    """
    HyperPoint is similar to HyperCube. But values in dimensions could be any integer.
    """
    def __init__(self, ndim: int, dimensions: Optional[List[Union[int, IntegerType]]] = None) -> None:
        """
        dimensions:
        each item: if item is IntegerType, it's dynamic, else it's static int value.
        """
        self.ndim = ndim
        self.dimensions = ([IntegerType(unsigned=False, bits=32)] for _ in range(ndim)) if dimensions is None else dimensions
        assert isinstance(dimensions, (list, tuple))
        assert len(self.dimensions) == self.ndim

    def __getitem__(self, key: int):
        assert key >= 0 and key < self.ndim
        return self.dimensions[key]
    
    def __setitem__(self, key: int, value: Union[int, IntegerType]):
        assert key >= 0 and key < self.ndim
        self.dimensions[key] = value

    def dim_is_dynamic(self, key: int):
        return isinstance(self[key], IntegerType)

    def dim_is_static(self, key: int):
        return not self.dim_is_dynamic(key)

    def has_dynamic(self):
        return any([self.dim_is_dynamic(i) for i in range(self.ndim)])
    
    def get_static_dims_with_keys(self):
        return [(i, self.dimensions[i]) for i in range(self.ndim) if self.dim_is_static(i)]


