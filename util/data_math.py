from dataclasses import dataclass
from typing import Any


def is_py_value(x: Any):
    return isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)


class DataType:
    bits: int = -1
    lanes: int = 1

    def __init__(self, value) -> None:
        self.value = value

    def as_type(self, type):
        return type(self.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def __add__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value + other.value)
        elif is_py_value(other):
            return self.__class__(self.value + other)
        else:
            raise NotImplementedError()

    def __radd__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value + self.value)
        elif is_py_value(other):
            return self.__class__(other + self.value)
        else:
            raise NotImplementedError()

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value - other.value)
        elif is_py_value(other):
            return self.__class__(self.value - other)
        else:
            raise NotImplementedError()

    def __rsub__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value - self.value)
        elif is_py_value(other):
            return self.__class__(other - self.value)
        else:
            raise NotImplementedError()

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value * other.value)
        elif is_py_value(other):
            return self.__class__(self.value * other)
        else:
            raise NotImplementedError()

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value * self.value)
        elif is_py_value(other):
            return self.__class__(other * self.value)
        else:
            raise NotImplementedError()

    def __floordiv__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value // other.value)
        elif is_py_value(other):
            return self.__class__(self.value // other)
        else:
            raise NotImplementedError()

    def __rfloordiv__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value // self.value)
        elif is_py_value(other):
            return self.__class__(other // self.value)
        else:
            raise NotImplementedError()

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value / other.value)
        elif is_py_value(other):
            return self.__class__(self.value / other)
        else:
            raise NotImplementedError()

    def __rtrueediv__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value / self.value)
        elif is_py_value(other):
            return self.__class__(other / self.value)
        else:
            raise NotImplementedError()

    def __mod__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value % other.value)
        elif is_py_value(other):
            return self.__class__(self.value % other)
        else:
            raise NotImplementedError()

    def __rmod__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value % self.value)
        elif is_py_value(other):
            return self.__class__(other % self.value)
        else:
            raise NotImplementedError()

    def __rshift__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value >> other.value)
        elif is_py_value(other):
            return self.__class__(self.value >> other)
        else:
            raise NotImplementedError()

    def __rrshift__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value >> self.value)
        elif is_py_value(other):
            return self.__class__(other >> self.value)
        else:
            raise NotImplementedError()

    def __lshift__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value << other.value)
        elif is_py_value(other):
            return self.__class__(self.value << other)
        else:
            raise NotImplementedError()

    def __rlshift__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value << self.value)
        elif is_py_value(other):
            return self.__class__(other << self.value)
        else:
            raise NotImplementedError()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataType):
            return self.value == other.value
        elif is_py_value(other):
            return self.value == other
        else:
            raise NotImplementedError()

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, DataType):
            return self.value > other.value
        elif is_py_value(other):
            return self.value > other
        else:
            raise NotImplementedError()

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, DataType):
            return self.value >= other.value
        elif is_py_value(other):
            return self.value >= other
        else:
            raise NotImplementedError()

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, DataType):
            return self.value < other.value
        elif is_py_value(other):
            return self.value < other
        else:
            raise NotImplementedError()

    def __le__(self, other: Any) -> bool:
        if isinstance(other, DataType):
            return self.value <= other.value
        elif is_py_value(other):
            return self.value <= other
        else:
            raise NotImplementedError()

    def __and__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value & other.value)
        elif is_py_value(other):
            return self.__class__(self.value & other)
        else:
            raise NotImplementedError()

    def __rand__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value & self.value)
        elif is_py_value(other):
            return self.__class__(other & self.value)
        else:
            raise NotImplementedError()

    def __or__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value | other.value)
        elif is_py_value(other):
            return self.__class__(self.value | other)
        else:
            raise NotImplementedError()

    def __ror__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value | self.value)
        elif is_py_value(other):
            return self.__class__(other | self.value)
        else:
            raise NotImplementedError()

    def __xor__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value ^ other.value)
        elif is_py_value(other):
            return self.__class__(self.value ^ other)
        else:
            raise NotImplementedError()

    def __rxor__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value ^ self.value)
        elif is_py_value(other):
            return self.__class__(other ^ self.value)
        else:
            raise NotImplementedError()

    def __pow__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(self.value**other.value)
        elif is_py_value(other):
            return self.__class__(self.value**other)
        else:
            raise NotImplementedError()

    def __rpow__(self, other: Any) -> Any:
        if isinstance(other, DataType):
            return self.__class__(other.value**self.value)
        elif is_py_value(other):
            return self.__class__(other**self.value)
        else:
            raise NotImplementedError()


class BooleanType(DataType):
    bits: int = -1
    lanes: int = 1


class Bool(BooleanType):
    bits: int = 1


class IntegerType(DataType):
    pass


class IntType(IntegerType):
    pass


class I4(IntType):
    bits: int = 4
    lanes: int = 1


class I8(IntType):
    bits: int = 8
    lanes: int = 1


class I16(IntType):
    bits: int = 16
    lanes: int = 1


Short = I16


class I32(IntType):
    bits: int = 32
    lanes: int = 1


Int = I32


class I64(IntType):
    bits: int = 64
    lanes: int = 1


class I128(IntType):
    bits: int = 128
    lanes: int = 1


class UIntType(IntegerType):
    pass


class U4(UIntType):
    bits: int = 4
    lanes: int = 1


class U8(UIntType):
    bits: int = 8
    lanes: int = 1


class U16(UIntType):
    bits: int = 16
    lanes: int = 1


class U32(UIntType):
    bits: int = 32
    lanes: int = 1


class U64(UIntType):
    bits: int = 64
    lanes: int = 1


class U128(UIntType):
    bits: int = 128
    lanes: int = 1


class FloatingType(DataType):
    pass


class FloatType(FloatingType):
    pass


class F8(FloatType):
    bits: int = 8
    lanes: int = 1


class F16(FloatType):
    bits: int = 16
    lanes: int = 1


Half = F16


class F32(FloatType):
    bits: int = 32
    lanes: int = 1


Float = F32


class F64(FloatType):
    bits: int = 64
    lanes: int = 1


Double = F64


class BFloatType(FloatingType):
    pass


class BF16(BFloatType):
    bits: int = 16
    lanes: int = 1


class TFloatType(FloatingType):
    pass


class TF32(TFloatType):
    bits: int = 32
    lanes: int = 1
