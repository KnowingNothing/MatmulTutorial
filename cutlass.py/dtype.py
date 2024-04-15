from dataclasses import dataclass


class DType:
    def is_same(self, another_type: "DType"):
        raise NotImplementedError()

@dataclass
class IntegerType(DType):
    unsigned: bool = False
    bits: int = 0

    def _is(self, unsigned: bool, bits: int):
        return self.unsigned == unsigned and self.bits == bits

    def is_int4_t(self):
        return self._is(False, 4)
    
    def is_int8_t(self):
        return self._is(False, 8)
    
    def is_int16_t(self):
        return self._is(False, 16)
    
    def is_int32_t(self):
        return self._is(False, 32)
    
    def is_int64_t(self):
        return self._is(False, 64)
    
    def is_uint4_t(self):
        return self._is(True, 4)
    
    def is_uint8_t(self):
        return self._is(True, 8)
    
    def is_uint16_t(self):
        return self._is(True, 16)
    
    def is_uint32_t(self):
        return self._is(True, 32)
    
    def is_uint64_t(self):
        return self._is(True, 64)
    
    @classmethod
    def int4_t(cls):
        return cls(False, 4)
    
    @classmethod
    def int8_t(cls):
        return cls(False, 8)
    
    @classmethod
    def int16_t(cls):
        return cls(False, 16)
    
    @classmethod
    def int32_t(cls):
        return cls(False, 32)
    
    @classmethod
    def int64_t(cls):
        return cls(False, 64)
    
    @classmethod
    def uint4_t(cls):
        return cls(True, 4)
    
    @classmethod
    def uint8_t(cls):
        return cls(True, 8)
    
    @classmethod
    def uint16_t(cls):
        return cls(True, 16)
    
    @classmethod
    def uint32_t(cls):
        return cls(True, 32)
    
    @classmethod
    def uint64_t(cls):
        return cls(True, 64)
    
    def is_same(self, another_type: "IntegerType"):
        return (isinstance(another_type, IntegerType) and
                (self.unsigned == another_type.unsigned) and
                (self.bits == another_type.bits))


@dataclass
class FloatingType(DType):
    sign_bits: int = 0
    exp_bits: int = 0
    mantissa_bits: int = 0

    @property
    def bits(self):
        return self.sign_bits + self.exp_bits + self.mantissa_bits
    
    def _is(self, sign: int, exp: int, mantissa: int):
        return self.sign_bits == sign and self.exp_bits == exp and self.mantissa_bits == mantissa
    
    def is_e3m4_t(self):
        return self._is(1, 3, 4)
    
    def is_e4m3_t(self):
        return self._is(1, 4, 3)
    
    def is_fp8_t(self):
        return self.is_e3m4_t() or self.is_e4m3_t()
    
    def is_half_t(self):
        return self._is(1, 5, 10)

    def is_float_t(self):
        return self._is(1, 8, 23)
    
    def is_double_t(self):
        return self._is(1, 11, 52)
    
    @classmethod
    def e3m4_t(cls):
        return cls(1, 3, 4)
    
    @classmethod
    def e4m3_t(cls):
        return cls(1, 4, 3)
    
    @classmethod
    def half_t(cls):
        return cls(1, 5, 10)
    
    @classmethod
    def float_t(cls):
        return cls(1, 8, 23)
    
    @classmethod
    def double_t(cls):
        return cls(1, 11, 52)
    
    def is_same(self, another_type: "FloatingType"):
        return (isinstance(another_type, FloatingType) and
                (self.sign_bits == another_type.sign_bits) and
                (self.exp_bits == another_type.exp_bits) and
                (self.mantissa_bits == another_type.mantissa_bits))

def int4_t():
    return IntegerType.int4_t()

def int8_t():
    return IntegerType.int8_t()

def int16_t():
    return IntegerType.int16_t()

def int32_t():
    return IntegerType.int32_t()

def int64_t():
    return IntegerType.int64_t()

def uint4_t():
    return IntegerType.uint4_t()

def uint8_t():
    return IntegerType.uint8_t()

def uint16_t():
    return IntegerType.uint16_t()

def uint32_t():
    return IntegerType.uint32_t()

def uint64_t():
    return IntegerType.uint64_t()

def e3m4_t():
    return FloatingType.e3m4_t()

def e4m3_t():
    return FloatingType.e4m3_t()

def half_t():
    return FloatingType.half_t()

def float_t():
    return FloatingType.float_t()

def double_t():
    return FloatingType.double_t()


def is_same_type(A_type: DType, B_type: DType):
    return A_type.is_same(B_type)