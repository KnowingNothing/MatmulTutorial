def round_up(a, b):
    return (a + b - 1) // b * b


def integer_log2(x):
    n = 0
    x >>= 1
    while x:
        x = x >> 1
        n += 1
    return n


class FastDivmodU64Pow2:
    def __init__(self, divisor=0) -> None:
        self.divisor = divisor
        self.shift_right = integer_log2(divisor)

    def divide(self, dividend):
        return dividend >> self.shift_right

    def modulus(self, dividend):
        return dividend & (self.divisor - 1)

    def divmod(self, dividend):
        quotient = self.divide(dividend)
        remainder = self.modulus(dividend)
        return quotient, remainder

    def __call__(self, dividend):
        return self.divmod(dividend)


class FastDivmodU64:
    def __init__(self, divisor):
        self.divisor = divisor

    def divmod(self, x):
        return (x // self.divisor, x % self.divisor)

    def __call__(self, x):
        return self.divmod(x)
