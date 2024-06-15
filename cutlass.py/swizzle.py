class Swizzle:
    def __init__(self, num_bits: int, num_base: int, num_shft: int):
        self.num_bits = num_bits
        self.num_base = num_base
        self.num_shft = num_shft

        assert self.num_bits >= 0, "MBase must be positive."
        assert self.num_bits >= 0, "BBits must be positive."
        assert (
            abs(self.num_shft) >= self.num_bits
        ), "abs(SShift) must be more than BBits."

        self.bit_msk = (1 << self.num_bits) - 1
        self.yyy_msk = self.bit_msk << (self.num_base + max(0, self.num_shft))
        self.zzz_msk = self.bit_msk << (self.num_base - min(0, self.num_shft))
        self.msk_sft = self.num_shft

        self.swizzle_code = self.yyy_msk | self.zzz_msk

    def apply(self, offset):
        if self.msk_sft >= 0:
            return offset ^ ((offset & self.yyy_msk) >> self.msk_sft)
        else:
            return offset ^ ((offset & self.yyy_msk) << -self.msk_sft)

    def __call__(self, offset):
        return self.apply(offset)


def test_swizzle():

    def get_ind_matrix(rows, cols):
        return [[(x, y) for y in range(cols)] for x in range(rows)]

    def get_row_major_ind(x, y, rows, cols):
        return x * cols + y

    def get_row_major_tuple(xy, rows, cols):
        return (xy // cols, xy % cols)

    def get_col_major_ind(x, y, rows, cols):
        return x + y * rows

    def get_col_major_tuple(xy, rows, cols):
        return (xy % rows, xy // rows)

    def print_matrix(mtx, rows, cols, func=lambda x: x, prompt=""):
        print(prompt)
        for x in range(rows):
            for y in range(cols):
                item = mtx[x][y]
                item = func(item)
                print(item, end=" ")
            print()

    # Swizzle<3, 4, 4>
    print("Swizzle<3,4,3>")
    rows = 8
    cols = 128
    mtx = get_ind_matrix(rows, cols)
    print_matrix(mtx, rows, cols, prompt="Original")
    print()
    swizzle = Swizzle(3, 4, 3)
    print_matrix(
        mtx,
        rows,
        cols,
        lambda tp: get_row_major_tuple(
            swizzle(get_row_major_ind(tp[0], tp[1], rows, cols)), rows, cols
        ),
        prompt="After swizzle",
    )
    print()

    # Swizzle<2, 0, -2>
    print("Swizzle<2,0,-2>")
    rows = 4
    cols = 4
    mtx = get_ind_matrix(rows, cols)
    print_matrix(mtx, rows, cols, prompt="Original")
    print()
    swizzle = Swizzle(2, 0, -2)
    print_matrix(
        mtx,
        rows,
        cols,
        lambda tp: get_row_major_tuple(
            swizzle(get_row_major_ind(tp[0], tp[1], rows, cols)), rows, cols
        ),
        prompt="After swizzle",
    )
    print()


if __name__ == "__main__":
    test_swizzle()
