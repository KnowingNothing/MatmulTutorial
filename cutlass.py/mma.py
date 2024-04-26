from dtype import DType, FloatingType, half_t, float_t
from enum import Enum
from tiling import HyperCube, HyperPoint
from dataclasses import dataclass
from hw_info import DeviceCoord, WARP_SIZE, WARP_PER_WARP_GROUP, WARP_GROUP_SIZE


class GmmaMajor(Enum):
    MajorMN = 0
    MajorK = 1


@dataclass
class GmmaDescriptor:
    start_address_14_bits: int = 0
    # 2 bits not used
    leading_byte_offset_14_bits: int = 0
    # 2 bits not used
    stride_byte_offset_14_bits: int = 0
    # 2 bits not used
    # 1 bit not used
    base_offset_3_bits: int = 0
    # 4 bits not used
    # 6 bits not used
    layout_type_2_bits: int = 0


class GmmaDescriptorIterator:
    pass


@dataclass
class SmemDesc(GmmaDescriptorIterator):
    gmma_major: GmmaMajor = GmmaMajor.MajorMN


@dataclass
class MMA_OP:
    M_tile: int = 0
    N_tile: int = 0
    K_tile: int = 0
    A_dtype: FloatingType = float_t()
    B_dtype: FloatingType = float_t()
    Accum_dtype: FloatingType = float_t()
    A_transpose: bool = False
    B_transpose: bool = False
    A_scale: int = 1
    B_scale: int = 1


class UniversalFMA(MMA_OP):
    pass


class SM90_SS(MMA_OP):
    
    def get_accumulator_matrix(self):
        assert self.M_tile == 64
        assert self.A_dtype.is_same(half_t())
        assert self.B_dtype.is_same(half_t())
        assert self.Accum_dtype.is_same(float_t())
        fragment_matrix = []
        for m in range(self.M_tile):
            rows = []
            for n in range(self.N_tile):
                rows.append([m,n,-1])
            fragment_matrix.append(rows)
        bits_per_element = self.Accum_dtype.bits
        elements_per_8_bytes = 64 // bits_per_element
        row_threads_per_part = 4
        elements_per_row_part = elements_per_8_bytes * row_threads_per_part
        col_threads_per_part = WARP_SIZE // row_threads_per_part
        number_row_parts = self.N_tile // elements_per_row_part
        number_col_parts = self.M_tile // col_threads_per_part
        number_warps_in_col = WARP_PER_WARP_GROUP
        number_col_parts_per_warp = number_col_parts // number_warps_in_col
        
        for tid in range(WARP_GROUP_SIZE):
            warp_id = tid // WARP_SIZE
            lane_id = tid % WARP_SIZE
            for row_id in range(number_col_parts_per_warp):
                for col_id in range(number_row_parts):
                    for item_id in range(elements_per_8_bytes):
                        x = warp_id * number_col_parts_per_warp * col_threads_per_part + row_id * col_threads_per_part + lane_id // row_threads_per_part
                        y = col_id * row_threads_per_part * elements_per_8_bytes + lane_id % row_threads_per_part * elements_per_8_bytes + item_id
                        fragment_matrix[x][y][2] = tid
        return fragment_matrix
                


class MMA_Traits:
    def __init__(self, mma_op: MMA_OP) -> None:
        self.mma_op = mma_op
        self.D_dtype = mma_op.Accum_dtype
        self.A_dtype = mma_op.A_dtype
        self.B_dtype = mma_op.B_dtype
        self.C_dtype = mma_op.Accum_dtype

        A_major = GmmaMajor.MajorK if not mma_op.A_transpose else GmmaMajor.MajorMN
        B_major = GmmaMajor.MajorK if not mma_op.B_transpose else GmmaMajor.MajorMN
        self.A_frag_type = SmemDesc(A_major)
        self.B_frag_type = SmemDesc(B_major)
        
        self.MNK_shape = HyperCube(3, [mma_op.M_tile, mma_op.N_tile, mma_op.K_tile])


def gmma_selector(
    A_dtype: DType,
    B_dtype: DType,
    C_dtype: DType,
    MNK_tiling: HyperCube,
    A_major: GmmaMajor,
    B_major: GmmaMajor,
):
    assert MNK_tiling.ndim == 3
    M_tile = MNK_tiling[0]
    N_tile = MNK_tiling[1]
    K_tile = MNK_tiling[2]
    assert (
        not MNK_tiling.has_dynamic()
    ), "Tiling info shouldn't have dynamic at GMMA level."
    assert M_tile % 64 == 0, "GMMA tiling M dimension should be multiple of 64."
    A_transpose = False if A_major == GmmaMajor.MajorK else True
    B_transpose = False if B_major == GmmaMajor.MajorK else True

    # FP32 accumulator
    if C_dtype.is_same(float_t()):
        # FP16 inputs
        if A_dtype.is_same(half_t()):
            assert A_dtype.is_same(B_dtype), "A and B should have the same dtype."
            assert K_tile % 16 == 0, "GMMA tiling K dimension should be multiple of 16."
            assert N_tile % 8 == 0, "GMMA tiling N dimension should be multiple of 8."
            switch_table = [
                (
                    k,
                    SM90_SS(
                        M_tile=64,
                        N_tile=k,
                        K_tile=16,
                        A_dtype=half_t(),
                        B_dtype=half_t(),
                        Accum_dtype=float_t(),
                        A_transpose=A_transpose,
                        B_transpose=B_transpose,
                    ),
                )
                for k in [256, 192, 128, 96, 64, 32, 16, 8]
            ]
            for k, v in switch_table:
                if N_tile % k == 0:
                    return v
    raise RuntimeError(
        f"Selector can't find proper GMMA operations for:\n"
        f"A_dtype: {A_dtype}, B_dtype: {B_dtype}, C_dtype: {C_dtype}\n"
        f"MNK_tiling: {MNK_tiling}, A_major: {A_major}, B_major: {B_major}\n"
    )

if __name__ == "__main__":
    wgmma_op = SM90_SS(
                        M_tile=64,
                        N_tile=128,
                        K_tile=16,
                        A_dtype=half_t(),
                        B_dtype=half_t(),
                        Accum_dtype=float_t(),
                        A_transpose=False,
                        B_transpose=False,
                    )
    frag = wgmma_op.get_accumulator_matrix()
    
    def print_matrix(mtx, rows, cols, func=lambda x: x, prompt=""):
        print(prompt)
        for x in range(rows):
            for y in range(cols):
                item = mtx[x][y]
                item = func(item)
                print(item, end=" ")
            print()
    
    print_matrix(frag, 64, 128, func=lambda x: f"T{x[2]}({x[0]},{x[1]})", prompt="Fragment WgMMA M64N128K16:")