from dataclasses import dataclass
from enum import Enum
from kernel import *


class PersistentTileSchedulerSm90:
    @dataclass
    class WorkTileInfo:
        M_idx: int = 0
        N_idx: int = 0
        L_idx: int = 0
        is_valid_tile: bool = False

    class RasterOrder(Enum):
        AlongM = 0
        AlongN = 1

    @dataclass
    class Arguments:
        max_swizzle_size: int = 1

    class Params:
        def __init__(self, blocks_per_problem=0, log_swizzle_size=0, raster_order=None):
            self.blocks_per_problem_ = blocks_per_problem
            self.log_swizzle_size_ = log_swizzle_size
            self.raster_order_ = (
                raster_order
                if raster_order is not None
                else PersistentTileSchedulerSm90.RasterOrder.AlongN
            )

    def __init__(self, gpu_kernel: GPUKernel) -> None:
        self.current_work_linear_idx_ = gpu_kernel.blockRange().x + gpu_kernel.blockRange().y * gpu_kernel
