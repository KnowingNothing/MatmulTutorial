from dataclasses import dataclass
from enum import Enum
from tiling import HyperCube, HyperPoint
from fast_math import FastDivmodU64Pow2, round_up, FastDivmodU64
from hw_info import KernelHardwareInfo, dim3, DeviceCoord


class TileParams:
    class RasterOrder(Enum):
        AlongM = (0,)
        AlongN = 1

    class RasterOrderOptions(Enum):
        Heuristic = (0,)
        AlongM = (1,)
        AlongN = 2

    def __init__(
        self, blocks_per_problem=0, log_swizzle_size=0, raster_order=RasterOrder.AlongN
    ) -> None:
        self.blocks_per_problem = blocks_per_problem
        self.log_swizzle_size = log_swizzle_size
        self.raster_order = raster_order
        self.divmod_batch = None
        self.divmod_cluster_shape_major = None
        self.divmod_cluster_shape_minor = None
        self.divmod_cluster_blk_major = None

    def initialize(
        self,
        problem_shape: HyperCube,  # m, n, k, l
        tile_shape: HyperCube,
        cluster_shape: HyperCube,
        hw_info: KernelHardwareInfo,
        max_swizzle_size: int,
        raster_order_option: RasterOrderOptions,
    ):
        problem_blocks: dim3 = TileParams.get_tiled_cta_shape_mnl(
            problem_shape, tile_shape, cluster_shape
        )
        self._initialize(
            problem_blocks,
            cluster_shape,
            hw_info,
            max_swizzle_size,
            raster_order_option,
        )

    def _initialize(
        self,
        problem_blocks: dim3,
        cluster_shape: HyperCube,
        hw_info: KernelHardwareInfo,
        max_swizzle_size: int,
        raster_order_option: "TileParams.RasterOrderOptions",
    ):
        log_swizzle_size = self.get_log_swizzle_size(
            problem_blocks.x, problem_blocks.y, max_swizzle_size
        )
        problem_blocks_m = round_up(
            problem_blocks.x, (1 << log_swizzle_size) * cluster_shape[0]
        )
        problem_blocks_n = round_up(
            problem_blocks.y, (1 << log_swizzle_size) * cluster_shape[1]
        )

        raster_order = self.get_rasterization_order(
            problem_blocks_m, problem_blocks_n, raster_order_option
        )

        self.blocks_per_problem = problem_blocks_m * problem_blocks_n * problem_blocks.z
        self.log_swizzle_size = log_swizzle_size
        self.raster_order = raster_order

        self.divmod_batch = FastDivmodU64(problem_blocks_m * problem_blocks_n)

        if raster_order == TileParams.RasterOrder.AlongN:
            self.divmod_cluster_shape_major = FastDivmodU64Pow2(cluster_shape[1])
            self.divmod_cluster_shape_minor = FastDivmodU64Pow2(cluster_shape[0])
            self.divmod_cluster_blk_major = FastDivmodU64(
                problem_blocks_n // cluster_shape[1]
            )

        else:
            self.divmod_cluster_shape_major = FastDivmodU64Pow2(cluster_shape[0])
            self.divmod_cluster_shape_minor = FastDivmodU64Pow2(cluster_shape[1])
            self.divmod_cluster_blk_major = FastDivmodU64(
                problem_blocks_m // cluster_shape[0]
            )

    @staticmethod
    def get_grid_shape(
        problem_shape: HyperCube,
        cta_shape: HyperCube,
        cluster_shape: HyperCube,
        hw_info: KernelHardwareInfo,
        max_swizzle_size: int,
        raster_order_option: "TileParams.RasterOrderOptions",
        truncate_by_problem_size: bool = True,
    ):
        problem_blocks: dim3 = TileParams.get_tiled_cta_shape_mnl(
            problem_shape, cta_shape, cluster_shape
        )
        return TileParams._get_grid_shape(
            problem_blocks,
            cluster_shape,
            hw_info,
            max_swizzle_size,
            raster_order_option,
            truncate_by_problem_size,
        )

    @staticmethod
    def _get_grid_shape(
        problem_blocks: dim3,
        cluster_shape: HyperCube,
        hw_info: KernelHardwareInfo,
        max_swizzle_size: int,
        raster_order_option: "TileParams.RasterOrderOptions",
        truncate_by_problem_size: bool = True,
    ):
        sm_count = hw_info.sm_count

        log_swizzle_size = TileParams.get_log_swizzle_size(
            problem_blocks.x, problem_blocks.y, max_swizzle_size
        )
        problem_blocks_m = round_up(
            problem_blocks.x, (1 << log_swizzle_size) * cluster_shape[0]
        )
        problem_blocks_n = round_up(
            problem_blocks.y, (1 << log_swizzle_size) * cluster_shape[1]
        )
        problem_blocks_total = problem_blocks_m * problem_blocks_n * problem_blocks.z

        raster_order: TileParams.RasterOrder = TileParams.get_rasterization_order(
            problem_blocks_m, problem_blocks_n, raster_order_option
        )

        launch_grid: dim3 = dim3(0, 0, 0)
        if raster_order == TileParams.RasterOrder.AlongN:
            launch_grid = dim3(cluster_shape[0], 1, 1)
        else:
            launch_grid = dim3(1, cluster_shape[1], 1)

        def possibly_truncate(x, y):
            if truncate_by_problem_size:
                return min(x, y)
            return x

        cluster_size = cluster_shape[0] * cluster_shape[1]
        if cluster_size == 1:
            if raster_order == TileParams.RasterOrder.AlongN:
                launch_grid.y = possibly_truncate(sm_count, problem_blocks_total)
            else:
                launch_grid.x = possibly_truncate(sm_count, problem_blocks_total)
        else:
            max_sm_per_gpc = 2 * 9
            min_num_gpc = 1 if sm_count < max_sm_per_gpc else sm_count // max_sm_per_gpc
            max_cta_occupancy_per_gpc = max_sm_per_gpc - (max_sm_per_gpc % cluster_size)
            cta_per_device = min_num_gpc * max_cta_occupancy_per_gpc

            num_gpc_residual = (
                0 if sm_count < max_sm_per_gpc else sm_count % max_sm_per_gpc
            )
            max_cta_occupancy_per_residual_gpc = num_gpc_residual - (
                num_gpc_residual % cluster_size
            )
            cta_per_device += max_cta_occupancy_per_residual_gpc

            if raster_order == TileParams.RasterOrder.AlongN:
                launch_grid.y = possibly_truncate(
                    cta_per_device // cluster_shape[0],
                    problem_blocks_total // cluster_shape[0],
                )
            else:
                launch_grid.x = possibly_truncate(
                    cta_per_device // cluster_shape[1],
                    problem_blocks_total // cluster_shape[1],
                )
        return launch_grid

    @staticmethod
    def get_log_swizzle_size(problem_ctas_m, problem_ctas_n, max_swizzle_size):
        min_cta_dim = min(problem_ctas_m, problem_ctas_n)
        if max_swizzle_size >= 8 and min_cta_dim >= 6:
            return 3
        elif max_swizzle_size >= 4 and min_cta_dim >= 3:
            return 2
        elif max_swizzle_size >= 2 and min_cta_dim >= 2:
            return 1
        else:
            return 0

    @staticmethod
    def get_rasterization_order(tiles_m, tiles_n, raster_order_option):
        if raster_order_option == TileParams.RasterOrderOptions.Heuristic:
            if tiles_n > tiles_m:
                return TileParams.RasterOrder.AlongM
            else:
                return TileParams.RasterOrder.AlongN
        else:
            if raster_order_option == TileParams.RasterOrderOptions.AlongN:
                return TileParams.RasterOrder.AlongN
            else:
                return TileParams.RasterOrder.AlongM

    @staticmethod
    def get_tiled_cta_shape_mnl(problem_shape, cta_shape, cluster_shape):
        cta_m = (problem_shape[0] + cta_shape[0] - 1) // cta_shape[0]
        cta_n = (problem_shape[1] + cta_shape[1] - 1) // cta_shape[1]

        return TileParams._get_tiled_cta_shape_mnl(
            problem_shape, cluster_shape, cta_m, cta_n
        )

    @staticmethod
    def _get_tiled_cta_shape_mnl(problem_shape, cluster_shape, cta_m, cta_n):
        problem_blocks_m = (
            (cta_m + cluster_shape[0] - 1) // cluster_shape[0] * cluster_shape[0]
        )
        problem_blocks_n = (
            (cta_n + cluster_shape[1] - 1) // cluster_shape[1] * cluster_shape[1]
        )

        return dim3(problem_blocks_m, problem_blocks_n, problem_shape[3])


@dataclass
class TileSchedulerArguments:
    max_swizzle_size: int = 1
    raster_order: TileParams.RasterOrderOptions = (
        TileParams.RasterOrderOptions.Heuristic
    )


class TileScheduler:

    @dataclass
    class WorkTileInfo:
        M_idx: int = 0
        N_idx: int = 0
        L_idx: int = 0
        is_valid_tile: bool = False

        def is_valid(self):
            return self.is_valid_tile

        @classmethod
        def invalid_work_tile(cls):
            return cls(-1, -1, -1, False)

        def is_final_split(self, k_tiles_per_output_tile):
            return True

        def reduction_subtile_idx(self):
            return -1

    def __init__(self, params: TileParams, dev_coord: DeviceCoord):
        self.schedule_params = params
        self.dev_coord = dev_coord
        if params.raster_order == TileParams.RasterOrder.AlongN:
            self.current_work_linear_idx = (
                dev_coord.blockIdx.x + dev_coord.blockIdx.y * dev_coord.gridDim.x
            )
        else:
            self.current_work_linear_idx = (
                dev_coord.blockIdx.x * dev_coord.gridDim.y + dev_coord.blockIdx.y
            )

        self.total_grid_size = (
            dev_coord.gridDim.x * dev_coord.gridDim.y * dev_coord.gridDim.z
        )

    @staticmethod
    def to_underlying_arguments(
        problem_shape_mnkl,
        tile_shape,
        cluster_shape,
        hw_info,
        arguments,
        workspace=None,
        epilogue_subtile=1,
    ):
        params = TileParams()
        params.initialize(
            problem_shape_mnkl,
            tile_shape,
            cluster_shape,
            hw_info,
            arguments.max_swizzle_size,
            arguments.raster_order,
        )
        return params

    def get_current_work(self):
        return self.get_current_work_for_linear_idx(self.current_work_linear_idx)

    def get_current_work_for_linear_idx(self, linear_idx):
        if linear_idx >= self.schedule_params.blocks_per_problem:
            return TileScheduler.WorkTileInfo.invalid_work_tile()

        work_idx_l, remainder = self.schedule_params.divmod_batch(linear_idx)
        blk_per_grid_dim, _ = self.schedule_params.divmod_cluster_shape_minor(remainder)
        work_idx_m, work_idx_n = TileScheduler.get_work_idx_m_and_n(
            self.dev_coord,
            blk_per_grid_dim,
            self.schedule_params.divmod_cluster_shape_major,
            self.schedule_params.divmod_cluster_shape_minor,
            self.schedule_params.divmod_cluster_blk_major,
            self.schedule_params.log_swizzle_size,
            self.schedule_params.raster_order,
        )
        return TileScheduler.WorkTileInfo(work_idx_m, work_idx_n, work_idx_l, True)

    def advance_to_next_work(self, advance_count=1):
        self.current_work_linear_idx += advance_count * self.total_grid_size

    @staticmethod
    def get_work_idx_m_and_n(
        dev_coord,
        blk_per_grid_dim,
        divmod_cluster_shape_major,
        divmod_cluster_shape_minor,
        divmod_cluster_blk_major,
        log_swizzle_size,
        raster_order,
    ):
        cluster_id, cluster_major_offset = divmod_cluster_shape_major(blk_per_grid_dim)
        cta_m_in_cluster, cta_n_in_cluster, _ = dev_coord.block_id_in_cluster()
        if raster_order == TileParams.RasterOrder.AlongN:
            cluster_minor_offset = cta_m_in_cluster
        else:
            cluster_minor_offset = cta_n_in_cluster

        offset = cluster_id & ((1 << log_swizzle_size) - 1)
        extra = cluster_id >> log_swizzle_size

        cluster_idx_minor_div_swizzle, cluster_idx_major = divmod_cluster_blk_major(
            extra
        )
        cluster_idx_minor = (
            cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset
        )
        minor_work_idx = (
            cluster_idx_minor * divmod_cluster_shape_minor.divisor
            + cluster_minor_offset
        )
        major_work_idx = (
            cluster_idx_major * divmod_cluster_shape_major.divisor
            + cluster_major_offset
        )

        if raster_order == TileParams.RasterOrder.AlongN:
            return (minor_work_idx, major_work_idx)
        else:
            return {major_work_idx, minor_work_idx}

    @staticmethod
    def get_tiled_cta_shape_mnl(problem_shape_mnkl, cta_shape, cluster_shape):
        return TileParams.get_tiled_cta_shape_mnl(
            problem_shape_mnkl, cta_shape, cluster_shape
        )

    @staticmethod
    def get_grid_shape(
        problem_shape_mnk,
        cta_shape,
        cluster_shape,
        hw_info,
        arguments,
        truncate_by_problem_size=True,
    ):
        problem_shape_mnkl = problem_shape_mnk.append_and_get(1)
        return TileParams.get_grid_shape(
            problem_shape_mnkl,
            cta_shape,
            cluster_shape,
            hw_info,
            arguments.max_swizzle_size,
            arguments.raster_order,
            truncate_by_problem_size,
        )


@dataclass
class SimpleTileScheduler:
    dev_coord: DeviceCoord
    cluster_m: int
    cluster_n: int
    block_m: int
    block_n: int
    linear_idx: int = 0
    m_blocks: int = 0
    n_blocks: int = 0

    @dataclass
    class WorkInfo:
        m_idx: int
        n_idx: int
        valid: bool

    def init(self, M, N):
        self.linear_idx = (
            dev_coord.blockIdx.x + dev_coord.blockIdx.y * dev_coord.gridDim.x
        )
        self.get_blocks_m_n(M, N)

    def get_current_work_info(self):
        m_idx, n_idx = self.get_current_m_n_idx()
        return SimpleTileScheduler.WorkInfo(
            m_idx, n_idx, self.linear_idx < self.m_blocks * self.n_blocks
        )

    def advance(self, number=1):
        self.linear_idx += number * self.dev_coord.gridDim.x * self.dev_coord.gridDim.y

    def get_current_m_n_idx(self):
        div_cluster_x = self.linear_idx // self.cluster_m
        mod_cluster_x = self.linear_idx % self.cluster_m
        div_cluster_xy = div_cluster_x // self.cluster_n
        mod_cluster_xy = div_cluster_x % self.cluster_n
        cluster_per_row = self.n_blocks // self.cluster_n
        cluster_row = div_cluster_xy // cluster_per_row
        cluster_col = div_cluster_xy % cluster_per_row
        m_idx = cluster_row * self.cluster_m + mod_cluster_x
        n_idx = cluster_col * self.cluster_n + mod_cluster_xy
        return (m_idx, n_idx)

    def get_blocks_m_n(self, M, N):
        self.m_blocks = (
            ((M + self.block_m - 1) // self.block_m + self.cluster_m - 1)
            // self.cluster_m
            * self.cluster_m
        )
        self.n_blocks = (
            ((N + self.block_n - 1) // self.block_n + self.cluster_n - 1)
            // self.cluster_n
            * self.cluster_n
        )


if __name__ == "__main__":
    problem_shape_mnk = HyperCube(3, [5120, 4096, 2048])
    problem_shape_mnkl = problem_shape_mnk.append_and_get(1)
    cta_shape = HyperCube(3, [128, 128, 64])
    cluster_shape = HyperCube(3, [2, 1, 1])
    hw_info = KernelHardwareInfo(
        0, KernelHardwareInfo.query_device_multiprocessor_count()
    )
    arguments = TileSchedulerArguments()
    gridDim = TileScheduler.get_grid_shape(
        problem_shape_mnk, cta_shape, cluster_shape, hw_info, arguments
    )
    print(gridDim)
    blockDim = dim3(128 * 3, 1, 1)
    clusterDim = dim3(2, 1, 1)
    dev_coord = DeviceCoord(gridDim, gridDim, blockDim, clusterDim)
    params = TileScheduler.to_underlying_arguments(
        problem_shape_mnkl, cta_shape, cluster_shape, hw_info, arguments
    )
    scheduler = TileScheduler(params, dev_coord)

    scheduler_matrix = []
    simple_scheduler_matrix = []
    for bx in range(gridDim.x):
        schedulers = []
        simple_schedulers = []
        for by in range(gridDim.y):
            dev_coord = DeviceCoord(gridDim, gridDim, blockDim, clusterDim)
            dev_coord.set_blockIdx(bx, by, 0)
            scheduler = TileScheduler(params, dev_coord)
            schedulers.append(scheduler)
            simple_scheduler = SimpleTileScheduler(
                dev_coord,
                cluster_shape[0],
                cluster_shape[1],
                cta_shape[0],
                cta_shape[1],
            )
            simple_scheduler.init(problem_shape_mnk[0], problem_shape_mnk[1])
            simple_schedulers.append(simple_scheduler)
        scheduler_matrix.append(schedulers)
        simple_scheduler_matrix.append(simple_schedulers)

    # scheduler_matrix[0][2].get_current_work()

    for repeat in range(2):
        for row_id in range(len(scheduler_matrix)):
            for col_id in range(len(scheduler_matrix[row_id])):
                sch = scheduler_matrix[row_id][col_id]
                simple_sch = simple_scheduler_matrix[row_id][col_id]
                golden = sch.get_current_work()
                answer = simple_sch.get_current_work_info()
                print(
                    "G",
                    sch.dev_coord.blockIdx,
                    "linear_idx=",
                    sch.current_work_linear_idx,
                    "(m, n ,l)=",
                    golden,
                )
                print(
                    "A",
                    simple_sch.dev_coord.blockIdx,
                    "linear_idx=",
                    simple_sch.linear_idx,
                    "(m, n ,l)=",
                    answer,
                )
                assert golden.M_idx == answer.m_idx
                assert golden.N_idx == answer.n_idx
                assert golden.is_valid_tile == answer.valid
                sch.advance_to_next_work()
                simple_sch.advance()
        print()
