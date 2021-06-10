import numpy as np
import math
import logging
import trajectory_supervisor


def get_linear_interpolation(ego_path: np.ndarray,
                             grid_res: float) -> np.ndarray:
    """
    The ego-trajectory is discretized into many points, related to each other with segments. However, the distance
    between two successive points can be higher than the grid resolution. To solve this problem, linear interpolation is
    applied. This means, that intermediate points are created between every two successive points, thus dividing the
    segments in sub-segments. The length of the sub-segments must be equal or lower than the grid resolution.

    :param ego_path:            path data of the ego vehicle with the columns [pos_x, pos_y, heading (-pi/+pi)]
    :param grid_res:            resolution of the occupancy grid (cell width / height in meters)
    :returns:
        * **ego_path_interp** - array containing the coords of the pts forming the extended polygonal chain of the traj.

    :Authors:
        * Maroua Ben Lakhal
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        10.01.2020

    """

    # initialize variables
    ego_x_zw = ego_path[:, 0]
    ego_y_zw = ego_path[:, 1]
    ego_heading_zw = ego_path[:, 2]
    ego_x = [ego_x_zw[0]]
    ego_y = [ego_y_zw[0]]
    ego_heading = [ego_heading_zw[0]]

    # -- adjust the number of predicted trajectory points to the grid resolution ---------------------------------------
    for u in range(0, ego_path.shape[0] - 1):
        # calculate the euclidean distance between every two successive points of the trajectory
        v1 = (ego_x_zw[u], ego_y_zw[u])
        v2 = (ego_x_zw[u + 1], ego_y_zw[u + 1])
        dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
        dist = math.sqrt(sum(dist))

        # get number of sub-segments to generate between two successive points of the trajectory
        if dist > grid_res:
            # In order to obtain the number of required points, we need to add 2:
            # 1 for additional sub segment due to rounding errors
            # 1 to obtain the number of required points from the number of required sub-segments
            n_steps = int(dist / grid_res) + 2
        else:
            # no need for interpolation if the euclidean distance between two successive points is inferior to the grid
            # resolution
            n_steps = 2

        # interpolate all segments of the trajectory: linear interpolation for each coordinate
        # (finer than grid resolution)
        x_steps = np.linspace(ego_x_zw[u], ego_x_zw[u + 1], n_steps)
        y_steps = np.linspace(ego_y_zw[u], ego_y_zw[u + 1], n_steps)

        # check if heading is crossing the -pi / +pi border --> interpolate accordingly
        if (abs(ego_heading_zw[u] - ego_heading_zw[u + 1])
                < abs(min(ego_heading_zw[u], ego_heading_zw[u + 1]) + 2 * math.pi
                      - max(ego_heading_zw[u], ego_heading_zw[u + 1]))):
            heading_steps = np.linspace(ego_heading_zw[u], ego_heading_zw[u + 1], n_steps)
        else:
            if ego_heading_zw[u] < ego_heading_zw[u + 1]:
                heading_steps = ((np.linspace(ego_heading_zw[u] + 2 * math.pi, ego_heading_zw[u + 1], n_steps)
                                  + math.pi) % 2 * math.pi) - math.pi
            else:
                heading_steps = ((np.linspace(ego_heading_zw[u], ego_heading_zw[u + 1] + 2 * math.pi, n_steps)
                                  + math.pi) % 2 * math.pi) - math.pi

        # form gradually the extended polygonal chain of the trajectory
        ego_x.extend(x_steps[1:])
        ego_y.extend(y_steps[1:])
        ego_heading.extend(heading_steps[1:])

    # convert back to numpy array
    ego_path_interp = np.column_stack((ego_x, ego_y, ego_heading))

    return ego_path_interp


def get_cell_id(origin: np.ndarray,
                pos_x: float,
                pos_y: float,
                grid_res: float) -> tuple:
    """
    This function locates the point on the grid and return the indices of the row and column of the cell containing the
    point.

    :param origin:          x, y coordinates of the origin (0, 0) in the occupancy grid
    :param pos_x:           x coordinates of a trajectory point
    :param pos_y:           y coordinates of a trajectory point
    :param grid_res:        resolution of the occupancy grid (cell width / height in meters)
    :returns:
        * **row** -         row-index of the cell containing the trajectory point
        * **column** -      column-index of the cell containing the trajectory point

    :Authors:
        * Maroua Ben Lakhal
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        10.01.2020

    """

    # particularity: the origin of the grid correspond to the lowest x value and the highest y value of the grid ranges
    row = int((origin[1] - pos_y) / grid_res)
    column = int((pos_x - origin[0]) / grid_res)

    return row, column


def get_helpers(grid_res: float,
                pos: float,
                direction: float,
                cell: int) -> tuple:
    """
    Get the direction of the moving unit and the distance from the current trajectory point to the next cell border.

    :param grid_res:             resolution of the occupancy grid (cell width / height in meters)
    :param pos:                  coordinates of the current trajectory point
    :param direction:            translation vector between two successive points of a trajectory
    :param cell:                 indices of the cell containing the trajectory point
    :returns:
        * **delta_cell** -       direction of moving unit
        * **dt** -               distances to next borders

    :Authors:
        * Maroua Ben Lakhal
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        10.01.2020

    """

    if direction > 0:
        delta_cell = 1
        dt = ((cell + 1) * grid_res - pos) / direction
    else:
        delta_cell = -1
        dt = (cell * grid_res - pos) / direction

    return delta_cell, dt


def ray_casting__all_directions(pos_x: float,
                                pos_y: float,
                                direction_x: float,
                                direction_y: float,
                                column: int,
                                row: int,
                                occ_map: dict) -> tuple:
    """
    Only interpolated points of the trajectory segments are located in the grid. However, the trajectory passes through
    other cells of the grid. In order to find those cells, an intermediate point between two successive points of a
    trajectory is created using the ray tracing method and located in the grid and the indices of the cell containing
    this point are returned through this function.

    .. note::
        More infos about the ray tracing method can be found under:
        https://theshoemaker.de/2016/02/ray-casting-in-2d-grids/

        Source:
        https://github.com/pfirsich/Ray-casting-test (modified)

    :param pos_x:           x-coordinate of the current trajectory point relative to the origin (in meters)
    :param pos_y:           y-coordinate of the current trajectory point relative to the origin (in meters)
    :param direction_x:     x translation vector between two successive points of a trajectory (in meters)
    :param direction_y:     y translation vector between two successive points of a trajectory (in meters)
    :param column:          column index of the cell containing the current trajectory point
    :param row:             row index of the cell containing the current trajectory point
    :param occ_map:         dict with the following keys
                                * grid:           occupancy grid (np-array) holding 1 for occ. and 0 for unocc. cells
                                * origin:         x, y coordinates of the origin (0, 0) in the occupancy grid
                                * resolution:     grid resolution in meters

    :returns:
        * **row** -         row index of the cell containing the ray point between two successive points of the traj.
        * **column** -      column index of the cell containing the ray point between two successive points of the traj.

    :Authors:
        * Maroua Ben Lakhal
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        10.01.2020

    """

    # position in relation to grid
    pos_x_grid = pos_x - occ_map['origin'][0]
    pos_y_grid = occ_map['origin'][1] - pos_y

    # get the direction of the moving unit and the distance from the current trajectory point to the next cell border in
    # x and y direction
    delta_column, dt_x = get_helpers(occ_map['resolution'], pos_x_grid, direction_x, column)
    delta_row, dt_y = get_helpers(occ_map['resolution'], pos_y_grid, direction_y, row)

    # determine the decisive translation direction based on the distance to next border
    if dt_x < dt_y:
        if 0 < column < occ_map['grid'].shape[1] - 1:
            column = column + delta_column
    else:
        if 0 < row < occ_map['grid'].shape[0] - 1:
            row = row + delta_row

    return row, column


def check_collision_occ_map(ego_path: np.ndarray,
                            occ_map: dict,
                            col_width: float,
                            plot_occ: bool = False) -> bool:
    """
    This function asses the trajectory safety for a given times instance regarding collision with the wall or any static
    obstacle on the race track. Only interpolated points of the trajectory segments are located in the grid and the
    collision is only checked in those cells. However, the trajectory passes through other cells of the grid, which may
    contain a static obstacle and must be considered for collision detection. In order to identify those cells, an
    intermediate point between two successive points of a trajectory is created and located on the grid.

    .. note::
        More infos about the ray tracing method can be found under:
        https://theshoemaker.de/2016/02/ray-casting-in-2d-grids/

        Source:
        https://github.com/pfirsich/Ray-casting-test (modified)

    :param ego_path:         path data of the ego vehicle with the following columns: [pos_x, pos_y, heading]
    :param occ_map:          dict with the following keys
                                * grid:           occupancy grid (np-array) holding 1 for occ. and 0 for unocc. cells
                                * origin:         x, y coordinates of the origin (0, 0) in the occupancy grid
                                * resolution:     grid resolution in meters
    :param col_width:        tracked width for collisions (with the trajectory lying in the center -> for minimal safety
                             use vehicle width, for conservative safety use vehicle diagonal)
    :param plot_occ:         (optional) if set to 'True', the occupancy grid will be plotted once a collision is
                             detected (use for debugging only!)
    :returns:
        * **safety** -              boolean 'False' for unsafe, 'True' for safe

    :Authors:
        * Maroua Ben Lakhal
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        10.01.2020

    """

    # --initialize the safety variable to 1, the path is expected to be safe before the check --------------------------
    safety = True

    # -- get coordinates of points forming the trajectory from the string holding the data of the ego vehicle ----------
    # first case: grid resolution is 1
    if occ_map['resolution'] <= 1.0:
        # The linear interpolation avoids the while loop needed to launch ray tracing so long until the next point is
        # reached, resulting in faster computation times.
        # This ensures that the next point of the trajectory is located in one of the 8 cells around the cell containing
        # the current trajectory point. As a result, ray tracing must be applied only once for a point.
        # The efficiency of this approach is validated using plots and examples in the thesis report.
        ego_path_interp = get_linear_interpolation(ego_path, occ_map['resolution'])
    # second case: grid resolution is bigger than one
    else:
        raise ValueError("Risk of no accurate safety results! Please give a finer grid resolution")

    # -- calculate paths in between the vehicle with with at least grid size resolution --------------------------------
    # determine shift vector (along vectors, the paths are shifted (pi/2 rotation is skipped, since def. is in -pi/+pi)
    shift_vector = np.column_stack((np.cos(ego_path_interp[:, 2]), np.sin(ego_path_interp[:, 2])))

    ego_paths = []
    for shift in np.linspace(-col_width / 2, col_width / 2, max(math.ceil(col_width / occ_map['resolution']), 1)):
        ego_paths.append(np.column_stack((ego_path_interp[:, 0] + shift_vector[:, 0] * shift,
                                          ego_path_interp[:, 1] + shift_vector[:, 1] * shift)))

    # checked_cell_id_row = []
    # checked_cell_id_column = []
    for ego_path_i in ego_paths:
        for i in range(ego_path_i.shape[0]):
            if safety:
                x_i0 = ego_path_i[i, 0]
                y_i0 = ego_path_i[i, 1]

                if i < ego_path_i.shape[0] - 1:
                    x_i1 = ego_path_i[i + 1, 0]
                    y_i1 = ego_path_i[i + 1, 1]
                else:
                    x_i1 = None
                    y_i1 = None

                # -- Assumption: the points to be checked are all inside th occupancy grid -----------------------------
                # get the indices of the cell of the occupancy grid containing each trajectory point
                row_i, column_i = get_cell_id(occ_map['origin'], x_i0, y_i0, occ_map['resolution'])
                # checked_cell_id_row.append(row_i)
                # checked_cell_id_column.append(column_i)

                # check if requested position within grid-map
                if row_i >= occ_map['grid'].shape[0] or column_i >= occ_map['grid'].shape[1]:
                    log = logging.getLogger("supervisor_logger")
                    log.info('supmod_static_collision | Could not determine static safety, since requested cell '
                             + str([row_i, column_i]) + ' is outside of grid-map with size '
                             + str(occ_map['grid'].shape) + ' !')
                    break

                if occ_map['grid'][row_i, column_i] == 1:
                    safety = False

                    log = logging.getLogger("supervisor_logger")
                    log.warning('supmod_static_collision | Collision detected! '
                                'Cell containing point ({:6.2f}, {:6.2f}) is occupied.'.format(x_i0, y_i0))
                    break

                if x_i1 is not None:
                    # get the indices of the cell of the occupancy grid containing the next trajectory point
                    row_next, column_next = get_cell_id(occ_map['origin'], x_i1, y_i1, occ_map['resolution'])

                    # Apply ray tracing. However, ray tracing can't be applied in the following cases:
                    # case1: current and successive points of the polygonal chain are contained in the same cell
                    # case2: the line between the centers of the cells containing the current and next trajectory points
                    #        is a vertical or a horizontal line
                    if (not ((row_i == row_next) and (abs(column_i - column_next) == 1))) and \
                       (not ((column_i == column_next) and (abs(row_i - row_next) == 1))) and \
                       (not ((row_i == row_next) and (column_i == column_next))):

                        # calculate translation vector between two successive points of trajectory in x, y direction
                        direction_x = x_i1 - x_i0
                        direction_y = y_i1 - y_i0

                        # get indices of cell containing the intermediate point identified via ray tracing
                        row_ray, column_ray = ray_casting__all_directions(pos_x=x_i0,
                                                                          pos_y=y_i0,
                                                                          direction_x=direction_x,
                                                                          direction_y=-direction_y,  # flipped axis!
                                                                          column=column_i,
                                                                          row=row_i,
                                                                          occ_map=occ_map)
                        # checked_cell_id_row.append(row_ray)
                        # checked_cell_id_column.append(column_ray)

                        if occ_map['grid'][row_ray, column_ray] == 1 and safety:
                            safety = False

                            log = logging.getLogger("supervisor_logger")
                            log.warning('supmod_static_collision | Collision detected! Collision in vicinity of the '
                                        'point ({:6.2f}, {:6.2f}) detected with ray tracing.'.format(x_i0, y_i0))

                            if plot_occ:
                                trajectory_supervisor.supervisor_modules.supmod_static_collision.src.\
                                    visualize_occupancy.visualize_occupancy(occ_map=occ_map,
                                                                            trajectory_cells=[[row_ray], [column_ray]],
                                                                            trajectory_coords=ego_path[:, 0:2])

    return safety
