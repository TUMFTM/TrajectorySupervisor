import numpy as np
from matplotlib import pyplot


def visualize_occupancy(occ_map: dict,
                        trajectory_cells: list,
                        trajectory_coords: np.array):
    """
    Visualizes a grid-map based collision check for a given trajectory.

    * Grid map is visualized in black / white
    * Investigated trajectory cells are visualized in light red
    * Cells occupied by trajectory and static obstacle are highlighted in yellow
    * The original trajectory path is visualized by a solid red line

    :param occ_map:             dict with the following keys
                                    * grid:           occupancy grid (np-array) holding 1 for occ. and 0 for unocc. cell
                                    * origin:         x, y coordinates of the origin (0, 0) in the occupancy grid
                                    * resolution:     grid resolution in meters
    :param trajectory_cells:    list with cell indexes determined as occ. by the trajectory [<row_ids>, <column_ids>]
    :param trajectory_coords:   numpy array with the original trajectory coordinates (plotted as an overlay)

    :Authors:
        * Maroua Ben Lakhal
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        14.11.2019

    """

    # flip colors of occupancy map
    occ_map['grid'] = 1 - occ_map['grid']

    # initialize grid map as gray scale (black / white)
    cmap = pyplot.cm.gray
    norm = pyplot.Normalize(0, 1)
    rgba = cmap(norm(occ_map['grid']))

    # set the trajectory cells red
    rgba[trajectory_cells[0], trajectory_cells[1], :3] = 1, 0.5, 0.5

    # set the cells with identified trajectory cell and wall yellow
    for row, column in zip(trajectory_cells[0], trajectory_cells[1]):
        if occ_map['grid'][row, column] == 0:
            rgba[row, column, :3] = 1, 1, 0

    # plot grid-map with occupied cells by trajectory
    pyplot.imshow(rgba, interpolation='nearest', extent=[0, occ_map['grid'].shape[1] * occ_map['resolution'],
                                                         0, occ_map['grid'][0] * occ_map['resolution']])

    # plot the original trajectory as an overlay
    pyplot.plot(trajectory_coords[:, 0] - occ_map['origin'][0],
                trajectory_coords[:, 1] - (occ_map['origin'][1] - occ_map['grid'].shape[0] * occ_map['resolution']),
                'r-')

    pyplot.show()
