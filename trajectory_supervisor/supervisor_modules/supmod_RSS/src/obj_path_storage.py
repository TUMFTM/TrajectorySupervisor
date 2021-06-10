from trajectory_supervisor.helper_funcs.src.lane_based_coordinate_system import distance
import numpy as np


def obj_path_storage(obj_path: np.ndarray,
                     obj_data: dict,
                     desired_dis: float,
                     desired_num: int) -> np.ndarray:
    """
    This function inputs the data of the object vehicle and withdraws the position out of the data and then save the
    point every '`desired_dis`' meter.

    :param obj_path:        path of the object vehicle
    :param obj_data:        data of the object vehicle
    :param desired_dis:     minimum required distance between points
    :param desired_num:     number of points which desired to be in the path(in order to save CPU)
    :returns:
        * **obj_path** -    the updated path

    :Authors:
        * Yujie Lian
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        06.07.2019

    """

    # withdraw the position from the data dictionary of object vehicle
    obj_pos = np.array([[obj_data["X"], obj_data["Y"]]])

    if obj_path is None:

        # if this is the first position then just assign the point to the path
        obj_path = obj_pos

    elif obj_path.shape[0] <= desired_num:

        # if the number of points in the path is smaller or equal the desired number then add the points to the path
        if distance(obj_path[-1], obj_pos[0]) >= desired_dis:

            # only if the distance between the points is larger than the desired_dis then add the points
            obj_path = np.append(obj_path, obj_pos, axis=0)

    if obj_path.shape[0] >= (desired_num + 1):

        # if the number of points is more than the desired_num then only take the last few points
        obj_path = obj_path[(-1 * (desired_num + 1)):-1]

    return obj_path
