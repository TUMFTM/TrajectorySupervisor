import numpy as np
import shapely.geometry
import shapely.errors
import math
import trajectory_supervisor

# default step size to generate points along the turing trajectories
STEP_SIZE_DEFAULT = 3.0

# default number of steps to be generated along the turning trajectory
NUM_STEPS_DEFAULT = 15


class ReachSetTurnLimit(object):
    """
    Class that supports the calculation of a simple reachable set, that is limited by the a turn radius curve to the
    left and right.
    """

    def __init__(self,
                 bound_l: np.ndarray,
                 bound_r: np.ndarray,
                 localgg: np.ndarray,
                 closed: bool,
                 trim_set_to_bounds: bool = False) -> None:
        """

        :param bound_l:             coordinates of the left bound (numpy array with columns x, y)
        :param bound_r:             coordinates of the right bound (numpy array with columns x, y)
        :param localgg:             maximum g-g-values at a certain position on the map (currently, only the max a used)
        :param trim_set_to_bounds:  if set to 'True', regions of the reachable set outside of the track are removed
                                    (computationally demanding)

        """

        self.__bound_l = bound_l
        self.__bound_r = bound_r

        self.__localgg = localgg

        self.__closed = closed

        self.__trim_set_to_bounds = trim_set_to_bounds

        # calculate patch of track (in order to trim to bounds by calculating intersection with reachable set)
        if closed:
            polygon_l = shapely.geometry.Polygon(bound_l)
            polygon_r = shapely.geometry.Polygon(bound_r)

            if polygon_l.area > polygon_r.area:
                self.__intersection_patch = polygon_l.difference(polygon_r)
            else:
                self.__intersection_patch = polygon_r.difference(polygon_l)
        else:
            self.__intersection_patch = shapely.geometry.Polygon(np.row_stack((bound_l, np.flipud(bound_r))))

    def calc_reach_set(self,
                       obj_pos: np.ndarray,
                       obj_heading: float,
                       obj_vel: float,
                       obj_length: float,
                       obj_width: float,
                       obj_turn: float,
                       dt: float,
                       t_max: float) -> tuple:
        """
        Calculates a simple reachable set, that is limited by the a turn radius curve to the left and right

        :param obj_pos:         position of the vehicle (x and y coordinate)
        :param obj_heading:     heading of the vehicle
        :param obj_vel:         velocity of the vehicle
        :param obj_length:      length of the vehicle [in m]
        :param obj_width:       width of the vehicle [in m]
        :param obj_turn:        turn radius of the vehicle [in m]
        :param dt:              desired temporal resolution of the reachable set
        :param t_max:           maximum temporal horizon for the reachable set
        :returns:
            * **poly** -        dict of reachable areas with:

                                * keys holding the evaluated time-stamps
                                * values holding the outlining coordinates as a np.ndarray with columns [x, y]

            * **outline** -     outline of shape that is removed from the reachable set


        """

        # ------ calculate simple reachable set and reduce size by bound trajectories ----------------------------------
        # get simple reach set
        a_max_dec = max(self.__localgg[:, 3])  # max deceleration
        reach_set = trajectory_supervisor.helper_funcs.src.reachable_set_simple.\
            simple_reachable_set(obj_pos=obj_pos,
                                 obj_heading=obj_heading,
                                 obj_vel=obj_vel,
                                 obj_length=obj_length,
                                 obj_width=obj_width,
                                 dt=dt,
                                 t_max=t_max,
                                 a_max=a_max_dec)

        # calculate turn radius trajectory
        trajectory_l = get_con_turn_trajecory(object_x=obj_pos[0],
                                              object_y=obj_pos[1],
                                              object_vel=obj_vel,
                                              object_theta=obj_heading,
                                              direction=1,
                                              turn_rad=obj_turn,
                                              localgg=self.__localgg,
                                              object_length=obj_length,
                                              object_width=obj_width)
        trajectory_r = get_con_turn_trajecory(object_x=obj_pos[0],
                                              object_y=obj_pos[1],
                                              object_vel=obj_vel,
                                              object_theta=obj_heading,
                                              direction=-1,
                                              turn_rad=obj_turn,
                                              localgg=self.__localgg,
                                              object_length=obj_length,
                                              object_width=obj_width)

        # get polygons from pure turn radius trajectories
        del_outline = np.row_stack((np.flipud(trajectory_l[:, 0:2]),
                                    trajectory_r[:, 0:2],
                                    trajectory_r[-1, 0:2] + 50.0 * np.array((math.cos(obj_heading),
                                                                             math.sin(obj_heading))),
                                    obj_pos.T + 50.0 * np.array((math.cos(obj_heading), math.sin(obj_heading)))
                                    - 50.0 * np.array((math.cos(obj_heading + np.pi / 2),
                                                       math.sin(obj_heading + np.pi / 2))),
                                    obj_pos.T - 50.0 * np.array((math.cos(obj_heading), math.sin(obj_heading)))
                                    - 50.0 * np.array((math.cos(obj_heading + np.pi / 2),
                                                       math.sin(obj_heading + np.pi / 2))),
                                    trajectory_l[-1, 0:2] + 50.0 * np.array((math.cos(obj_heading + np.pi),
                                                                             math.sin(obj_heading + np.pi)))
                                    ))

        del_patch = shapely.geometry.Polygon(del_outline)

        # perform shapely difference for each time-step
        for t_key in reach_set.keys():
            # convert reachable set to shapely polygon
            poly_reach_t = shapely.geometry.Polygon(reach_set[t_key])

            # subtract deletion patch from reachable set polygon
            try:
                red_tmp = poly_reach_t.difference(del_patch)
            except shapely.errors.TopologicalError:
                red_tmp = poly_reach_t

            # if configured to remove patches besides track
            if self.__trim_set_to_bounds:
                red_tmp = red_tmp.intersection(self.__intersection_patch)

            # convert to coordinates
            if red_tmp:
                # if coordinates present (not wiped out completely), extract outline coordinates
                red_tmp = trajectory_supervisor.helper_funcs.src.shapely_conversions.\
                    extract_polygon_outline(shapely_geometry=red_tmp)

                # add outline coordinates to reach set
                if red_tmp is not None:
                    reach_set[t_key] = red_tmp

        return reach_set, del_outline


def get_pure_turn_trajectory(object_x: float,
                             object_y: float,
                             object_theta: float,
                             direction: float,
                             turn_rad: float,
                             object_length: float,
                             object_width: float):
    """
    Calculate a trajectory sticking to the specified turing radius (no lateral forces assumed).

    :param object_x:        x-position of the vehicle
    :param object_y:        y-position of the vehicle
    :param object_theta:    heading of the vehicle
    :param direction:       1:left -1: right
    :param turn_rad:        turning radius of vehicle
    :param object_length:   length of the vehicle
    :param object_width :   width of the vehicle
    :returns:
        * **trajectory** -  pure turn radius trajectory as a np.ndarray with columns [x, y]
    """

    # rotation matrix
    rot_mat = rotation_matrix(angle=object_theta)

    # sample angular values ranging from 0 to pi/2
    ang_arr = np.linspace(0.0, np.pi / 2, 10)

    # calculate coordinates along turning radius translated to outer edge of vehicle shape
    # object: front left corner and arc moving to the left
    if direction == 1:
        pos_ref = np.dot(rot_mat, np.array((-0.5 * object_width,
                                            0.5 * object_length))).T + np.column_stack((object_x, object_y))

        trajectory = np.dot(rot_mat, np.array((turn_rad * np.cos(ang_arr) - turn_rad,
                                               turn_rad * np.sin(ang_arr)))).T + pos_ref

    # object: front right corner and arc moving to the right
    else:
        pos_ref = np.dot(rot_mat, np.array((0.5 * object_width,
                                            0.5 * object_length))).T + np.column_stack((object_x, object_y))

        trajectory = np.dot(rot_mat, np.array((turn_rad * np.cos(np.pi - ang_arr) + turn_rad,
                                               turn_rad * np.sin(np.pi - ang_arr)))).T + pos_ref

    return trajectory


def get_con_turn_trajecory(object_x: float,
                           object_y: float,
                           object_theta: float,
                           object_vel: float,
                           direction: float,
                           localgg: np.ndarray,
                           turn_rad: float,
                           object_length: float,
                           object_width: float,
                           step_size: float = STEP_SIZE_DEFAULT):
    """
    Calculates a conservative steering trajectory by using maximum longitudinal and lateral acceleration (decoupled),
    until the turn-radius is the limiting factor.

    :param object_x:        x-position of the vehicle
    :param object_y:        y-position of the vehicle
    :param object_theta:    heading of the vehicle
    :param object_vel:      absolute velocity of the vehicle
    :param direction:       1:left -1: right
    :param localgg:         maximum g-g-values at a certain position on the map
    :param turn_rad:        turning radius of vehicle
    :param object_length:   length of the vehicle
    :param object_width :   width of the vehicle
    :param step_size:       size of steps along the planned trajectory
    :returns:
        * **trajectory** -  pure steering trajectory as a np.ndarray with columns [x, y, heading]
    """

    # max lateral acceleration
    a_max_lat = max(localgg[:, 4])
    a_max_lon = max(localgg[:, 3])

    trajectory = np.zeros((NUM_STEPS_DEFAULT, 3))

    # calculate first point in trajectory
    rot_mat = rotation_matrix(angle=object_theta)
    # object: front left corner and arc moving to the left
    if direction == 1:
        trajectory[0, 0:2] = np.dot(rot_mat, np.array((-0.5 * object_width,
                                                       0.5 * object_length))).T + np.column_stack((object_x, object_y))

    # object: front right corner and arc moving to the right
    else:
        trajectory[0, 0:2] = np.dot(rot_mat, np.array((0.5 * object_width,
                                                       0.5 * object_length))).T + np.column_stack((object_x, object_y))
    trajectory[0, 2] = object_theta

    for i in range(1, trajectory.shape[0]):
        # CALC RADIUS BASED ON CURRENT VEL
        r = max(object_vel ** 2 / a_max_lat, turn_rad)

        # CALC NEW POS BASED ON RADIUS
        # Assumption:  A is at (0,0) and B is at (AB, 0), i.e. in the x-direction. Point C is in the pos. y-direction.
        # Source:      https://math.stackexchange.com/a/1989113
        pos_nxt_x = -direction * (step_size ** 2) / (2 * r)
        pos_nxt_y = math.sqrt(step_size ** 2 - pos_nxt_x ** 2)

        # rotate and translate
        rot_mat = rotation_matrix(angle=trajectory[i - 1, 2])
        trajectory[i, 0:2] = np.dot(rot_mat, np.array((pos_nxt_x, pos_nxt_y))) + trajectory[i - 1, 0:2]

        # calculate new heading
        trajectory[i, 2] = math.atan2(trajectory[i, 1] - trajectory[i - 1, 1],
                                      trajectory[i, 0] - trajectory[i - 1, 0]) - np.pi / 2

        # CALC NEW VEL
        object_vel = math.sqrt(max(object_vel ** 2 - 2 * a_max_lon * step_size, 0.0))

    return trajectory


def rotation_matrix(angle: float):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s), (s, c)))
