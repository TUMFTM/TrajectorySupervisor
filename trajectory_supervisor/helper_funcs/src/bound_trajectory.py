import numpy as np


def get_bound_trajectory(object_x: float,
                         object_y: float,
                         object_theta: float,
                         object_vel: float,
                         direction: int,
                         localgg: np.ndarray,
                         long_step: float,
                         bound: np.ndarray,
                         index_track: list,
                         object_length: float,
                         object_width: float,
                         tangential_t: float,
                         t_div: float,
                         delta_t: float,
                         object_turn_rad: float,
                         closed: bool) -> tuple:
    """
    Calculate an outer trajectory for a reachable set, that tries to incorporate kinematics for a proper response.

    For example, the simple reachable set might end non-tangential to the track bounds which would not be executed on
    purpose by any vehicle --> more space is claimed by the reachable set than necessary. The bound trajectory tries
    to identify the outermost trajectory of a reachable set, that is still drivable, i.e. ends tangential to the bounds.

    NOTE: BETA and does not cope closed tracks yet! (e.g. cut_out_bound (selection of relevant bound segment; returned
                                                     "first_index" of this function -> further usage of it; ...))

    :param object_x:            x-position of the vehicle [in m]
    :param object_y:            y-position of the vehicle [in m]
    :param object_theta:        heading of the vehicle [in rad]
    :param object_vel:          x-velocity of the vehicle [in m/s]
    :param direction:           1:left -1: right
    :param localgg:             maximum g-g-values at a certain position on the map (columns: x, y, s, ax, ay)
    :param long_step:           longitudinal step width boundary of occupancy [in m]
    :param bound:               bound of the race track which SHOULD match the choice of direction
    :param index_track:         index of first & last index of track boundary relevant for calculation
    :param object_length:       length of the vehicle [in m]
    :param object_width :       width of the vehicle [in m]
    :param tangential_t:        approximation of the switching time [in s]
    :param t_div:               max time step needed for trajectory calculation [in s]
    :param delta_t:             time step of double arc [in s]
    :param object_turn_rad:     turning radius of vehicle [in m]
    :param closed:              boolean flag indicating a closed track
    :returns:
        * **trajectory** -      trajectory indicating new bound of reach set (either transition to bound or pure
                                steering) - np.array with columns [x, y, t, heading]
        * **bound_reachset** -  boundary of reachable set as np.array with columns [x y t heading]
        * **index_track** -     first & last index of cut out (relevant) track boundary
        * **tangential_t** -    new approximation of the switching time, remains same if pure steering [in s]

    :Authors:
        * Nils Rack
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        09.07.2020

    """

    # get the intersection point of bound and regular reachable set
    bound_reachset, __, inter_found, inter_point, index_track = \
        get_bound_reach_intersect(object_x=object_x,
                                  object_y=object_y,
                                  object_theta=object_theta,
                                  object_vel=object_vel,
                                  direction=direction,
                                  localgg=localgg,
                                  long_step=long_step,
                                  bound=bound,
                                  index_track=index_track,
                                  object_length=object_length,
                                  object_width=object_width,
                                  closed=closed)

    # save index of cut out track boundary + tolerance for next iteration
    # first index
    if index_track[0] > 1:
        if index_track[0] < bound.shape[0] - 1:
            index_track[0] -= 2
        # reset if end reached
        else:
            index_track[0] = 0
    else:
        index_track[0] = 0

    # last index
    if index_track[1] > bound.shape[0] - 5:
        index_track[1] = bound.shape[0]
    else:
        index_track[1] += 4

    # -- calculate longitudinal distance to intersection -----------
    rot_mat = np.array([[np.cos(-object_theta), -np.sin(-object_theta)],
                        [np.sin(-object_theta), np.cos(-object_theta)]])

    if inter_found == 1:
        long_dis = (rot_mat[1, 0] * (inter_point[0] - object_x) + rot_mat[1, 1] * (inter_point[1] - object_y))
    else:
        long_dis = 0.0

    # -- calculate outer trajectories ------------------------------
    if inter_found == 1:
        trajectory, tangential_t = \
            get_edge_trajectory(object_x=object_x,
                                object_y=object_y,
                                object_theta=object_theta,
                                object_vel=object_vel,
                                localgg=localgg,
                                direction=direction,
                                t_approximation=tangential_t,
                                delta_t=delta_t,
                                max_t=t_div,
                                bound=bound[index_track[0]:index_track[1] + 1, :],
                                lower_limit_long=long_dis,
                                object_length=object_length,
                                object_width=object_width,
                                closed=closed)

    else:
        trajectory = get_pure_steering(object_x=object_x,
                                       object_y=object_y,
                                       object_theta=object_theta,
                                       object_vel=object_vel,
                                       direction=direction,
                                       t_end=t_div * 4,
                                       delta_t=delta_t,
                                       localgg=localgg,
                                       turn_rad=object_turn_rad,
                                       object_length=object_length,
                                       object_width=object_width)

    return trajectory, bound_reachset, index_track, tangential_t


def get_edge_trajectory(object_x: float,
                        object_y: float,
                        object_theta: float,
                        object_vel: float,
                        localgg: np.ndarray,
                        direction: float,
                        t_approximation: float,
                        delta_t: float,
                        max_t: float,
                        bound: np.ndarray,
                        lower_limit_long: float,
                        object_length: float,
                        object_width: float,
                        closed: bool) -> tuple:
    """
    Calculates the edge trajectory of a pure steering maneuver + connected extremal trajectory
       => a_lat(t<t_s)=a_max
       => a_lat(t>t_s)=-a_max

    :param object_x:            x-position of the vehicle
    :param object_y:            y-position of the vehicle
    :param object_theta:        heading of the vehicle
    :param object_vel:          x-velocity of the vehicle
    :param localgg:             maximum g-g-values at a certain position on the map (columns: x, y, s, ax, ay)
    :param direction:           1:left -1: right
    :param delta_t:             time step of double arc
    :param t_approximation:     approximation of the switching time
    :param max_t:               max time step needed for trajectory calculation
    :param bound:               bound of the race track which SHOULD match the choice of direction
    :param lower_limit_long:    lower longitudinal limit of cutting window
    :param object_length:       length of the vehicle
    :param object_width :       width of the vehicle
    :param closed:              boolean flag indicating a closed track

    :returns:
        * **edge_trajectory** - points on the arc trajectory with equally divided distance between each other.
                                np. array with [x, y, t, heading] columns
        * **t** -               new approximation of the switching time [in s]

    """

    # tuning parameter
    tolerance_inter = 0.25
    tolerance_slope = 0.99
    if direction == 1:
        cutting_window_x = np.array([30, 10])
    else:
        cutting_window_x = np.array([10, 30])
    cutting_window_y = np.array([lower_limit_long - 4, 80])
    long_step = 4  # longitudinal step width boundary of occupancy

    # previous t_switch saved
    if t_approximation == 0:
        n_iter = 5
        t_air = 0
        t_inter = 1.5
    # no previous t_switch known
    else:
        n_iter = 10
        t_air = t_approximation - 0.1
        t_inter = t_approximation + 0.1

    # max lateral acceleration and max deceleration
    a_max_long = max(localgg[:, 3])
    a_max_lat = max(localgg[:, 4])

    # get the cut out race track bound
    track_bound, __ = cut_out_bound(bound=bound,
                                    ref_x=object_x,
                                    ref_y=object_y,
                                    ref_theta=object_theta,
                                    delta_x=cutting_window_x,
                                    delta_y=cutting_window_y,
                                    closed=closed)

    # -- calculate distance to cg --
    # object: front left corner
    if direction == 1:
        delta_x = - np.sin(-object_theta) * 0.5 * object_length - np.cos(-object_theta) * 0.5 * object_width
        delta_y = - np.cos(-object_theta) * 0.5 * object_length + np.sin(-object_theta) * 0.5 * object_width

    # object: front right corner
    else:
        delta_x = - np.sin(-object_theta) * 0.5 * object_length + np.cos(-object_theta) * 0.5 * object_width
        delta_y = - np.cos(-object_theta) * 0.5 * object_length - np.sin(-object_theta) * 0.5 * object_width

    bound_occ = get_bound_reachable_set(object_x=delta_x,
                                        object_y=delta_y,
                                        object_theta=object_theta,
                                        object_vel=object_vel,
                                        direction=-direction,
                                        localgg=localgg,
                                        long_step=long_step,
                                        object_length=object_length,
                                        object_width=object_width)

    slope = False
    n = 0

    while True:
        n += 1
        t = (t_inter + t_air) / 2

        arc = get_double_arc(object_x=object_x,
                             object_y=object_y,
                             object_theta=object_theta,
                             object_vel=object_vel,
                             direction=direction,
                             t_switch=t,
                             delta_t=delta_t,
                             localgg=localgg,
                             object_length=object_length,
                             object_width=object_width)

        # attach boundary of occupancy
        temp = bound_occ[:, :2] + arc[-1, :2]
        bound = np.concatenate((temp, bound_occ[:, 2:]), axis=1)

        auxiliary_trajectory = np.vstack((arc[:, :], bound[1:, :]))

        # check for intersection - sweep line
        inter, inter_point, index = sweep_line_intersection(set1=auxiliary_trajectory[:, :2],
                                                            set2=track_bound,
                                                            theta=object_theta,
                                                            tolerance=tolerance_inter)

        # if switching time at start, calculate "air" trajectory
        if t < 0.0:
            break

        if inter:
            # check if tangential
            slope = check_slope(s1=auxiliary_trajectory[index[0]:index[0] + 2, :2],
                                s2=track_bound[index[1]:index[1] + 2, :],
                                tolerance=tolerance_slope)
            t_inter = t

        else:
            t_air = t

        # if conditions fulfilled or maximum number of iterations reached
        if slope or n >= n_iter:
            break

    # no tangential trajectory found => use closest "air" trajectory
    if not slope:

        t = t_air

        arc = get_double_arc(object_x=object_x,
                             object_y=object_y,
                             object_theta=object_theta,
                             object_vel=object_vel,
                             direction=direction,
                             t_switch=t,
                             delta_t=delta_t,
                             localgg=localgg,
                             object_length=object_length,
                             object_width=object_width)

        # calculate height of extremal trajectory as the height of the bound of occupancy
        tau = 2 * np.sqrt(2 / 3) * object_vel / a_max_long * np.cos(4 * np.pi / 3 + 1 / 3 * np.arccos(-1))

        y_tilde = np.sqrt(2 / 3) * 2 / 3 * object_vel ** 2 / a_max_long

        h = np.sqrt((0.5 * a_max_lat * tau ** 2) ** 2
                    - (a_max_lat / a_max_long) ** 2 * (y_tilde - object_vel * tau) ** 2)

        # accounting rounding error
        h = h - 0.001

        # extremal trajectory
        ext_trajectory = get_ext_trajectory(x=arc[-1, 0],
                                            y=arc[-1, 1],
                                            theta=object_theta,
                                            vel=object_vel,
                                            direction=-direction,
                                            h=h,
                                            localgg=localgg,
                                            delta_t=delta_t)

        # add time of first part of trajectory
        ext_trajectory[:, 2] = ext_trajectory[:, 2] + arc[-1, 2]

        edge_trajectory = np.vstack((arc[:, :], ext_trajectory[1:, :]))

    else:
        # intersection with arc
        if index[0] + 1 < arc.shape[0]:

            # trajectory to short => extend with track boundary
            if arc[index[0], 2] < max_t:

                track_bound_trajectory = get_track_bound_traj(bound=track_bound[index[1]:, :],
                                                              end_point_traj=arc[index[0], :],
                                                              t_end=max_t,
                                                              delta_t=delta_t,
                                                              object_vel=object_vel)

                edge_trajectory = np.vstack((arc[:index[0] + 1, :], track_bound_trajectory[:, :]))

            # trajectory long enough
            else:

                edge_trajectory = arc[:index[0] + 1, :]

        # intersection with bound
        else:

            # rotation matrix x_tilde = rot*x
            rot_mat = np.array([[np.cos(-object_theta), -np.sin(-object_theta)],
                                [np.sin(-object_theta), np.cos(-object_theta)]])

            # calculate height of extremal trajectory at point of intersection
            h = direction * (rot_mat[0, 0] * (inter_point[0] - arc[-1, 0])
                             + rot_mat[0, 1] * (inter_point[1] - arc[-1, 1]))

            # accounting rounding error
            if h < 0:
                h = 0.1

            ext_trajectory = get_ext_trajectory(x=arc[-1, 0],
                                                y=arc[-1, 1],
                                                theta=object_theta,
                                                vel=object_vel,
                                                direction=-direction,
                                                h=h,
                                                localgg=localgg,
                                                delta_t=delta_t)

            # add time of first part of trajectory
            ext_trajectory[:, 2] = ext_trajectory[:, 2] + arc[-1, 2]

            # trajectory to short => extend with track boundary (last point of ext_trajectory not n * delta_t => -2)
            if ext_trajectory[-1, 2] < max_t:
                track_bound_trajectory = get_track_bound_traj(bound=track_bound[index[1]:, :],
                                                              end_point_traj=ext_trajectory[-1, :],
                                                              t_end=max_t,
                                                              delta_t=delta_t,
                                                              object_vel=object_vel)

                edge_trajectory = np.vstack((arc[:, :], ext_trajectory[1:, :], track_bound_trajectory[:, :]))
            else:
                edge_trajectory = np.vstack((arc[:, :], ext_trajectory[1:, :]))

    return edge_trajectory, t


def get_bound_reach_intersect(object_x: float,
                              object_y: float,
                              object_theta: float,
                              object_vel: float,
                              direction: float,
                              localgg: np.ndarray,
                              long_step: float,
                              bound: np.ndarray,
                              index_track: list,
                              object_length: float,
                              object_width: float,
                              closed: bool) -> tuple:
    """
    Calculates intersection between track bound and bound of reachable set.

    :param object_x:            x-position of the vehicle
    :param object_y:            y-position of the vehicle
    :param object_theta:        heading of the vehicle
    :param object_vel:          absolute velocity of the vehicle
    :param direction:           1:left -1: right
    :param localgg:             maximum g-g-values at a certain position on the map (columns: x, y, s, ax, ay)
    :param long_step:           longitudinal step width boundary of occupancy
    :param bound:               boundary of race track according to choice of direction
    :param index_track:         first & last index of track boundary relevant for calculation
    :param object_length:       length of the vehicle
    :param object_width :       width of the vehicle
    :param closed:              boolean flag indicating whether the considered track is closed or not

    :returns:
        * **bound_reachset** -  boundary of reachable set as np.array with columns [x y t heading]
        * **index[0]** -        index of boundary segment
        * **inter** -           boolean intersection found
        * **inter_point** -     point of intersection as a np.array with column [x y]
        * **index_track** -     first & last index of cut out track boundary

    """

    bound_reachset = get_bound_reachable_set(object_x=object_x,
                                             object_y=object_y,
                                             object_theta=object_theta,
                                             object_vel=object_vel,
                                             direction=direction,
                                             localgg=localgg,
                                             long_step=long_step,
                                             object_length=object_length,
                                             object_width=object_width)

    if direction == 1:
        delta_x = np.array([30, 10])
    else:
        delta_x = np.array([10, 30])

    # extract relevant segment of track bound
    track_bound, first_index = cut_out_bound(bound=bound[index_track[0]:index_track[1] + 1, :],
                                             ref_x=object_x,
                                             ref_y=object_y,
                                             ref_theta=object_theta,
                                             delta_x=delta_x,
                                             delta_y=np.array([0, 60]),
                                             closed=closed)

    # first & last index of extracted track bound
    index_track = [index_track[0] + first_index, index_track[0] + first_index + track_bound.shape[0] - 1]

    # # check fo intersection - brute force
    # inter, inter_point = brute_force_intersection(set1=bound_reach,
    #                                               set2=track_bound)

    # check for intersection - sweep line
    inter_found, inter_point, index = sweep_line_intersection(set1=bound_reachset[:, :2],
                                                              set2=track_bound,
                                                              theta=object_theta,
                                                              tolerance=0)

    return bound_reachset, index[0], inter_found, inter_point, index_track


def get_bound_reachable_set(object_x: float,
                            object_y: float,
                            object_theta: float,
                            object_vel: float,
                            direction: float,
                            localgg: np.ndarray,
                            long_step: float,
                            object_length: float,
                            object_width: float) -> np.ndarray:
    """
    Calculates front most boundary of reachable set, left side => direction = 1
                                                     right side => direction = -1
    The boundary calculation uses a friction ellipse as its car model.
    As the max lateral and longitudinal acceleration the max g-values of the tires are used.

    :param object_x:        x-position of the vehicle
    :param object_y:        y-position of the vehicle
    :param object_theta:    heading of the vehicle
    :param object_vel:      absolute velocity of the vehicle
    :param direction:       1:left -1: right
    :param localgg:         maximum g-g-values at a certain position on the map (columns: x, y, s, ax, ay)
    :param long_step:       longitudinal step width
    :param object_length:   length of the vehicle
    :param object_width :   width of the vehicle

    :returns:
        * **bound** -       boundary of reachable set defined via connected segments in a np.ndarray
                            with columns [x, y, tau, heading]

    """

    # max lateral displacement - tuning parameter
    max_lat_dis = 25.0

    # max acceleration
    a_max_long = max(localgg[:, 3])
    a_max_lat = max(localgg[:, 4])

    # maximum long. displacement - end of domain
    max_long_dis = 2 / 3 * np.sqrt(2 / 3) * object_vel ** 2 / a_max_long

    # step too short
    if 2 * long_step > max_long_dis:
        long_step = max_long_dis / 3

    # boundary of reachable set
    n = int(np.ceil(max_long_dis / long_step))
    bound = np.zeros([n, 4])

    # inverse rotation matrix x=rot*x_tilde
    rot_mat_in = np.array([[np.cos(-object_theta), np.sin(-object_theta)],
                           [-np.sin(-object_theta), np.cos(-object_theta)]])

    # object: front left corner
    if direction == 1:
        x_f = object_x + np.sin(-object_theta) * 0.5 * object_length - np.cos(-object_theta) * 0.5 * object_width
        y_f = object_y + np.cos(-object_theta) * 0.5 * object_length + np.sin(-object_theta) * 0.5 * object_width
    # object: front right corner
    else:
        x_f = object_x + np.sin(-object_theta) * 0.5 * object_length + np.cos(-object_theta) * 0.5 * object_width
        y_f = object_y + np.cos(-object_theta) * 0.5 * object_length - np.sin(-object_theta) * 0.5 * object_width

    # auxiliary variables to save prior position
    temp_x = 0
    temp_y = 0

    # -- calculate segment coordinates ---------------------------------------------------------------------------------
    for i in range(0, n):

        # y-coordinate of boundary in object coordinates
        y_tilde = i * long_step
        if i == 0:
            y_tilde = 0.01

        # check if long. distance reached end of max long distance => break loop
        if y_tilde > max_long_dis:
            bound = bound[0:i, :]
            break

        # tau
        tau = 2 * np.sqrt(2 / 3) * object_vel / a_max_long * np.cos(4 * np.pi / 3 + 1 / 3 * np.arccos(
            - np.sqrt(3 / 2) * 3 / 2 * a_max_long * y_tilde / object_vel ** 2))

        # x-coordinate of boundary in object coordinates
        x_tilde = - direction * np.sqrt((0.5 * a_max_lat * tau ** 2) ** 2
                                        - (a_max_lat / a_max_long) ** 2 * (y_tilde - object_vel * tau) ** 2)

        # lateral displacement reached maximum => break loop
        if x_tilde > max_lat_dis or x_tilde < -max_lat_dis:
            bound = bound[0:i, :]
            break

        # calculate heading of trajectory using finite difference
        if i == 0:
            heading = 0
        else:
            heading = np.arctan((x_tilde - temp_x) / (y_tilde - temp_y))
            temp_x = x_tilde
            temp_y = y_tilde

        # transform coordinates into global system an save in bound
        if i == 0:
            bound[i, 0] = x_f
            bound[i, 1] = y_f
        else:
            bound[i, 0] = x_f + rot_mat_in[0, 0] * x_tilde + rot_mat_in[0, 1] * y_tilde
            bound[i, 1] = y_f + rot_mat_in[1, 0] * x_tilde + rot_mat_in[1, 1] * y_tilde

        bound[i, 2] = tau
        bound[i, 3] = object_theta - heading

        # temporarily displayed ----------------------------------------------------------------------------------------
        # Boundary as a function of t - domain smaller than above
        # x_tilde = - direction * np.sqrt( 0.25 * a_max**2 * t**4 - 0.25 * a_max**4 * t**6 / object_vel**2 )
        # y_tilde = object_vel * t - 0.5 * a_max**2 * t**3 / object_vel
        # --------------------------------------------------------------------------------------------------------------

    return bound


def cut_out_bound(bound: np.ndarray,
                  ref_x: float,
                  ref_y: float,
                  ref_theta: float,
                  delta_x: np.ndarray,
                  delta_y: np.ndarray,
                  closed: bool) -> tuple:
    """
    Extract a segment of a path (e.g track bound), which is within a window of
        [object_x - delta_x[0], object_x + delta_x[1]],
        [ref_y + delta_y[0], ref_y + delta_y[1]]

    :param bound:           bound, where segment should be extracted
    :param ref_x:           x-position of the reference pose
    :param ref_y:           y-position of the reference pose
    :param ref_theta:       heading of the reference pose
    :param delta_x:         width of window => delta_x[0] distance to left of ref pos (in direction of heading)
                                               delta_x[1] distance to right of ref pos (in direction of heading)
    :param delta_y:         length of window => delta_y[0] lower bound starting at cg
                                                delta_y[1] upper bound starting at cg
    :param closed:          flag indicating whether provided bound is closed or not

    :returns:
        * **extr_path** -   extracted path segment
        * **first_index** - first index of path which is cut extracted

    """

    # rotation matrix x_tilde = rot*x
    rot_mat = np.array([[np.cos(-ref_theta), -np.sin(-ref_theta)],
                        [np.sin(-ref_theta), np.cos(-ref_theta)]])

    # Rotate object coordinates into object coordinate system
    ref_x_rot = rot_mat[0, 0] * ref_x + rot_mat[0, 1] * ref_y
    ref_y_rot = rot_mat[1, 0] * ref_x + rot_mat[1, 1] * ref_y

    # cut bound
    extr_path = np.zeros([bound.shape[0], 2])
    k = 0

    # to prevent interrupted segments
    stop = 0

    # index of last element being not in the cut window
    first_index = -1

    # NOTE: add overlapping bound here, if closed
    for (x, y) in zip(bound[:, 0], bound[:, 1]):

        # Rotate bound coordinates into object coordinate system
        x_tilde = rot_mat[0, 0] * x + rot_mat[0, 1] * y
        y_tilde = rot_mat[1, 0] * x + rot_mat[1, 1] * y

        # Check if coordinate within window
        if ((ref_x_rot - delta_x[0]) <= x_tilde <= (ref_x_rot + delta_x[1])) and \
                ((ref_y_rot + delta_y[0]) <= y_tilde <= (ref_y_rot + delta_y[1])):
            extr_path[k + 1, 0] = x
            extr_path[k + 1, 1] = y

            # NOTE: temporary fix, overlapping bounds (closed track) must be handled differently
            if k + 2 >= bound.shape[0]:
                break

            k = k + 1

            stop = 1
        else:
            if stop == 1:
                break
            first_index += 1

    # add last element before window
    if first_index >= 0:
        extr_path[0, 0] = bound[first_index, 0]
        extr_path[0, 1] = bound[first_index, 1]
    else:
        extr_path = extr_path[1:, :]
        k = k - 1

    return extr_path[:k + 1, :], first_index


def sweep_line_intersection(set1: np.ndarray,
                            set2: np.ndarray,
                            theta: float,
                            tolerance: float) -> tuple:
    """
    Calculates intersection between two sets of connected segments using a sweep line algorithm.
    The algorithm is a adapted version of the set-based intersection algorithm in:
    https://www.sciencedirect.com/science/article/pii/S0098300499000710
    The sweep line moves in the direction of the object
    CAUTION: The algorithm can only be used for sets where der y-coordinate in the rotated coordinate
    system is strictly increasing with rising indices

    :param set1:                    first set of coodinates (columns x, y)
    :param set2:                    second set of coodinates (columns x, y)
    :param theta:                   heading of vehicle
    :param tolerance                tolerance of segments in set 1 in orthogonal direction of the object
                                    >0 tolerance to the right
                                    <0 tolerance to the left
    :returns:
        * **inter_found** -         true when intersection found
        * **intersection_point** -  intersection coordinates as a np.array with column [x, y], Default [0,0]
        * **index** -               first indexes of intersecting segments np.array [n_1, n_2]

    """

    # check set size
    if set1.shape[0] < 2 or set2.shape[0] < 2:
        return 0, np.array([0, 0]), np.array([0, 0])

    # rotation matrix x_tilde = rot*x
    rot_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
                        [np.sin(-theta), np.cos(-theta)]])

    # rotate sets
    set1_rot = np.zeros([set1.shape[0], 2])
    set2_rot = np.zeros([set2.shape[0], 2])

    for i in range(0, set1.shape[0]):
        set1_rot[i, :] = np.array([rot_mat[0, 0] * set1[i, 0] + rot_mat[0, 1] * set1[i, 1],
                                   rot_mat[1, 0] * set1[i, 0] + rot_mat[1, 1] * set1[i, 1]])

    for j in range(0, set2.shape[0]):
        set2_rot[j, :] = np.array([rot_mat[0, 0] * set2[j, 0] + rot_mat[0, 1] * set2[j, 1],
                                   rot_mat[1, 0] * set2[j, 0] + rot_mat[1, 1] * set2[j, 1]])

    # sweep line Status: initialization
    n_1 = 0
    n_2 = 0
    edge1 = set1_rot[n_1:(n_1 + 2), :]
    edge2 = set2_rot[n_2:(n_2 + 2), :]

    # intersection point and status
    intersection_point = np.array([0, 0])
    inter_found = 0

    while not inter_found:

        # Update Sweep line edges
        # next vertex neighbour of edge 1
        if edge1[1, 1] < edge2[1, 1] and n_1 < set1_rot.shape[0] - 2:
            n_1 = n_1 + 1
            edge1 = set1_rot[n_1:(n_1 + 2), :]
        # next vertex neighbour of edge 2
        elif edge2[1, 1] < edge1[1, 1] and n_2 < set2_rot.shape[0] - 2:
            n_2 = n_2 + 1
            edge2 = set2_rot[n_2:(n_2 + 2), :]
        # end of set is reached
        else:
            return 0, np.array([0, 0]), np.array([0, 0])

        # check for intersection
        if tolerance == 0:
            inter_found, intersection_point = check_intersect(edge1[0, :], edge1[1, :], edge2[0, :], edge2[1, :])
        # to include tolerances: replace edge1 with the two diagonals of the tolerated area
        else:
            intersect1, intersection_point1 = check_intersect(np.array([edge1[0, 0], edge1[0, 1]]),
                                                              np.array([edge1[1, 0] + tolerance, edge1[1, 1]]),
                                                              edge2[0, :], edge2[1, :])
            intersect2, intersection_point2 = check_intersect(np.array([edge1[0, 0] + tolerance, edge1[0, 1]]),
                                                              np.array([edge1[1, 0], edge1[1, 1]]),
                                                              edge2[0, :], edge2[1, :])

            if intersect1:
                inter_found = intersect1
                intersection_point = intersection_point1
            if intersect2:
                inter_found = intersect2
                intersection_point = intersection_point2

    # rotate back into global system
    intersection_point = np.array([rot_mat[0, 0] * intersection_point[0] - rot_mat[0, 1] * intersection_point[1],
                                   - rot_mat[1, 0] * intersection_point[0] + rot_mat[1, 1] * intersection_point[1]])

    return inter_found, intersection_point, np.array([n_1, n_2])


def get_double_arc(object_x: float,
                   object_y: float,
                   object_theta: float,
                   object_vel: float,
                   direction: float,
                   t_switch: float,
                   delta_t: float,
                   localgg: np.ndarray,
                   object_length: float,
                   object_width: float) -> np.ndarray:
    """
    Calculates trajectory of pure steering maneuver with t_switch at a_lat from +a_max to - a_max

    :param object_x:        x-position of the vehicle
    :param object_y:        y-position of the vehicle
    :param object_theta:    heading of the vehicle
    :param object_vel:      absolute velocity of the vehicle
    :param direction:       1:left -1: right
    :param t_switch:        time of switching from full lat. acceleration to full lat. deceleration
    :param delta_t:         time step
    :param localgg:         maximum g-g-values at a certain position on the map (columns: x, y, s, ax, ay)
    :param object_length:   length of the vehicle
    :param object_width :   width of the vehicle

    :returns:
        * **trajectory** -  double arc trajectory as a np.ndarray with columns [x, y, t, heading]
    """

    # variables
    t_end = 2 * t_switch
    n_seg = max(int(t_end / delta_t) + (t_end % delta_t > 0) + 1, 1)
    t = 0

    # trajectory array
    trajectory = np.zeros([n_seg, 4])

    # max acceleration
    a_max_lat = max(localgg[:, 4])

    # inverse rotation matrix x=rot*x_tilde
    rot_mat_in = np.array([[np.cos(-object_theta), np.sin(-object_theta)],
                           [-np.sin(-object_theta), np.cos(-object_theta)]])

    # object: front left corner
    if direction == 1:
        x_f = object_x + np.sin(-object_theta) * 0.5 * object_length - np.cos(-object_theta) * 0.5 * object_width
        y_f = object_y + np.cos(-object_theta) * 0.5 * object_length + np.sin(-object_theta) * 0.5 * object_width
    # object: front right corner
    else:
        x_f = object_x + np.sin(-object_theta) * 0.5 * object_length + np.cos(-object_theta) * 0.5 * object_width
        y_f = object_y + np.cos(-object_theta) * 0.5 * object_length - np.sin(-object_theta) * 0.5 * object_width

    # auxiliary variables to save prior position
    temp_x = 0
    temp_y = 0

    # calculate segment coordinates
    for i in range(0, n_seg):

        # x-coordinate of trajectory in object coordinates
        if t < t_switch:
            x_tilde = - direction * a_max_lat * t ** 2 / 2
        else:
            x_tilde = - direction * a_max_lat * (-t_switch ** 2 + 2 * t_switch * t - 0.5 * t ** 2)

        # y-coordinate of trajectory in object coordinates
        y_tilde = object_vel * t

        # calculate heading of trajectory using difference quotient
        if i == 0:
            heading = 0
        else:
            # prevent division by zero
            if abs(y_tilde - temp_y) < 0.00001:
                heading = 0.9999 * np.pi
            else:
                heading = np.arctan((x_tilde - temp_x) / (y_tilde - temp_y))
            temp_x = x_tilde
            temp_y = y_tilde

        # transform coordinates into global system an save in bound
        trajectory[i, 0] = x_f + rot_mat_in[0, 0] * x_tilde + rot_mat_in[0, 1] * y_tilde
        trajectory[i, 1] = y_f + rot_mat_in[1, 0] * x_tilde + rot_mat_in[1, 1] * y_tilde
        trajectory[i, 2] = t
        trajectory[i, 3] = object_theta - heading

        # time of iteration
        t = t + delta_t

        # ensure last point at t_end
        if t > t_end:
            t = t_end

    return trajectory


def check_slope(s1: np.ndarray,
                s2: np.ndarray,
                tolerance: float) -> bool:
    """
    Checks if two segment have the same slope within the defined tolerance

    :param s1:             start and end point of segment 1 as a np.array with column [x, y]
    :param s2:             start and end point of segment 2 as a np.array with column [x, y]
    :param tolerance:      tolerance as the minimum value of cosine(angle between c and d)

    :returns:
        * **equal** -      boolean => true when segments have same slope
    """

    # get connection vector
    c = np.array([s1[1, 0] - s1[0, 0], s1[1, 1] - s1[0, 1]])
    d = np.array([s2[1, 0] - s2[0, 0], s2[1, 1] - s2[0, 1]])

    equal = False
    cosine = np.dot(c, d) / (np.linalg.norm(c) * np.linalg.norm(d))

    if cosine > tolerance:
        equal = True

    return equal


def get_ext_trajectory(x: float,
                       y: float,
                       theta: float,
                       vel: float,
                       direction: float,
                       h: float,
                       localgg: np.ndarray,
                       delta_t: float) -> np.ndarray:
    """
    Calculates extremal trajectory (trajectory reaching height h with the shortest long. path)

    :param x:               x-position of the vehicle
    :param y:               y-position of the vehicle
    :param theta:           heading of the vehicle
    :param vel:             absolute velocity of the vehicle
    :param direction:       1:left -1: right
    :param h:               height at intersection
    :param localgg:         maximum g-g-values at a certain position on the map (columns: x, y, s, ax, ay)
    :param delta_t:         time step

    :returns:
        * **trajectory** -  trajectory as a np.ndarray with columns [x, y, t, heading]
    """

    # max acceleration
    a_max_long = max(localgg[:, 3])
    a_max_lat = max(localgg[:, 4])

    # calculate acceleration ratio
    zeta = np.pi + np.arcsin(2 / np.sqrt(3) * np.cos(2 / 3 * np.pi + 1 / 3
                                                     * np.arccos(3 * np.sqrt(3) * a_max_long ** 2 * h
                                                                 / (vel ** 2 * a_max_lat))))

    # time at height h
    t_end = np.sqrt(2 * h / (a_max_lat * np.sin(zeta)))
    n_seg = int(t_end / delta_t) + 1

    # variables
    t = 0
    trajectory = np.zeros([n_seg, 4])

    # inverse rotation matrix x=rot*x_tilde
    rot_mat_in = np.array([[np.cos(-theta), np.sin(-theta)],
                           [-np.sin(-theta), np.cos(-theta)]])

    # auxiliary variables to save prior position
    temp_x = 0
    temp_y = 0

    # calculate segment coordinates
    for i in range(0, n_seg):

        # x-coordinate of trajectory in object coordinates
        x_tilde = - direction * 0.5 * a_max_lat * np.sin(zeta) * t ** 2

        # y-coordinate of trajectory in object coordinates
        y_tilde = vel * t + 0.5 * a_max_long * np.cos(zeta) * t ** 2

        # calculate heading of trajectory using difference quotient
        if i == 0:
            heading = 0
        else:
            heading = np.arctan((x_tilde - temp_x) / (y_tilde - temp_y))
            temp_x = x_tilde
            temp_y = y_tilde

        # transform coordinates into global system an save in bound
        trajectory[i, 0] = x + rot_mat_in[0, 0] * x_tilde + rot_mat_in[0, 1] * y_tilde
        trajectory[i, 1] = y + rot_mat_in[1, 0] * x_tilde + rot_mat_in[1, 1] * y_tilde
        trajectory[i, 2] = t
        trajectory[i, 3] = theta - heading

        # time of iteration
        t = t + delta_t

    return trajectory


def get_track_bound_traj(bound: np.ndarray,
                         end_point_traj: np.ndarray,
                         t_end: float,
                         delta_t: float,
                         object_vel: float) -> np.ndarray:
    """
    Calculates extension of trajectory which follows track boundary with object velocity

    :param bound:           bound of race track starting with segment which intersects
    :param end_point_traj:  last point of trajectory as a np.ndarray with the column [x, y, t, heading]
    :param t_end:           end time of trajectory extension
    :param delta_t:         time step
    :param object_vel:      absolute velocity of the vehicle

    :returns:
        * **extension** -   trajectory extension as a np.ndarray with columns [x, y, t, heading]
    """

    # set t to end time of trajectory
    t = end_point_traj[2]

    # trajectory extension array
    n_seg = int((t_end - t) / delta_t) + 1
    extension = np.zeros([n_seg, 4])

    # stack end point + bound
    path = np.vstack((end_point_traj[0:2], bound[1:, 0:2]))

    i = 0
    k = 0

    time = t

    while t < t_end and i < path.shape[0] - 1:

        # time of next point
        t = t + delta_t

        # find next point on path
        while time < t and i < path.shape[0] - 1:

            # length of segment [i, i+1]
            length = np.sqrt((path[i + 1, 0] - path[i, 0]) ** 2 + (path[i + 1, 1] - path[i, 1]) ** 2)

            # elapsed time
            time = time + length / object_vel

            i = i + 1

        heading = - np.sign(path[i, 0] - path[i - 1, 0]) * np.pi / 2 + np.arctan((path[i, 1] - path[i - 1, 1])
                                                                                 / (path[i, 0] - path[i - 1, 0]))

        extension[k, 0] = path[i - 1, 0] + (((length / object_vel) - (time - t)) / (length / object_vel)
                                            * (path[i, 0] - path[i - 1, 0]))
        extension[k, 1] = path[i - 1, 1] + (((length / object_vel) - (time - t)) / (length / object_vel)
                                            * (path[i, 1] - path[i - 1, 1]))
        extension[k, 2] = t
        extension[k, 3] = heading

        k = k + 1

    return extension[:k, :]


def get_pure_steering(object_x: float,
                      object_y: float,
                      object_theta: float,
                      object_vel: float,
                      direction: float,
                      t_end: float,
                      delta_t: float,
                      localgg: np.ndarray,
                      turn_rad: float,
                      object_length: float,
                      object_width: float) -> np.ndarray:
    """
    Calculates trajectory of pure steering maneuver

    :param object_x:        x-position of the vehicle
    :param object_y:        y-position of the vehicle
    :param object_theta:    heading of the vehicle
    :param object_vel:      absolute velocity of the vehicle
    :param direction:       1:left -1: right
    :param t_end:           time interval of calculated trajectory
    :param delta_t:         time step
    :param localgg:         maximum g-g-values at a certain position on the map (columns: x, y, s, ax, ay)
    :param turn_rad:        turning radius of vehicle
    :param object_length:   length of the vehicle
    :param object_width :   width of the vehicle

    :returns:
        * **trajectory** -  pure steering trajectory as a np.ndarray with columns [x, y, t, heading]
    """

    # Tuning Parameters
    n_seg = int(t_end / delta_t) + 1  # number of segments

    # trajectory array
    trajectory = np.zeros([n_seg, 4])

    # max lateral acceleration
    a_max_lat = max(localgg[:, 4])

    # inverse rotation matrix x=rot*x_tilde
    rot_mat_in = np.array([[np.cos(-object_theta), np.sin(-object_theta)],
                           [-np.sin(-object_theta), np.cos(-object_theta)]])

    # object: front left corner
    if direction == 1:
        x_f = object_x + np.sin(-object_theta) * 0.5 * object_length - np.cos(-object_theta) * 0.5 * object_width
        y_f = object_y + np.cos(-object_theta) * 0.5 * object_length + np.sin(-object_theta) * 0.5 * object_width
    # object: front right corner
    else:
        x_f = object_x + np.sin(-object_theta) * 0.5 * object_length + np.cos(-object_theta) * 0.5 * object_width
        y_f = object_y + np.cos(-object_theta) * 0.5 * object_length - np.sin(-object_theta) * 0.5 * object_width

    # start position
    trajectory[0, 0] = x_f
    trajectory[0, 1] = y_f
    trajectory[0, 2] = 0
    trajectory[0, 3] = object_theta

    # auxiliary variables to save prior position
    temp_x = 0
    temp_y = 0

    # loop variables
    t = delta_t
    x_tilde = 0
    v_x_tilde = 0

    # calculate segment coordinates
    for i in range(1, n_seg):

        # x-coordinate and x-velocity of trajectory in object coordinates
        x_tilde = x_tilde + v_x_tilde * delta_t - direction * a_max_lat * delta_t ** 2 / 2
        v_x_tilde = v_x_tilde - direction * a_max_lat * delta_t

        # y-coordinate of trajectory in object coordinates
        y_tilde = object_vel * t

        # calculate heading of trajectory using difference quotient
        heading = np.arctan((x_tilde - temp_x) / (y_tilde - temp_y))

        # calculate current turn radius
        a_lat = np.cos(heading) * a_max_lat                                             # current lateral acceleration
        v_abs = np.sqrt((y_tilde - temp_y) ** 2 + (x_tilde - temp_x) ** 2) / delta_t    # current absolute velocity
        r = v_abs ** 2 / a_lat

        # limit lateral movement when turn radius is below minimum
        if r < turn_rad:
            prev_phi = object_theta - trajectory[i - 1, 3]                              # previous phi
            delta_phi = - direction * (y_tilde - temp_y) / (np.cos(prev_phi) * turn_rad)
            x_tilde = temp_x + np.tan(delta_phi + prev_phi) * (y_tilde - temp_y)
            heading = np.arctan((x_tilde - temp_x) / (y_tilde - temp_y))
            v_x_tilde = (x_tilde - temp_x) / delta_t

        # transform coordinates into global system an save in bound
        trajectory[i, 0] = x_f + rot_mat_in[0, 0] * x_tilde + rot_mat_in[0, 1] * y_tilde
        trajectory[i, 1] = y_f + rot_mat_in[1, 0] * x_tilde + rot_mat_in[1, 1] * y_tilde
        trajectory[i, 2] = t
        trajectory[i, 3] = object_theta - heading

        # increment time
        t = t + delta_t

        # save position for next iteration
        temp_x = x_tilde
        temp_y = y_tilde

    return trajectory


def check_intersect(u1: np.ndarray,
                    u2: np.ndarray,
                    v1: np.ndarray,
                    v2: np.ndarray) -> tuple:
    """
    Checks if two line segments given by coordinates u1(start-point),u2 (end-point) and v1,v2 intersect.
    Code is based on ideas in: http://geomalgorithms.com/a05-_intersect-1.html

    :param u1:               start point of segment u as a np.array with column [x, y]
    :param u2:               end point of segment u as a np.array with column [x, y]
    :param v1:               start point of segment v as a np.array with column [x, y]
    :param v2:               end point of segment v as a np.array with column [x, y]

    :returns:
        * **intersect** -    boolean => true when segments intersect
        * **intersection** - coordinates of intersection as a np.array with column [x, y]
    """

    # define threshold for parallel intersection test
    epsilon = 0.000001

    # helper vectors
    u = u2 - u1
    v = v2 - v1
    w = u1 - v1

    dnom = perp_dot(v, u)
    if -epsilon < dnom < epsilon:  # parallel line
        return 0, np.array(([0, 0]))

    s_i = -perp_dot(v, w) / dnom
    if s_i <= 0 or s_i >= 1:  # out of segment u
        return 0, np.array(([0, 0]))

    t_i = -perp_dot(u, w) / dnom
    if t_i <= 0 or t_i >= 1:  # out of segment v
        return 0, np.array(([0, 0]))

    # intersection in between segment boundaries
    return 1, u1 + s_i * u


def perp_dot(a: np.ndarray,
             b: np.ndarray) -> np.ndarray:
    """
    This is a helper function to calculate the Perp Dot Product, which equals a two dimensional dot product, in which
    the first vector ist replaced by the perpendicular of itself.
    More information: https://mathworld.wolfram.com/PerpDotProduct.html

    :param a:       multiplier vector as a np.ndarray with columns [x, y]
    :param b:       multiplicand vector as a np.ndarray with columns [x, y]


    :returns:
        * **dot** - product as a np.ndarray with columns [x, y]
    """

    return np.dot(np.array([-a[1], a[0]]), b)
