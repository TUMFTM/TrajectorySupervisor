import numpy as np
import math
import trajectory_planning_helpers as tph


def distance(pos_1,
             pos_2) -> float:
    # calculate distance between two poses
    length = np.sqrt(pow((pos_1[0] - pos_2[0]), 2) + pow((pos_1[1] - pos_2[1]), 2))
    return length
# ----------------------------------------------------------------------------------------------------------------------


def calc_center_line(bound_l: np.ndarray,
                     bound_r: np.ndarray) -> tuple:
    """
    Calculates the center-line as well as the s-coordinate along this line for a pair of given bounds. The bounds must
    hold the same amount of elements.

    :param bound_l:             bound of the left track
    :param bound_r:             bound of the right track
    :returns:
        * **center_line** -     the cartesian coordinates of the center line
        * **s** -               the cumulative distance of every points after interpolation (s-coordinate)

    :Authors:
        * Yujie Lian
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        02.06.2019

    """

    # calculate center line
    center_line = 0.5 * (bound_r + bound_l)

    # calculating the Frenet s coordinate (along the track):
    # 1. calculate the distance between the points on the central line one by one
    # 2. Add the distance up
    s = np.cumsum(np.sqrt(np.sum(np.power(np.diff(center_line, axis=0), 2), axis=1)))
    s = np.insert(s, 0, 0.0)

    return center_line, s
# ----------------------------------------------------------------------------------------------------------------------


def angle3pt(a: tuple, b: tuple, c: tuple) -> float:
    """
    Calculate the angle by turning from coordinate a to c around b.

    :param a:             coordinate a (x, y)
    :param b:             coordinate b (x, y)
    :param c:             coordinate c (x, y)
    :returns:
        * **ang** -       angle between a and c

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        02.06.2019
    """
    ang = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])

    if ang > math.pi:
        ang -= 2 * math.pi
    elif ang <= -math.pi:
        ang += 2 * math.pi

    return ang
# ----------------------------------------------------------------------------------------------------------------------


def global_2_lane_based(pos: np.ndarray,
                        center_line: np.ndarray,
                        s_course: np.ndarray,
                        closed: bool = False) -> tuple:
    """
    This function transforms a global coordinate into the lane_based coordinate system by the following steps:

    #. find the intersection point of the vertical line of the input point and the center-line (global coordinate)
    #. calculate the distance between the intersection point and the input point ==> new x coordinate
    #. use the vector pointing from the intersection and the vector in the direction of the curve (from the intersection
       point to the next point on the central curve) to decide whether the point is on the right or left hand side
    #. correspondingly assign the proper sing (+/-) to the new x coordinate
    #. return the new coordinate in the lane_based coordinate system

    :param pos:                  the input position/coordinate(e.g the pos of ego vehicle or the objective vehicle)
    :param center_line:          the coordinate of the center_line line in global coordinate
    :param s_course:             the cumulative distance along the center_line line
    :param closed:               is the race course a closed or an open circuit. Default closed circuit
    :returns:
        * **n** -                the n coordinate in the lane_based system (lateral)
        * **s** -                the s coordinate in the lane_based system (longitudinal)
        * **angle_head_track** - the angle of the heading_track

    :Authors:
        * Yujie Lian
        * Yves Huberty
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        02.06.2019
    """
    # calculate squared distances between path array and reference poses
    distances2 = np.power(center_line[:, 0] - pos[0], 2) + np.power(center_line[:, 1] - pos[1], 2)

    # get smallest index
    idx_nb = np.argmin(distances2)

    if closed:
        idx1 = idx_nb - 1
        idx2 = idx_nb + 1
        idx3 = idx_nb + 2
        if idx2 > (center_line.shape[0] - 1):
            idx2 = 0
            idx3 = 1
        if idx3 > (center_line.shape[0] - 1):
            idx3 = 0
    else:
        idx1 = max(idx_nb - 1, 0)
        idx2 = min(idx_nb + 1, np.size(center_line, axis=0) - 1)
        idx3 = min(idx_nb + 2, np.size(center_line, axis=0) - 2)

    # get angle between input point, closest point and next point on center line
    ang1 = abs(angle3pt(tuple(pos), center_line[idx_nb, :], center_line[idx2, :]))

    # Extract neighboring points (A and B)
    if ang1 <= math.pi / 2:
        a_pos = center_line[idx_nb, :]
        b_pos = center_line[idx2, :]
        c_pos = center_line[idx3, :]
    else:
        a_pos = center_line[idx1, :]
        b_pos = center_line[idx_nb, :]
        c_pos = center_line[idx2, :]

    # calculate the heading of the track using the next center_line point (in 0 = north convention)
    heading_track = b_pos - a_pos
    angle_heading_track = tph.normalize_psi.normalize_psi(math.atan2(heading_track[1],
                                                                     heading_track[0]) - 1 / 2 * math.pi)

    # get sign (left / right of center line) by calculating angle between point of interest (pos) and center line
    sign = -np.sign(angle3pt(b_pos, tuple(pos), a_pos))

    # get point (s_pos) perpendicular on the line between the two closest points
    # https://stackoverflow.com/questions/10301001/perpendicular-on-a-line-segment-from-a-given-point
    temp = (((pos[0] - a_pos[0]) * (b_pos[0] - a_pos[0]) + (pos[1] - a_pos[1]) * (b_pos[1] - a_pos[1]))
            / (np.power(b_pos[0] - a_pos[0], 2) + np.power(b_pos[1] - a_pos[1], 2)))
    s_pos = [a_pos[0] + temp * (b_pos[0] - a_pos[0]), a_pos[1] + temp * (b_pos[1] - a_pos[1])]

    # calculate distance between closest point on center line and s_pos
    ds = np.sqrt(np.power(a_pos[0] - s_pos[0], 2) + np.power(a_pos[1] - s_pos[1], 2))

    # calculate distance between point of interest (pos) and s_pos
    n = np.sqrt(np.power(pos[0] - s_pos[0], 2) + np.power(pos[1] - s_pos[1], 2)) * sign

    # Calculate length of line segment [a_pos, b_pos]
    len_ab = np.sqrt(np.power(b_pos[0] - a_pos[0], 2) + np.power(b_pos[1] - a_pos[1], 2))

    # Calculate length of segment [a_pos, b_pos] parallel to center line but moved by n
    len_ab_offset = len_ab - n * math.tan(angle3pt(a_pos, b_pos, c_pos))

    # Calculate reduced ds
    ds_reduced = ds * len_ab / len_ab_offset

    # get total s_course
    if ang1 <= math.pi / 2:
        s = s_course[idx_nb] + ds_reduced
    else:
        s = s_course[idx1] + ds_reduced

    return n, s, angle_heading_track
# ---------------------------------------------------------------------------------------------------------------------


def lane_based_2_global(p_lane_based: np.ndarray,
                        ref_line: np.ndarray,
                        s_course: np.ndarray,
                        closed: bool = False) -> list:
    """
    This function transforms a lane based coordinate to a global coordinate.
    This functions uses only an approximation. This function uses a linear interpolation between the points of the
    reference line.

    :param p_lane_based:         the lane based coordinate of the input point [p_n, p_s]
    :param ref_line:             the coordinate of the reference line in global coordinate
    :param s_course:             the cumulative distance along the center_line line
    :param closed:               is the race course a closed or an open circuit. Default closed circuit
    :returns:
        * **p_global** -         the global coordinates of the input point

    :Authors:
        * Yves Huberty

    :Created on:
        02.06.2019

    """

    p_n = p_lane_based[0]
    p_s = p_lane_based[1]

    # find the closest cumulative distance ('-1' -> np.searchsorted doc)
    id_min = np.searchsorted(s_course, p_s, side='right') - 1
    idx2 = id_min + 1
    idx3 = id_min + 2

    # assign a, b and c
    if closed:
        if idx2 > (ref_line.shape[0] - 1):
            a_pos = ref_line[id_min, :]
            b_pos = ref_line[0, :]
            c_pos = ref_line[1, :]
        elif idx3 > (ref_line.shape[0] - 1):
            a_pos = ref_line[id_min, :]
            b_pos = ref_line[id_min + 1, :]
            c_pos = ref_line[0, :]
        else:
            a_pos = ref_line[id_min, :]
            b_pos = ref_line[id_min + 1, :]
            c_pos = ref_line[id_min + 2, :]
    else:
        if idx2 > (ref_line.shape[0] - 1):
            a_pos = ref_line[id_min, :]
            b_pos = ref_line[id_min, :]
            c_pos = ref_line[id_min, :]
        elif idx3 > (ref_line.shape[0] - 1):
            a_pos = ref_line[id_min, :]
            b_pos = ref_line[id_min + 1, :]
            c_pos = ref_line[id_min + 1, :]
        else:
            a_pos = ref_line[id_min, :]
            b_pos = ref_line[id_min + 1, :]
            c_pos = ref_line[id_min + 2, :]

    # calculate ds
    ds = p_s - s_course[id_min]

    # calculate length of line segment [ab]
    len_ab = np.sqrt(np.power(b_pos[0] - a_pos[0], 2) + np.power(b_pos[1] - a_pos[1], 2))

    # calculate length of segment [a_pos, b_pos] parallel to center line but moved by n
    len_ab_offset = len_ab - p_n * math.tan(angle3pt(a_pos, b_pos, c_pos))

    # calculate the original size of ds
    ds_stretched = ds * len_ab_offset / len_ab

    # calculate angle heading track
    heading_track = b_pos - a_pos
    angle_heading_track = tph.normalize_psi.normalize_psi(math.atan2(heading_track[1],
                                                                     heading_track[0]) - 1 / 2 * math.pi)

    # calculation of the global coordinate of the input
    p_x = (ref_line[id_min, 0] + ds_stretched * math.cos(angle_heading_track + math.pi / 2)
           + p_n * math.cos(angle_heading_track + math.pi))
    p_y = (ref_line[id_min, 1] + ds_stretched * math.sin(angle_heading_track + math.pi / 2)
           + p_n * math.sin(angle_heading_track + math.pi))
    p_global = [p_x, p_y]

    return p_global
# ---------------------------------------------------------------------------------------------------------------------


def vector_global_2_lane(vector: np.ndarray,
                         angle_heading_track: float) -> tuple:
    """
    This function transforms a vector in the global coordinate system into the lane-based coordinate system.
    The main idea is to rotate the vector based on the difference of angle between the heading of the track and the
    vector itself.

    :param vector:                the vector that needs to be transformed
    :param angle_heading_track:   the angle of the track-heading
    :returns:
        * **vector_lane** -       the vector in the lane-based coordinate system
        * **angle_vector_lane** - the transformed angle in the lane-based coordinate system

    :Authors:
        * Yujie Lian
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        06.07.2019
    """

    # angle using the convention(north: 0, east: -pi/2)
    angle_vector = tph.normalize_psi.normalize_psi(math.atan2(vector[1], vector[0]) - 1 / 2 * math.pi)

    # calculate the length of the vector
    len_vector = math.sqrt(pow(vector[0], 2) + pow(vector[1], 2))

    # calculate the new angle of the vector
    angle_vector_lane = tph.normalize_psi.normalize_psi(angle_vector - angle_heading_track)

    # assign the new angle to the vector in the lane_based coordinate system
    vector_lane = np.array([(math.sin(angle_vector_lane) * len_vector),
                            (math.cos(angle_vector_lane) * len_vector)])

    return vector_lane, angle_vector_lane
# ---------------------------------------------------------------------------------------------------------------------


def poly_global_2_lane(reachable_set_global: np.ndarray,
                       ref_line: np.ndarray,
                       s_course: np.ndarray,
                       closed: bool = False) -> list:
    """
    This function transforms the reachable set from a global coordinate to the lane-based coordinate. This function
    works only for reachable sets consisting of polygons with 6 points

    :param reachable_set_global:            the reachable set in global coordinates
    :param ref_line:                        the reference line for the lane based coordinate system
    :param s_course:                        the cumulative distance of the lane based coordinate system
    :param closed:                          boolean flag, whether the track is assumed closed or not
    :returns:
        * **reachable_set_lane_based** -    the reachable set in lane based coordinates

    :Authors:
        * Yves Huberty

    :Created on:
        04.04.2020

    """

    reachable_set_lane_based = []

    for poly_global in reachable_set_global:
        poly_lane_based = np.empty([len(poly_global), 3])
        i = 0
        for p_global in poly_global:
            p_n, p_s, p_angle_heading_track = global_2_lane_based(pos=p_global,
                                                                  center_line=ref_line,
                                                                  s_course=s_course,
                                                                  closed=closed)
            poly_lane_based[i, 0] = p_n
            poly_lane_based[i, 1] = p_s
            poly_lane_based[i, 2] = p_angle_heading_track
            i += 1

        reachable_set_lane_based.append(poly_lane_based)

    return reachable_set_lane_based
# ---------------------------------------------------------------------------------------------------------------------


def poly_lane_2_global(reachable_set_lane_based: list,
                       ref_line: np.ndarray,
                       s_course: np.ndarray,
                       closed: bool = False) -> list:
    """
    This function transforms a reachable-set from a lane-based coordinate system to the global coordinate system. This
    function works only for reachable sets consisting of polygons with 6 points.

    :param reachable_set_lane_based:        the reachable set in lane based coordinates
    :param ref_line:                        the reference line for the lane based coordinate system
    :param s_course:                        the cumulative distance of the lane based coordinate system
    :param closed:                          boolean flag, whether the track is assumed closed or not
    :returns:
        * **reachable_set_global** -        the reachable set in global coordinates

    :Authors:
        * Yves Huberty

    :Created on:
        28.04.2020

    """

    reachable_set_global = []

    for poly_lane_based in reachable_set_lane_based:
        nbr_items = np.size(poly_lane_based, 0)
        poly_global = np.empty([nbr_items, 2])
        i = 0
        for p_lane_based in poly_lane_based:
            poly_global[i, :] = lane_based_2_global(p_lane_based=p_lane_based,
                                                    ref_line=ref_line,
                                                    s_course=s_course,
                                                    closed=closed)
            i += 1

        reachable_set_global.append(poly_global)

    return reachable_set_global
# ---------------------------------------------------------------------------------------------------------------------
