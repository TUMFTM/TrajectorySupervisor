import numpy as np
import logging
import shapely.geometry


def check_collision_lanelet(ego_path: np.ndarray,
                            track_shape: shapely.geometry.Polygon,
                            col_width: float,
                            veh_length: float) -> tuple:
    """
    Asses the trajectory safety for a given times instance regarding collision with the wall. A driving tube (based on
    vehicle width) along the ego-path is checked against the track bounds.

    :param ego_path:            path data of the ego vehicle with the following columns: [pos_x, pos_y, heading]
    :param track_shape:         shapely object describing track
    :param col_width:           tracked width for collisions (with the trajectory lying in the center -> for minimal
                                safety use vehicle width, for conservative safety use vehicle diagonal)
    :param veh_length:          length of vehicle in meters
    :returns:
        * **safety** -          boolean value - 'False' for unsafe, 'True' for safe
        * **safety_param** -    safety parameter dict (logged parameters, here: intersection coordinates)

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        09.07.2020
    """

    # init safety-param dict
    safety_param = dict()

    # setup tube for ego trajectory (extend ego trajectory at start and end for half of vehicle length)
    de1 = np.diff(ego_path[:2, :], axis=0)[0]
    de2 = np.diff(ego_path[-2:, :], axis=0)[0]
    ego_path_ext = np.vstack((ego_path[0, :] - de1 * veh_length / max(2 * np.hypot(de1[0], de1[1]), 0.001),
                              ego_path,
                              ego_path[-1, :] + de2 * veh_length / max(2 * np.hypot(de2[0], de2[1]), 0.001)))
    ego_tube = shapely.geometry.LineString(ego_path_ext).buffer(col_width / 2)

    # -- check for intersection / boundary collision --
    bound_collision = not track_shape.contains(ego_tube)

    # if intersecting, raise warning and try to locate intersection location
    if bound_collision:
        # calculate location of intersection
        intersect_pnts = ego_tube.intersection(track_shape.boundary)

        # convert intersection points to list
        intersect_pnts_list = []

        if intersect_pnts.geom_type == 'LineString' and intersect_pnts.coords:
            intersect_pnts = intersect_pnts.coords.xy
            for i in range(len(intersect_pnts[0])):
                intersect_pnts_list.append([intersect_pnts[0][i], intersect_pnts[1][i]])

        elif intersect_pnts.geom_type == 'MultiLineString':
            for linestring in intersect_pnts:
                linestring = linestring.coords.xy
                for i in range(len(linestring[0])):
                    intersect_pnts_list.append([linestring[0][i], linestring[1][i]])

        elif intersect_pnts.geom_type == 'MultiPoint':
            for point in intersect_pnts:
                intersect_pnts_list.append([point.x, point.y])

        elif intersect_pnts.geom_type == 'Point':
            intersect_pnts_list.append([intersect_pnts.x, intersect_pnts.y])

        else:
            # unsupported intersection geometry - log warning
            log = logging.getLogger("supervisor_logger")
            log.warning('supmod_static_collision | Collision detected! Could not log / visualize location due to '
                        'unsupported intersection geometry (' + intersect_pnts.geom_type + ')')

        if intersect_pnts_list:
            # add coordinates to safety-param dict
            safety_param['stat_intersect'] = intersect_pnts_list

            # log warning
            log = logging.getLogger("supervisor_logger")
            log.warning('supmod_static_collision | Collision detected! For example, a collision in vicinity of the po'
                        'int ({:6.2f}, {:6.2f}) detected.'.format(intersect_pnts_list[0][0], intersect_pnts_list[0][1]))

    return not bound_collision, safety_param
