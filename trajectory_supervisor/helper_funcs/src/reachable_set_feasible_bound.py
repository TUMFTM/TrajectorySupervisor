import numpy as np
import shapely.geometry
import shapely.errors
import trajectory_supervisor


class ReachSetFeasible(object):
    """
    Class that handles calculation of a simple reachable set, that is limited by the track bounds (including a
    maneuver that ensures to be parallel to the bound on the outermost position)
    """

    def __init__(self,
                 bound_l: np.ndarray,
                 bound_r: np.ndarray,
                 localgg: np.ndarray,
                 ax_max_machines: np.ndarray,
                 closed: bool,
                 trim_set_to_bounds: bool = False) -> None:
        """
        :param bound_l:             coordinates of the left bound (numpy array with columns x, y)
        :param bound_r:             coordinates of the right bound (numpy array with columns x, y)
        :param localgg:             maximum g-g-values at a certain position on the map (currently, only the max a used)
        :param ax_max_machines:     velocity dependent maximum acceleration (motor limits)
        :param trim_set_to_bounds:  if set to 'True', regions of the reachable set outside of the track are removed
                                    (computationally demanding)

        """

        self.__tangential_t = dict()
        self.__index_track = dict()

        self.__bound_l = bound_l
        self.__bound_r = bound_r

        self.__localgg = localgg
        self.__ax_max_machines = ax_max_machines

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
                       obj_key: str,
                       obj_pos: np.ndarray,
                       obj_heading: float,
                       obj_vel: float,
                       obj_length: float,
                       obj_width: float,
                       obj_turn: float,
                       dt: float,
                       t_max: float) -> tuple:
        """
        Calculates a simple reachable set, that is limited by the track bounds (including a maneuver that ensures to be
        parallel to the bound on the outermost position)

        :param obj_key:         sting identifier of object (in order to initialize with previous solution, if existing)
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

        # -- check whether tangential_t already exist for object => if not create one
        if not (obj_key in self.__tangential_t):
            self.__tangential_t[obj_key] = {'left': 0,
                                            'right': 0}

        # -- check whether index track already exist for object => if not create one
        if not (obj_key in self.__index_track):
            self.__index_track[obj_key] = {'first_l': 0,
                                           'last_l': (self.__bound_l.shape[0] - 1),
                                           'first_r': 0,
                                           'last_r': (self.__bound_r.shape[0] - 1)}

        # -- time when acceleration and deceleration vehicle footprints divide
        a_max_dec = max(self.__localgg[:, 3])  # max deceleration
        a_max_acc = np.interp(obj_vel, self.__ax_max_machines[:, 0], self.__ax_max_machines[:, 1])  # max acceleration
        t_div = np.sqrt(2 * obj_length / (a_max_acc + a_max_dec))

        # -- Calculate intersection between boundary of occupancy and track boundaries--------------------------
        # -- left boundary ----------------
        trajectory_l, bound_reachset_l, index_track, tangential_t = trajectory_supervisor.helper_funcs.src.\
            bound_trajectory.get_bound_trajectory(object_x=obj_pos[0],
                                                  object_y=obj_pos[1],
                                                  object_theta=obj_heading,
                                                  object_vel=obj_vel,
                                                  direction=1,
                                                  localgg=self.__localgg,
                                                  long_step=3,
                                                  bound=self.__bound_l,
                                                  index_track=[self.__index_track[obj_key]['first_l'],
                                                               self.__index_track[obj_key]['last_l']],
                                                  object_length=obj_length,
                                                  object_width=obj_width,
                                                  tangential_t=self.__tangential_t[obj_key]['left'],
                                                  t_div=t_div,
                                                  delta_t=dt,
                                                  object_turn_rad=obj_turn,
                                                  closed=self.__closed)

        self.__index_track[obj_key]['first_l'], self.__index_track[obj_key]['last_l'] = index_track
        self.__tangential_t[obj_key]['left'] = tangential_t

        # -- right boundary --------------
        trajectory_r, bound_reachset_r, index_track, tangential_t = trajectory_supervisor.helper_funcs.src.\
            bound_trajectory.get_bound_trajectory(object_x=obj_pos[0],
                                                  object_y=obj_pos[1],
                                                  object_theta=obj_heading,
                                                  object_vel=obj_vel,
                                                  direction=-1,
                                                  localgg=self.__localgg,
                                                  long_step=3,
                                                  bound=self.__bound_r,
                                                  index_track=[self.__index_track[obj_key]['first_r'],
                                                               self.__index_track[obj_key]['last_r']],
                                                  object_length=obj_length,
                                                  object_width=obj_width,
                                                  tangential_t=self.__tangential_t[obj_key]['right'],
                                                  t_div=t_div,
                                                  delta_t=dt,
                                                  object_turn_rad=obj_turn,
                                                  closed=self.__closed)

        self.__index_track[obj_key]['first_r'], self.__index_track[obj_key]['last_r'] = index_track
        self.__tangential_t[obj_key]['right'] = tangential_t

        # ------ calculate simple reachable set and reduce size by bound trajectories ----------------------------------
        # get simple reach set
        reach_set = trajectory_supervisor.helper_funcs.src.reachable_set_simple.\
            simple_reachable_set(obj_pos=obj_pos,
                                 obj_heading=obj_heading,
                                 obj_vel=obj_vel,
                                 obj_length=obj_length,
                                 obj_width=obj_width,
                                 dt=dt,
                                 t_max=t_max,
                                 a_max=a_max_dec)

        # get polygons from bound trajectories
        del_outline = np.row_stack((np.flipud(trajectory_l[:, 0:2]),
                                    trajectory_r[:, 0:2],
                                    trajectory_r[0, 0:2] + 300.0 * np.array((np.cos(trajectory_r[-1, 3]),
                                                                             np.sin(trajectory_r[-1, 3]))),
                                    obj_pos.T + 300.0 * np.array((np.cos(obj_heading), np.sin(obj_heading)))
                                    - 300.0 * np.array((np.cos(obj_heading + np.pi / 2),
                                                        np.sin(obj_heading + np.pi / 2))),
                                    obj_pos.T - 300.0 * np.array((np.cos(obj_heading), np.sin(obj_heading)))
                                    - 300.0 * np.array(
                                        (np.cos(obj_heading + np.pi / 2), np.sin(obj_heading + np.pi / 2))),
                                    trajectory_l[0, 0:2] - 300.0 * np.array((np.cos(trajectory_l[-1, 3]),
                                                                             np.sin(trajectory_l[-1, 3])))
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
