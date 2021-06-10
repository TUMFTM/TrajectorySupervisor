import json
import numpy as np
import shapely.geometry
import shapely.errors
import logging
import configparser
import scenario_testing_tools as stt

from trajectory_supervisor.supervisor_modules.supmod_rule_reach_sets.src.rules import RuleRoboraceSeasonAlpha, RuleF1
from trajectory_supervisor.helper_funcs.src import reachable_set_turn_limit, reachable_set_feasible_bound, \
    shapely_conversions, reachable_set_simple, path_matching


class SupModReachSets(object):
    """
    Supervisor module that uses reachable-set to check the dynamic environment. The reachable-sets are reduced using the
    race rules.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>
        * Yves Huberty

    :Created on:
        22.01.2020

    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 supmod_config_path: str,
                 veh_params: dict,
                 a_max: float) -> None:
        """
        Init the rule-based reachable set SupMod.

        :param supmod_config_path:  path to Supervisor config file
        :param veh_params:          dict of vehicle parameters; must hold the following keys:
                                      veh_width -     width of the ego-vehicle [in m]
                                      veh_length -    length of the ego-vehicle [in m]
        :param a_max:               maximum assumed acceleration for other vehicles
                                    (used to calculate the reachable set - one value for both lateral and longitudinal)

        """

        # -- read configuration file -----------------------------------------------------------------------------------
        reach_set_param = configparser.ConfigParser()
        if not reach_set_param.read(supmod_config_path):
            raise ValueError('Specified config file does not exist or is empty!')

        # -- get vehicle specifications --------------------------------------------------------------------------------
        self.__veh_width = veh_params['veh_width']
        self.__veh_length = veh_params['veh_length']

        # define vehicle footprint (relative to COG in middle)
        self.__veh_shape = np.array([[0.5 * self.__veh_width, 0.5 * self.__veh_length],  # front right
                                     [-0.5 * self.__veh_width, 0.5 * self.__veh_length],  # front left
                                     [-0.5 * self.__veh_width, -0.5 * self.__veh_length],  # rear left
                                     [0.5 * self.__veh_width, -0.5 * self.__veh_length]])  # rear right

        # -- get parameters for the dynamic safety assessment from configuration file ----------------------------------
        self.__tmax = reach_set_param.getfloat('RULE_REACH_SET', 't_max')        # prediction horizon in s
        self.__dt = reach_set_param.getfloat('RULE_REACH_SET', 'dt')             # temporal resolution of reach-set in s
        self.__amax = a_max

        # -- reachable-set calculation method --------------------------------------------------------------------------
        self.__reachset_method = reach_set_param.get('RULE_REACH_SET', 'reachset_method')
        self.__reachset_trim_to_bound = reach_set_param.getboolean('RULE_REACH_SET', 'reachset_trim_to_bound')

        # -- initialize Frenet frame -----------------------------------------------------------------------------------
        self.__ref_line = None
        self.__s_course = None
        self.__norm_vec = None
        self.__tw_left = None
        self.__tw_right = None
        self.__center_line = None

        self.__turn_radius = None

        # -- initialize track style (closed or open circuit) -----------------------------------------------------------
        self.__closed = False

        # init reach-set bound reduction
        self.__reach_set = None

        logging.getLogger('supervisor_logger').debug("supmod_rule_reach_sets | Completed initialization!")

        # -- initialize rules ------------------------------------------------------------------------------------------
        self.__rule_dict = {}
        if json.loads(reach_set_param.get('RULE_REACH_SET', 'rules_enabled'))['roborace_alpha']:
            t_trigger = reach_set_param.getfloat('RULE_REACH_SET', 't_trigger_roborace')
            self.__rule_dict['roborace_alpha'] = RuleRoboraceSeasonAlpha.RuleRoboraceSeasonAlpha(t_trigger=t_trigger)

        if json.loads(reach_set_param.get('RULE_REACH_SET', 'rules_enabled'))['f1']:
            param_alongside = json.loads(reach_set_param.get('RULE_REACH_SET', 'alongside_parameters'))
            param_defense = json.loads(reach_set_param.get('RULE_REACH_SET', 'defense_parameters'))
            self.__rule_dict['f1'] = RuleF1.RuleF1(veh_width=self.__veh_width,
                                                   veh_length=self.__veh_length,
                                                   dict_config_alongside=param_alongside,
                                                   dict_config_defense=param_defense)

        return

    # ------------------------------------------------------------------------------------------------------------------
    # DESTRUCTOR -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __del__(self) -> None:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # UPDATE MAP -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def update_map(self,
                   ref_line: np.ndarray,
                   norm_vec: np.ndarray,
                   tw_left: np.ndarray,
                   tw_right: np.ndarray,
                   localgg: np.ndarray,
                   ax_max_machines: np.ndarray,
                   turn_radius: float,
                   zone_file_path: str = None) -> None:
        """
        Update the internal map representation.

        :param ref_line:            cartesian coordinates of the reference line for the Frenet frame
        :param norm_vec:            normal-vectors based on each of the reference-line coordinates
        :param tw_left:             track-width to the left of reference point (along normal-vector)
        :param tw_right:            track-width to thr right of reference point (along normal-vector)
        :param zone_file_path:      path pointing to a .json-zone-file (if 'None', no zone-based reduction)
        :param localgg:             maximum g-g-values at a certain position on the map (currently, only the max a used)
        :param ax_max_machines:     velocity dependent maximum acceleration (motor limits)
        :param turn_radius:         (estimated) turn-radius of the vehicles [in m]

        """
        # -- update reference line -------------------------------------------------------------------------------------
        self.__ref_line = ref_line

        self.__turn_radius = turn_radius

        # -- calculate the cumulative distance along the reference line ------------------------------------------------
        self.__s_course = np.cumsum(np.sqrt(np.sum(np.power(np.diff(self.__ref_line, axis=0), 2), axis=1)))
        self.__s_course = np.insert(self.__s_course, 0, 0.0)
        self.__norm_vec = norm_vec
        self.__tw_left = tw_left
        self.__tw_right = tw_right

        # -- update track style (closed or open) -----------------------------------------------------------------------
        if np.hypot(ref_line[0, 0] - ref_line[-1, 0], ref_line[0, 1] - ref_line[-1, 1]) < 35.0:
            self.__closed = True
        else:
            self.__closed = False

        # -- init reach set calculation method -------------------------------------------------------------------------
        if self.__reachset_method == "bound":
            self.__reach_set = reachable_set_feasible_bound.\
                ReachSetFeasible(bound_l=ref_line - norm_vec * tw_left,
                                 bound_r=ref_line + norm_vec * tw_right,
                                 localgg=localgg,
                                 ax_max_machines=ax_max_machines,
                                 closed=self.__closed,
                                 trim_set_to_bounds=self.__reachset_trim_to_bound)
        elif self.__reachset_method == "turn":
            self.__reach_set = reachable_set_turn_limit.\
                ReachSetTurnLimit(bound_l=ref_line - norm_vec * tw_left,
                                  bound_r=ref_line + norm_vec * tw_right,
                                  localgg=localgg,
                                  closed=self.__closed,
                                  trim_set_to_bounds=self.__reachset_trim_to_bound)
        elif self.__reachset_method == "simple":
            if self.__reachset_trim_to_bound:
                self.__reach_set = reachable_set_simple.ReachSetSimple(bound_l=ref_line - norm_vec * tw_left,
                                                                       bound_r=ref_line + norm_vec * tw_right,
                                                                       closed=self.__closed)
            else:
                self.__reach_set = reachable_set_simple.ReachSetSimple()

        # -- update map representation of rules ------------------------------------------------------------------------
        if 'roborace_alpha' in self.__rule_dict:
            self.__rule_dict['roborace_alpha'].update_map_info(ref_line=self.__ref_line,
                                                               s_array=self.__s_course,
                                                               tw_left=self.__tw_left,
                                                               tw_right=self.__tw_right,
                                                               normal_vectors=self.__norm_vec,
                                                               zone_file_path=zone_file_path,
                                                               closed=self.__closed)

            # if valid roborace rules, disable f1 rules
            if self.__rule_dict['roborace_alpha'].status() and 'f1' in self.__rule_dict.keys():
                del self.__rule_dict['f1']

        if 'f1' in self.__rule_dict:
            self.__rule_dict['f1'].update_map_info(ref_line=self.__ref_line,
                                                   s_array=self.__s_course,
                                                   tw_left=self.__tw_left,
                                                   tw_right=self.__tw_right,
                                                   normal_vectors=self.__norm_vec,
                                                   closed=self.__closed)

    # ------------------------------------------------------------------------------------------------------------------
    # CALC SCORE -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def calc_score(self,
                   traj: np.ndarray,
                   objects: dict) -> tuple:
        """
        Calculate safety score (boolean value) based on reachable-sets
        Basic implementation checking if the centroid of the trajectory at a time t is inside the reachable-set of any
        object vehicle at the time t. Does not consider the width and length of object vehicles. The
        evaluation is binary. When the condition is violated once, the safety score is set to 0. Else it is 1.

        :param traj:                        trajectory of the ego-veh. with the following columns:
                                            [s, pos_x, pos_y, heading, curvature, velocity, acceleration]
        :param objects:                     data of all objects on in the scene, structured in a dictionary holding
                                            information about x_pos, y_pos, heading, type, form, size, id and velocity
        :returns:
            * **safety_reach_sets** -       safety score based on reachable sets. True = safe, False = possibly not safe
            * **rule_reach_set_params** -   parameter dict (here: coords of the reach-sets of all the object vehicles)

        """

        # check if map was provided
        if self.__ref_line is None:
            raise ValueError("Map information was not provided yet, could not perform score calculation!")

            # init safety score to true (safe)
        safety_reach_sets = True

        # -- calculate the reachable set for every object in objects and test for intersection with ego trajectory -----
        # initialize param dict. The reachable sets of the object vehicles are saved here
        rule_reach_set_params = dict()

        # calculate ego-occupation
        ego_occupation_map = get_ego_occupation(ego_traj=traj,
                                                t_occ=np.arange(0.0, self.__tmax + self.__dt / 2, self.__dt),
                                                veh_shape=self.__veh_shape)

        # -- PREPARE / PRE-PROCESS RULES -------------------------------------------------------------------------------
        if 'roborace_alpha' in self.__rule_dict:
            self.__rule_dict['roborace_alpha'].prep_rule_eval(pos_ego=traj[0, 1:3],
                                                              vel_ego=traj[0, 5],
                                                              object_list=objects)

        # -- CALC S COORDINATE FOR START AND END OF PLANNING HORIZON ---------------------------------------------------
        # get s coordinates of ego-trajectory start and end
        s_traj_ego_start = path_matching.get_s_coord(ref_line=self.__ref_line,
                                                     pos=traj[0, 1:3],
                                                     s_array=self.__s_course,
                                                     closed=self.__closed)[0]
        s_traj_ego_end = path_matching.get_s_coord(ref_line=self.__ref_line,
                                                   pos=traj[-1, 1:3],
                                                   s_array=self.__s_course,
                                                   closed=self.__closed)[0]

        # iterate through all the object vehicles
        for obj_key in objects.keys():
            s_veh = path_matching.get_s_coord(ref_line=self.__ref_line,
                                              pos=(objects[obj_key]['X'], objects[obj_key]['Y']),
                                              s_array=self.__s_course,
                                              closed=self.__closed)[0]

            # only objects in range of ego-trajectory (objects behind ego are ignored)
            if (s_traj_ego_start <= s_veh <= s_traj_ego_end) \
                    or (s_traj_ego_start > s_traj_ego_end) and (s_veh <= s_traj_ego_end or s_traj_ego_start <= s_veh):

                # -- CALCULATE REACHABLE SET FOR OBJECT ----------------------------------------------------------------
                if self.__reachset_method == "simple":
                    reach_set = self.__reach_set.\
                        calc_reach_set(obj_pos=np.array([[objects[obj_key]['X']], [objects[obj_key]['Y']]]),
                                       obj_heading=objects[obj_key]['theta'],
                                       obj_vel=objects[obj_key]['v_x'],
                                       dt=self.__dt,
                                       t_max=self.__tmax,
                                       a_max=self.__amax,
                                       obj_length=objects[obj_key].get('length', self.__veh_length),
                                       obj_width=objects[obj_key].get('width', self.__veh_width))

                elif self.__reachset_method == "turn":
                    reach_set, del_outline = self.__reach_set.\
                        calc_reach_set(obj_pos=np.array([[objects[obj_key]['X']], [objects[obj_key]['Y']]]),
                                       obj_heading=objects[obj_key]['theta'],
                                       obj_vel=objects[obj_key]['v_x'],
                                       obj_length=objects[obj_key].get('length', self.__veh_length),
                                       obj_width=objects[obj_key].get('width', self.__veh_width),
                                       obj_turn=self.__turn_radius,
                                       dt=self.__dt,
                                       t_max=self.__tmax,)

                    # uncomment for debugging(will highlight the deletion area for the reach sets in the log viewer)
                    # key = "bound_reach_set_outline_" + obj_key
                    # rule_reach_set_params[key] = del_outline

                elif self.__reachset_method == "bound":
                    reach_set, del_outline = self.__reach_set.\
                        calc_reach_set(obj_key=obj_key,
                                       obj_pos=np.array([[objects[obj_key]['X']], [objects[obj_key]['Y']]]),
                                       obj_heading=objects[obj_key]['theta'],
                                       obj_vel=objects[obj_key]['v_x'],
                                       obj_length=objects[obj_key].get('length', self.__veh_length),
                                       obj_width=objects[obj_key].get('width', self.__veh_width),
                                       obj_turn=self.__turn_radius,
                                       dt=self.__dt,
                                       t_max=self.__tmax)

                    # uncomment for debugging (will highlight the deletion area for the reach sets in the log viewer)
                    # key = "bound_reach_set_outline_" + obj_key
                    # rule_reach_set_params[key] = del_outline

                else:
                    raise ValueError("Provided reachset method (" + str(self.__reachset_method) + ") is not supported!")

                # -- PROCESS RULE BASED REDUCTIONS ---------------------------------------------------------------------
                deletion_patches = {}
                if 'roborace_alpha' in self.__rule_dict:
                    tmp_del_ptch = self.__rule_dict['roborace_alpha'].rule_eval(obj_key=obj_key)
                    deletion_patches = update_dict(dict1=deletion_patches, dict2=tmp_del_ptch)

                if 'f1' in self.__rule_dict:
                    tmp_del_ptch = self.__rule_dict['f1'].rule_eval(ego_pos=traj[0, 1:3],
                                                                    ego_s_pos=s_traj_ego_start,
                                                                    obj_pos=np.array((objects[obj_key]['X'],
                                                                                      objects[obj_key]['Y'])),
                                                                    obj_s_pos=s_veh,
                                                                    obj_key=obj_key)
                    deletion_patches = update_dict(dict1=deletion_patches, dict2=tmp_del_ptch)

                    # uncomment for debugging (will highlight the deletion area for the reach sets in the log viewer)
                    # if deletion_patches['all']:
                    #     key = "bound_reach_set_outline_" + obj_key
                    #     rule_reach_set_params[key] = deletion_patches['all'][-1]

                # -- REDUCE REACHABLE SET BASED ON RULES ---------------------------------------------------------------
                reach_set = reduce_reach_set(reach_set=reach_set,
                                             deletion_patches=deletion_patches)

                # -- CHECK FOR COLLISION -------------------------------------------------------------------------------
                coll_check, coll_set_ego, coll_set_obj = check_collision(obj_occupation_map=reach_set,
                                                                         ego_occupation_map=ego_occupation_map,
                                                                         obj_key=obj_key)
                safety_reach_sets = safety_reach_sets and not coll_check

                # if collision, add collision outlines to parameter dict
                if coll_check:
                    rule_reach_set_params["rresecolego_" + obj_key] = [coll_set_ego[:, 0], coll_set_ego[:, 1]]
                    rule_reach_set_params["rresecolobj_" + obj_key] = [coll_set_obj[:, 0], coll_set_obj[:, 1]]

                # add reduced reachable set to parameter dict
                for t_stamp in reach_set.keys():
                    params_key = "rrese_" + obj_key + "_" + str(t_stamp)
                    rule_reach_set_params[params_key] = [reach_set[t_stamp][:, 0], reach_set[t_stamp][:, 1]]

        return safety_reach_sets, rule_reach_set_params


def get_ego_occupation(ego_traj: np.ndarray,
                       t_occ: np.ndarray,
                       veh_shape: np.ndarray) -> dict:

    """
    Get occupation over time for the ego-vehicle (and return it as dict with the time-stamps as keys).

    :param ego_traj:            ego-trajectory with columns [s, x, y, psi, kappa, v, a]
    :param t_occ:               time-stamps, for which a ego-occupancy should be returned
    :param veh_shape:           numpy array holding the rel. coordinates (based on COG) of the vehicle-footprint corners
    :returns:
        * **occupation_map** -  dict of occupation areas with keys holding the evaluated time-stamps and values holding
          shapely polygons

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        08.07.2020

    """

    occupation_map = dict()

    # -- CALCULATE TIME-STAMPS OF TRAJECTORY POINTS --------------------------------------------------------------------
    v_avg = 0.5 * (ego_traj[1:, 5] + ego_traj[:-1, 5])
    t_stamps = np.cumsum(np.concatenate(([0], np.diff(ego_traj[:, 0]) / np.where(v_avg > 0.01, v_avg, 0.01))))

    for i, t in enumerate(t_occ):
        # -- INTERPOLATE VEHICLE POSE AT DESIRED TIME-STAMP ------------------------------------------------------------
        pos = np.array([np.interp(t, t_stamps, ego_traj[:, 1]), np.interp(t, t_stamps, ego_traj[:, 2])])
        heading = stt.interp_heading.interp_heading(heading=ego_traj[:, 3],
                                                    t_series=t_stamps,
                                                    t_in=t)

        # -- GET VEHICLE FOOTPRINT AT CONSIDERED TIME-STAMP ------------------------------------------------------------
        # determine rotation matrix
        rot_mat = np.array([[np.cos(heading), -np.sin(heading)],
                            [np.sin(heading), np.cos(heading)]])

        # rotate and translate ego footprint (edges of vehicle) - repeat fist point for closed shape
        occupation_map[t] = shapely.geometry.Polygon(np.array([np.dot(rot_mat, veh_shape[0, :]) + pos,
                                                               np.dot(rot_mat, veh_shape[1, :]) + pos,
                                                               np.dot(rot_mat, veh_shape[2, :]) + pos,
                                                               np.dot(rot_mat, veh_shape[3, :]) + pos]))

    return occupation_map


def reduce_reach_set(reach_set: dict,
                     deletion_patches: dict) -> dict:
    """
    This function reduces the size of a reachable set by subtracting the 'deletion_patch' along all time-stamps.

    :param reach_set:               dict of reachable areas with keys holding the evaluated time-stamps and values
                                    holding the outlining coordinates as a np.ndarray with columns [x, y]
    :param deletion_patches:        avoidance patches with keys holding relevant time-stamps and values a list of
                                    patches defined by outlining coordinates as np.ndarray with columns [x, y]
                                    NOTE: patches applicable for all time-stamps are provided with the key 'all'
                                    NOTE: temporal patches must hold the EXACT matching key as in "reach_set"
    :returns:
        * **reach_set_reduced** -   reduced reachable set (same format as 'rule_zone')

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        04.09.2020
    """

    reach_set_reduced = dict()

    # -- generate polygon list for "all" patches -----------------------------------------------------------------------
    poly_del_list_all = []
    deletion_patches_all = deletion_patches.get('all', [])

    for del_patch in deletion_patches_all:
        # convert deletion patches to shapely polygons
        poly_del_list_all.append(shapely.geometry.Polygon(del_patch))

    # loop through all time steps of considered reachable set
    for t_key in reach_set.keys():
        # -- convert reachable set to shapely polygon ------------------------------------------------------------------
        poly_reach_t = shapely.geometry.Polygon(reach_set[t_key])

        # -- extend deletion patch list by specific candidates for time-step -------------------------------------------
        # create copy of "all" deletion list
        poly_del_list = list(poly_del_list_all)

        # extract matching deletion patch for considered time-stamp, if present
        deletion_patches_t = deletion_patches.get(t_key, [])

        for del_patch in deletion_patches_t:
            # convert deletion patches to shapely polygons
            poly_del_list.append(shapely.geometry.Polygon(del_patch))

        # -- subtract deletion patches from reachable set polygon ------------------------------------------------------
        red_tmp = None

        for del_patch in poly_del_list:
            try:
                if red_tmp is None:
                    red_tmp = poly_reach_t.difference(del_patch)
                else:
                    red_tmp = red_tmp.difference(del_patch)

            except shapely.errors.TopologicalError:
                logging.getLogger('supervisor_logger').warning("supmod_rule_reach_sets | Skipped one reduction due to "
                                                               "resulting invalid shape.")

        # -- convert to coordinates ------------------------------------------------------------------------------------
        if red_tmp:
            # if coordinates present (not wiped out completely), extract outline coordinates
            red_tmp = shapely_conversions.extract_polygon_outline(shapely_geometry=red_tmp)

            # add outline coordinates to reach set
            if red_tmp is not None:
                reach_set_reduced[t_key] = red_tmp

        else:
            reach_set_reduced[t_key] = reach_set[t_key]

    return reach_set_reduced


def check_collision(obj_occupation_map: dict,
                    ego_occupation_map: dict,
                    obj_key: str = "") -> tuple:
    """
    This function checks if their might be a collision between the ego vehicle and the object vehicle based on the
    planed trajectory of the ego-vehicle and the reachable set of the object-vehicle. The function iterates through the
    steps of the trajectory comparing the planed position to the corresponding reachable set. If the planed position is
    inside the reachable set a collision is possible.
    If the length or the width are 0, the collision is checked for the center point of the vehicle. Else the collision
    is checked for the vehicles boundaries.


    :param obj_occupation_map:  dict of occupation areas with keys holding the evaluated time-stamps and values holding
                                the outlining coordinates as a np.ndarray with columns [x, y]
    :param ego_occupation_map:  dict of occupation areas with keys holding the evaluated time-stamps and values holding
                                a shapely polygon
    :param obj_key:             object key
    :returns:
        * **collision** -       'True' if a collision is possible, 'False' if a collision is excluded
        * **coll_set_ego** -    None if no collision occurred, else np.array of colliding outline with columns [x, y]
        * **coll_set_obj** -    None if no collision occurred, else np.array of colliding outline with columns [x, y]

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        08.07.2020

    """
    occ_t_keys = list(sorted(obj_occupation_map.keys()))
    occ_t_keys_float = [float(t) for t in occ_t_keys]

    # get maximum with half of delta between values (assumption: first time-stamp at 0.0)
    occ_t_max = max(occ_t_keys_float) * (1 + 1 / len(occ_t_keys_float) * 0.5)

    # -- GO THROUGH ALL TRAJECTORY TIME-STAMPS AND CHECK FOR COLLISION -------------------------------------------------
    for i, t_stamp_ego in enumerate(sorted(ego_occupation_map.keys())):
        # -- CHECK IF TIME-STAMP IS WITHIN RANGE OF OCCUPATION SET -----------------------------------------------------
        if float(t_stamp_ego) > occ_t_max:
            break

        # -- FIND CLOSEST TIME-STAMP IN OCCUPATION MAPS AND EXTRACT COORDS ---------------------------------------------
        t_stamp_obj = occ_t_keys[np.argpartition(np.abs(np.array(occ_t_keys_float) - float(t_stamp_ego)), 1)[0]]

        # -- CHECK POLYGONS FOR INTERSECTION AND OVERLAP ---------------------------------------------------------------
        # create shape from occupation-set
        occ_line = shapely.geometry.Polygon(obj_occupation_map[t_stamp_obj])

        # check for intersections (intersecting borders or one encapsulates other - all tackled with "intersects")
        collision = occ_line.intersects(ego_occupation_map[t_stamp_ego])

        # if collision detected, return immediately
        if collision:
            logging.getLogger('supervisor_logger').warning("supmod_rule_reach_sets | Detected collision with object"
                                                           " '" + obj_key + "' at " + "%.2fs into the ego-trajectory."
                                                           % t_stamp_ego)

            # return colliding sets
            coll_set_ego = np.column_stack(ego_occupation_map[t_stamp_ego].exterior.coords.xy)
            coll_set_obj = obj_occupation_map[t_stamp_obj]

            return True, coll_set_ego, coll_set_obj

    # no collision detected
    return False, None, None


def update_dict(dict1: dict,
                dict2: dict) -> dict:
    """
    Merges two dicts, assuming each key holds a list of values. If the two dicts hold the same keys, the corresponding
    values (lists) are fused.

    :param dict1:           first dictionary
    :param dict2:           second dictionary
    :returns:
        * **fused_dict** -  returned combined dict

    """

    # init returned dict
    fused_dict = dict1

    # for all keys in dict2
    for key in dict2.keys():
        if key in fused_dict:
            fused_dict[key].extend(dict2[key])

        else:
            fused_dict[key] = dict2[key]

    return fused_dict
