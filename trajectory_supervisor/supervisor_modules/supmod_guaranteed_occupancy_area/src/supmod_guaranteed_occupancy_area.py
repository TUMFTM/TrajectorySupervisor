import os
import json
import logging
import hashlib
import numpy as np
import configparser
import shapely.geometry

import scenario_testing_tools as stt
import trajectory_supervisor


class SupModGuaranteedOccupancyArea(object):
    """
    Class that checks, whether the trajectory of the ego-vehicle against the guaranteed occupation (based on vehicle
    dynamics) of other vehicles. The guaranteed occupation of other vehicles is offline computed within a discretized
    velocity grid and then loaded online.

    :Authors:
        * Maximilian Bayerlein
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        20.01.2020

    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 localgg: np.ndarray,
                 ax_max_machines: np.ndarray,
                 veh_params: dict,
                 supmod_config_path: str,
                 occupation_map_path: str = None) -> None:
        """
        Init the guaranteed occupancy SupMod.

        :param localgg:             maximum g-g-values at a certain position on the map (currently, only the max a used)
        :param ax_max_machines:     long. acceleration limits by the electrical motors, columns [vx, ax_max_machines];
                                    velocity in m/s, accelerations in m/s2. They should be handed in without considering
                                    drag resistance, i.e. simply by calculating F_x_drivetrain / m_veh
        :param veh_params:          dict of vehicle parameters; must hold the following keys:
                                      veh_width -     width of the ego-vehicle [in m]
                                      veh_length -    length of the ego-vehicle [in m]
                                      turn_rad -      turning radius of vehicle [in m]
        :param supmod_config_path:  path pointing to the config file hosting relevant parameters
        :param occupation_map_path: (optional) path pointing to location where occupation map can be stored to avoid
                                    offline calculation on every launch

        """

        # check shape of ax_max_machines
        if ax_max_machines.shape[1] != 2:
            raise RuntimeError("ax_max_machines must consist of the two columns [vx, ax_max_machines]!")

        # top level path (module directory)
        file_dict = {'supmod_config': supmod_config_path,
                     'occupation_map': occupation_map_path}

        # set vehicle parameters
        self.__veh_width = veh_params['veh_width']
        self.__veh_length = veh_params['veh_length']
        self.__veh_turn_rad = veh_params['turn_rad']

        # read configuration file
        safety_param = configparser.ConfigParser()
        if not safety_param.read(file_dict['supmod_config']):
            raise ValueError('Specified config file does not exist or is empty!')

        # -- GET PARAMETERS FOR THE OCCUPATION AREA ANALYSIS  FROM CONFIGURATION FILE ----------------------------------
        self.__t_max_occ = safety_param.getfloat('GUAR_OCCUPATION', 't_max')
        self.__d_t_occ = safety_param.getfloat('GUAR_OCCUPATION', 'dt')
        v_max_occ = safety_param.getfloat('GUAR_OCCUPATION', 'v_max')
        d_v_occ = safety_param.getfloat('GUAR_OCCUPATION', 'd_v')
        nmb_states_occ = safety_param.getint('GUAR_OCCUPATION', 'nmb_states')

        # define vehicle footprint (relative to COG in middle)
        self.__veh_shape = get_veh_footprint(length=self.__veh_length,
                                             width=self.__veh_width)

        # -- LOAD OCCUPATION AREA DATA ---------------------------------------------------------------------------------
        self.__occupation_maps = None

        # get md5 for current parameter file
        calculated_md5 = md5(file_dict['supmod_config'])

        # load occupation map, if existing
        if file_dict['occupation_map'] is not None and os.path.isfile(file_dict['occupation_map']):
            with open(file_dict['occupation_map'], 'r') as f:
                self.__occupation_maps = json.load(f)

        # if no file existing or parameters changed
        if self.__occupation_maps is None or calculated_md5 != self.__occupation_maps['md5_key']:
            # Warn user
            logging.getLogger("supervisor_logger").info("supmod_guaranteed_occupancy | Could not find a matching "
                                                        "occupation map for the given parameterization! Triggered "
                                                        "recalculation.")

            # calculate new maps
            self.__occupation_maps = trajectory_supervisor.supervisor_modules.supmod_guaranteed_occupancy_area.src.\
                guar_occ_calc.guar_occ_calc(t_max=self.__t_max_occ,
                                            d_t=self.__d_t_occ,
                                            v_max=v_max_occ,
                                            d_v=d_v_occ,
                                            localgg=localgg,
                                            ax_max_mach=ax_max_machines,
                                            nmb_states=nmb_states_occ,
                                            veh_length=self.__veh_length,
                                            veh_width=self.__veh_width,
                                            turn_rad=self.__veh_turn_rad,
                                            md5_key=calculated_md5,
                                            export_path=file_dict['occupation_map'])

            # notify user on completion
            logging.getLogger("supervisor_logger").info("Occupation map creation succeeded.")

        # remove MD5 key, if present
        if 'md5_key' in self.__occupation_maps:
            del self.__occupation_maps['md5_key']

    # ------------------------------------------------------------------------------------------------------------------
    # DESTRUCTOR -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __del__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # CALC SCORE -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def calc_score(self,
                   ego_traj: np.ndarray,
                   objects: dict) -> tuple:
        """
        Assesses the trajectory safety for a given time instance regarding collision with other vehicles regarding the
        occupation area of these vehicles. Therefore, the following steps are executed:

        #. Calculate ego occupation
        #. Generate occupation area for every object (by loading occupation map)
        #. Check for collision (relating ego occupation and occupation areas of objects)

        :param ego_traj:                data of the ego vehicle [s, pos_x, pos_y, heading, curvature, velocity,
                                        acceleration] for a given time instance
        :param objects:                 data of all objects on the course, structured in a dictionary holding
                                        information about x_pos, y_pos, heading, type, form, size, id and velocity
        :returns:
            * **safety** -              boolean value false: unsafe, true: safe
            * **safety_parameters** -   dict of related parameters (relevant for score) to be logged

        """

        # init collision tracker list (to be extended by scores of objects in object-list)
        collision_tracker = [False]
        safety_parameters = dict()

        # -- GET OCCUPATION MAP OF EGO-VEHICLE -------------------------------------------------------------------------

        # get time-steps of obj-occupation
        t_obj_occ = np.linspace(0.0, self.__t_max_occ, int(self.__t_max_occ / self.__d_t_occ))

        ego_occupation_map = get_ego_occupation(ego_traj=ego_traj,
                                                t_occ=t_obj_occ,
                                                veh_shape=self.__veh_shape)

        # for all objects (that hold the listed keys)
        for obj_key in objects.keys():
            if ('v_x' and 'Y' and 'X' and 'theta') in objects[obj_key].keys():
                # -- SET OCCUPATION MAP TO POSITION OF VEHICLE AND ROTATE IT TO HEADING DIRECTION ----------------------
                obj_occupation_map = get_obj_occupation(occupation_maps=self.__occupation_maps,
                                                        object_vel=objects[obj_key]['v_x'],
                                                        object_psi=objects[obj_key]['theta'] + np.pi / 2,  # north = 0.0
                                                        object_x=objects[obj_key]['X'],
                                                        object_y=objects[obj_key]['Y'])

                # since the object occupation is pre-computed, raise warning if assumed vehicle dimensions differ a lot
                # compared to the actual received ones
                if objects[obj_key]['form'] != "rectangle":
                    logging.getLogger("supervisor_logger").warning('supmod_guaranteed_occupation | Found object ('
                                                                   + obj_key + ') of form "' + objects[obj_key]['form']
                                                                   + '" but expected "rectangle". Used dims of '
                                                                     'ego-vehicle instead!')

                elif (abs(objects[obj_key]['width'] - self.__veh_width) > 0.1
                      or abs(objects[obj_key]['length'] - self.__veh_length) > 0.1):
                    logging.getLogger("supervisor_logger").warning('supmod_guaranteed_occupation | Precomputed '
                                                                   'occupancy set was calculated for vehicle dimensions'
                                                                   ' {:.2f}x{:.2f} but obj "{}" has dimensions '
                                                                   '{:.2f}x{:.2f}! Occupation estimation might be '
                                                                   'wrong!'.format(self.__veh_length, self.__veh_width,
                                                                                   obj_key, objects[obj_key]['length'],
                                                                                   objects[obj_key]['width']))

                # -- CHECK FOR INTERSECTION BETWEEN TRAJECTORY AND OCCUPATION AREA -------------------------------------
                collision_tmp, coll_set_ego, coll_set_obj = check_collision(obj_occupation_map=obj_occupation_map,
                                                                            ego_occupation_map=ego_occupation_map,
                                                                            obj_key=str(obj_key))

                collision_tracker.append(collision_tmp)

                # if collision, add collision outlines to parameter dict
                if collision_tmp:
                    safety_parameters["gocccolego_" + obj_key] = [coll_set_ego[:, 0], coll_set_ego[:, 1]]
                    safety_parameters["gocccolobj_" + obj_key] = [coll_set_obj[:, 0], coll_set_obj[:, 1]]

                # store parameters to dict
                for t_stamp in obj_occupation_map.keys():
                    params_key = "gocc_" + obj_key + "_" + str(t_stamp)
                    safety_parameters[params_key] = obj_occupation_map[t_stamp].transpose().tolist()

        # -- MERGE COLLISION SCORES ------------------------------------------------------------------------------------
        # only safe, if not one single collision is detected
        safety_score = not any(collision_tracker)

        return safety_score, safety_parameters


def get_obj_occupation(occupation_maps: dict,
                       object_vel: float,
                       object_psi: float,
                       object_x: float,
                       object_y: float) -> dict:
    """
    From a set of pre-computed occupation-maps ('`occupation_maps`'), retrieve the relevant one (based on
    object-velocity and velocity discretization) and then rotate and translate it to the pose of the object.

    :param occupation_maps:     dict containing all occupation maps
    :param object_vel:          velocity of the vehicle
    :param object_psi:          heading of the vehicle
    :param object_x:            x-position of the vehicle
    :param object_y:            y-position of the vehicle
    :returns:
        * **occupation_map** -  dict of occupation areas with:

            * keys holding the evaluated time-stamps
            * values holding the outlining coordinates as a np.ndarray with columns [x, y]

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        01.04.2020
    """

    # -- EXTRACT PROPER OCCUPATION MAP SERIES FOR GIVEN VELOCITY -------------------------------------------------------
    # extract available velocity levels
    occ_keys = list(occupation_maps.keys())
    occ_keys_float = [float(v0) for v0 in occ_keys]

    # find closest velocity level (key) to current velocity and extract accompanied map
    occupation_map = dict(occupation_maps[occ_keys[np.argpartition(np.abs(np.array(occ_keys_float)
                                                                          - object_vel), 1)[0]]])

    # -- ROTATION MATRIX TURNING THE OCCUPATION MAP IN THE DIRECTION OF THE VEHICLES HEADING ---------------------------
    rot_mat = np.array([[np.cos(object_psi), -np.sin(object_psi)], [np.sin(object_psi), np.cos(object_psi)]])
    object_pos = np.array([object_x, object_y])

    # -- ROTATE POINTS FOR ALL TIME INTERVALS AND SHIFT TO OBJECT POSITION ---------------------------------------------
    for t in occupation_map.keys():

        # convert stored lists of x and y coordinates to coordinate columns
        occupation_map[t] = np.column_stack(occupation_map[t])

        for i in range(occupation_map[t].shape[0]):
            occupation_map[t][i, :] = np.dot(rot_mat, occupation_map[t][i, :]) + object_pos

    return occupation_map


def get_ego_occupation(ego_traj: np.ndarray,
                       t_occ: np.ndarray,
                       veh_shape: np.ndarray) -> dict:
    """
    Get occupation over time for the ego vehicle (and return it as dict with the time-stamps as keys).

    :param ego_traj:            ego-trajectory with columns [s, x, y, psi, kappa, v, a]
    :param t_occ:               time-stamps, for which a ego-occupancy should be returned
    :param veh_shape:           numpy array holding the rel. coordinates (based on COG) of the vehicle-footprint corners
    :returns:
        * **occupation_map** -  dict of occupation areas with keys holding the evaluated time-stamps and values holding
          the outlining coordinates as a np.ndarray with columns [x, y]

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        01.04.2020
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
        occupation_map[t] = np.array([np.dot(rot_mat, veh_shape[0, :]) + pos,
                                      np.dot(rot_mat, veh_shape[1, :]) + pos,
                                      np.dot(rot_mat, veh_shape[2, :]) + pos,
                                      np.dot(rot_mat, veh_shape[3, :]) + pos])

    return occupation_map


def check_collision(obj_occupation_map: dict,
                    ego_occupation_map: dict,
                    obj_key: str) -> tuple:
    """
    This function checks for collisions by analyzing if matching time-stamps of the ego and object occupation map
    intersect at any point.

    :param obj_occupation_map:   dict of occupation areas with keys holding the evaluated time-stamps and values holding
                                 the outlining coordinates as a np.ndarray with columns [x, y]
    :param ego_occupation_map:   dict of occupation areas with keys holding the evaluated time-stamps and values hodling
                                 the outlining coordinates as a np.ndarray with columns [x, y]
    :param obj_key:              object key
    :returns:
        * **collision** -        returns true, if a collision is detected
        * **coll_set_ego** -    None if no collision occurred, else np.array of colliding outline with columns [x, y]
        * **coll_set_obj** -    None if no collision occurred, else np.array of colliding outline with columns [x, y]

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>
        * Maximilian Bayerlein

    :Created on:
        01.04.2020

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
        # create shapes from ego-footprint and occupation-set
        occ_line = shapely.geometry.Polygon(obj_occupation_map[t_stamp_obj])
        ego_line = shapely.geometry.Polygon(ego_occupation_map[t_stamp_ego])

        # check for intersections (intersecting borders or one encapsulates other - all tackled with "intersects")
        collision = occ_line.intersects(ego_line)

        # if collision detected, return immediately
        if collision:
            logging.getLogger("supervisor_logger").warning("supmod_guaranteed_occupation | Detected collision with "
                                                           "object '" + obj_key + "' at " + "%.2fs into the "
                                                           "ego-trajectory." % t_stamp_ego)

            # return colliding sets
            coll_set_ego = ego_occupation_map[t_stamp_ego]
            coll_set_obj = obj_occupation_map[t_stamp_obj]

            return True, coll_set_ego, coll_set_obj

    # no collision detected
    return False, None, None


def get_veh_footprint(length: float,
                      width: float) -> np.ndarray:
    """
    Returns all exterior points of a vehicle, relative to a COG residing in the middle of the shape at (0, 0).

    :param length:          length of the vehicle
    :param width:           width of the vehicle
    :returns:
        * **footprint** -   exterior coordinates, numpy array with columns x, y

    """

    # define vehicle footprint (relative to COG in middle)
    return np.array([[0.5 * width, 0.5 * length],    # front right
                     [-0.5 * width, 0.5 * length],   # front left
                     [-0.5 * width, -0.5 * length],  # rear left
                     [0.5 * width, -0.5 * length]])  # rear right


def md5(fname) -> str:
    """
    Get the md5-hash for a given file.

    :param fname:           path to file of interest
    :returns:
        * **hash_md5** -    md5-hash for the provided file

    """

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
