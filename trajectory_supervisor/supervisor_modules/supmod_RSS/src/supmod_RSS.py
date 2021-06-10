import math
import time
import json
import logging
import numpy as np
import configparser
from trajectory_supervisor.supervisor_modules.supmod_RSS.src import obj_path_storage as path_storage, \
    acceleration_profile as acc
from trajectory_supervisor.helper_funcs.src import lane_based_coordinate_system as lane
import trajectory_planning_helpers as tph


class SupModRSS(object):
    """
    Class handling the safety rating based on the RSS principle presented by Shalev Shwartz.

    NOTE: This SupMod is a drafted version and in BETA state, handle with care.

    .. note::
        For further details:

        S. Shalev-Shwartz, S. Shammah, and A. Shashua, “On a Formal Model of Safe and Scalable Self-driving Cars,” 2017.

    :Authors:
        * Yujie Lian
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        15.05.2019
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 supmod_config_path: str,
                 veh_params: dict,
                 localgg: np.ndarray):
        """
        Init the RSS SupMod.

        :param localgg:             maximum g-g-values at a certain position on the map (currently, only the max a used)
        :param supmod_config_path:  path pointing to the config file hosting relevant parameters
        :param veh_params:          dict of vehicle parameters; must hold the following keys:
                                      veh_width -     width of the ego-vehicle [in m]
                                      veh_length -    length of the ego-vehicle [in m]

        """

        self.__track_centerline = None
        self.__track_s_course = None

        # read configuration file
        rss_param = configparser.ConfigParser()
        if not rss_param.read(supmod_config_path):
            raise ValueError('Specified config file does not exist or is empty!')

        # -- set internal variables ------------------------------------------------------------------------------------
        self.__timer_safety_overlap = time.time()
        self.__obj_path = dict()

        # -- get parameters for the RSS safety assessment from configuration file --------------------------------------
        self.__t_react = rss_param.getfloat('RSS', 't_react')
        self.__t_debounce = rss_param.getfloat('RSS', 't_debounce')
        self.__a_long_max = max(localgg[:, 3])
        self.__a_lat_max = max(localgg[:, 4])
        self.__flag_curv = json.loads(rss_param.get('RSS', 'flag_curv'))
        self.__veh_width = veh_params['veh_width']
        self.__veh_length = veh_params['veh_length']

        return

    # ------------------------------------------------------------------------------------------------------------------
    # DESTRUCTOR -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __del__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # UPDATE MAP -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def update_map(self,
                   bound_l: np.ndarray,
                   bound_r: np.ndarray,) -> None:
        """
        Update the internal map representation.

        :param bound_l:       left bound of the track
        :param bound_r:       right bound of the track

        """

        # transform the data into the lane_based coordinate system - the coordinate of the center line of the bounds,
        # and the accumulated distance of every points (new Y)
        self.__track_centerline, self.__track_s_course = lane.calc_center_line(bound_l=bound_l,
                                                                               bound_r=bound_r)

        return

    # ------------------------------------------------------------------------------------------------------------------
    # CALC SCORE -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def calc_score(self,
                   ego_pos: np.ndarray,
                   ego_heading: float,
                   ego_curv: float,
                   ego_vel: float,
                   objects: dict):
        """
        This function uses kinematic data of the ego as well as the object vehicle and use the data to calculate the
        safety of the vehicle. Since the curvature of the object vehicle is needed but not provided, the path of the
        object vehicle is used as the input of the function in order to calculate the curvature of the vehicle.
        Due to the noise, the calculated curvature might not be that accurate, so there's also a flag which gives an
        option to use the calculated curvature or not.

        :param ego_pos:                 numpy array holding global x and y  of the ego-vehicle
        :param ego_heading:             heading of the ego-vehicle in the global coordinate frame
        :param ego_curv:                curvature of the path at the current position
        :param ego_vel:                 absolute velocity of the ego-vehicle
        :param objects:                 dict of objects, each holding a dict (with at least: 'X', 'Y', 'theta', 'v_x')
        :returns:
            * **safety_rss** -          the binary result of the safety assessment(true: safe, false: not safe)
            * **safety_parameters** -   a descriptive dict holding more detailed log / debugging information

        """

        # Check if map was received
        if self.__track_centerline is None:
            raise ValueError("Could not process data, since map was not received beforehand!")

        # Check if any objects vehicle present (return safe and no parameters)
        if len(objects.keys()) <= 0:
            return True, dict()

        # --------------------------------------------------------------------------------------------------------------
        # - EGO-VEHICLE | FRENET FRAME TRANSFORMATION and ACCELERATION LIMITS ------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # -- Frenet-Frame ----------------------------------------------------------------------------------------------
        # transform pos from global coordinate system to lane-based coordinate system (s, t)
        ego_pos_t, ego_pos_s, angle_heading_track = lane.global_2_lane_based(pos=ego_pos,
                                                                             center_line=self.__track_centerline,
                                                                             s_course=self.__track_s_course)
        # print("s: " + str(ego_pos_s) + "   t: " + str(ego_pos_t))

        # velocity
        ego_vel_x = ego_vel * math.sin(-ego_heading)
        ego_vel_y = ego_vel * math.cos(ego_heading)
        [ego_vel_t, ego_vel_s], ego_vel_lane_angle = lane.\
            vector_global_2_lane(vector=np.array([ego_vel_x, ego_vel_y]),
                                 angle_heading_track=angle_heading_track)

        # -- Acceleration Limits ---------------------------------------------------------------------------------------
        # calculate lon. acc. of ego-vehicle based on friction ellipse and limitation of the motor
        a_acc_long_max_ego, a_dec_long_max_ego = acc.acceleration_profile(a_lat_max=self.__a_lat_max,
                                                                          a_long_max=self.__a_long_max,
                                                                          velocity=ego_vel,
                                                                          velocity_angle=ego_vel_lane_angle,
                                                                          curvature=ego_curv,
                                                                          acc_type='long')

        # calculate the lateral acceleration of the ego based on the friction ellipse and limitation of the motor
        a_acc_left_max_ego, a_acc_right_max_ego = acc.acceleration_profile(a_lat_max=self.__a_lat_max,
                                                                           a_long_max=self.__a_long_max,
                                                                           velocity=ego_vel,
                                                                           velocity_angle=ego_vel_lane_angle,
                                                                           curvature=ego_curv,
                                                                           acc_type='lat')

        # for every object in dict
        safety_rss = 1
        safety_parameters = dict()
        for key in objects:
            # save the path of the object vehicle
            self.__obj_path[key] = path_storage.obj_path_storage(obj_path=self.__obj_path.get('test', None),
                                                                 obj_data=objects[key],
                                                                 desired_dis=0.5,
                                                                 desired_num=4)

            obj_pos = np.array([objects[key]['X'], objects[key]['Y']])
            obj_heading = objects[key]['theta']
            obj_vel = objects[key]['v_x']
            obj_length = objects[key].get('length', self.__veh_length)
            obj_width = objects[key].get('width', self.__veh_width)

            if objects[key]['form'] != "rectangle":
                logging.getLogger("supervisor_logger").warning('supmod_RSS | Found object (' + key + ') of form "'
                                                               + objects[key]['form'] + '" but expected "rectangle". '
                                                               'Used dimensions of ego-vehicle instead!')

            # ----------------------------------------------------------------------------------------------------------
            # - IF ACTIVATED: ESTIMATE CURRENT CURVATURE (based on spline calculation) ---------------------------------
            # ----------------------------------------------------------------------------------------------------------
            if self.__flag_curv == 1:
                if self.__obj_path[key].shape[0] < 3:
                    obj_curv = 0
                else:

                    # calculate the vector of the first two points and the last two points
                    obj_path_vec = np.stack((self.__obj_path[key][1:, 0] - self.__obj_path[key][:-1, 0],
                                             self.__obj_path[key][1:, 1] - self.__obj_path[key][:-1, 1]), axis=1)

                    # calculate the angle of the two vectors
                    obj_path_angle = np.arctan2(obj_path_vec[:, 1], obj_path_vec[:, 0]) - np.pi / 2

                    # transform the angle based on the angle convention
                    obj_path_angle[obj_path_angle < -np.pi] += 2 * np.pi

                    # calculate the coefficients of the spline
                    x_coeff, y_coeff, _, _ = tph.calc_splines.calc_splines(path=self.__obj_path[key],
                                                                           psi_s=obj_path_angle[0],
                                                                           psi_e=obj_path_angle[-1])

                    # calculate the points on the spline
                    dpoints, spln_inds, t_val, dists_int = tph.interp_splines.interp_splines(coeffs_x=x_coeff,
                                                                                             coeffs_y=y_coeff,
                                                                                             stepsize_approx=2.0,
                                                                                             incl_last_point=True)

                    # calculate the curvature of the spline
                    _, kappa = tph.calc_head_curv_an.calc_head_curv_an(coeffs_x=x_coeff,
                                                                       coeffs_y=y_coeff,
                                                                       ind_spls=spln_inds,
                                                                       t_spls=t_val)

                    # take the last value of kappa as the curvature of the last point (current curvature)
                    obj_curv = kappa[-1]
            else:
                obj_curv = 0

            # ----------------------------------------------------------------------------------------------------------
            # - OBJ-VEHICLE | FRENET FRAME TRANSFORMATION and ACCELERATION LIMITS --------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            # -- Frenet-Frame ------------------------------------------------------------------------------------------
            # position
            obj_pos_t, obj_pos_s, angle_heading_track = lane.global_2_lane_based(pos=obj_pos,
                                                                                 center_line=self.__track_centerline,
                                                                                 s_course=self.__track_s_course)

            # velocity
            obj_vel_x = obj_vel * math.sin(-obj_heading)
            obj_vel_y = obj_vel * math.cos(obj_heading)
            [obj_vel_t, obj_vel_s], obj_vel_lane_angle = lane.\
                vector_global_2_lane(vector=np.array([obj_vel_x, obj_vel_y]),
                                     angle_heading_track=angle_heading_track)

            # -- Acceleration Limits -----------------------------------------------------------------------------------
            # calculate lon. acc. of object-vehicle based on friction ellipse and limitation of the motor
            a_acc_long_max_obj, a_dec_long_max_obj = acc.acceleration_profile(a_lat_max=self.__a_long_max,
                                                                              a_long_max=self.__a_long_max,
                                                                              velocity=obj_vel,
                                                                              velocity_angle=obj_vel_lane_angle,
                                                                              curvature=obj_curv,
                                                                              acc_type='long')

            # calculate the lateral acceleration of the object based on the friction ellipse and limitation of the motor
            a_acc_left_max_obj, a_acc_right_max_obj = acc.acceleration_profile(a_lat_max=self.__a_lat_max,
                                                                               a_long_max=self.__a_long_max,
                                                                               velocity=obj_vel,
                                                                               velocity_angle=obj_vel_lane_angle,
                                                                               curvature=obj_curv,
                                                                               acc_type='lat')

            # ----------------------------------------------------------------------------------------------------------
            # - LONGITUDINAL ASSESSMENT (x-axis) -----------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------
            # initial lon. distance between the two vehicles
            # NOTE: This does not work on a closed track (crossing start line)!
            d_s_0 = abs(ego_pos_s - obj_pos_s)

            # -- brake distances ---------------------------------------------------------------------------------------
            # this boolean value is used to distinguish situations in the further code
            # NOTE: This does not work on a closed track (crossing start line)!
            obj_veh_in_front = ego_pos_s <= obj_pos_s

            # if the ego vehicle is behind the obj vehicle
            if obj_veh_in_front:
                # velocity of the ego- and obj-vehicle after accelerating within reaction time
                ego_vel_s2 = ego_vel_s + a_acc_long_max_ego * self.__t_react

                # ego distance until a full stop is reached (incl. acceleration during reaction time)
                d_s_ego_acc = (ego_vel_s * self.__t_react + 0.5 * a_acc_long_max_ego * pow(self.__t_react, 2)
                               + pow(ego_vel_s2, 2) / (2 * a_dec_long_max_ego))

                # obj distance until a full stop is reached, when immediately braking
                d_s_obj_dec = pow(obj_vel_s, 2) / (2 * a_dec_long_max_obj)

                # minimal safe longitudinal distance
                d_min_lon = max(d_s_ego_acc - d_s_obj_dec, 0.0)
            else:
                # velocity of the ego- and obj-vehicle after accelerating within reaction time
                obj_vel_s2 = obj_vel_s + a_acc_long_max_obj * self.__t_react

                # ego distance until a full stop is reached, when immediately braking
                d_s_ego_dec = pow(ego_vel_s, 2) / (2 * a_dec_long_max_ego)

                # obj distance until a full stop is reached (incl. acceleration during reaction time)
                d_s_obj_acc = (obj_vel_s * self.__t_react + 0.5 * a_acc_long_max_obj * pow(self.__t_react, 2)
                               + pow(obj_vel_s2, 2) / (2 * a_dec_long_max_obj))

                # minimal safe longitudinal distance
                d_min_lon = max(d_s_obj_acc - d_s_ego_dec, 0.0)

            # ----------------------------------------------------------------------------------------------------------
            # - LATERAL ASSESSMENT (y-axis) ----------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------
            # initial lat. distance between the two vehicles
            d_t_0 = abs(obj_pos_t - ego_pos_t)

            # check relative lateral position of obj-vehicle
            obj_veh_to_the_right = ego_pos_t >= obj_pos_t

            # NOTE - Shwartz's mu-lateral-velocity is considered to be '0' here
            # if the ego-vehicle is on the left side of the obj-vehicle
            if obj_veh_to_the_right:
                # map lateral acc. of the vehicles based on their relative position (acceleration and deceleration)
                a_acc_lat_max_ego = a_acc_right_max_ego
                a_dec_lat_max_ego = a_acc_left_max_ego
                a_acc_lat_max_obj = a_acc_left_max_obj
                a_dec_lat_max_obj = a_acc_right_max_obj

                # calculate the velocity after the reaction time
                ego_vel_t2 = ego_vel_t - a_acc_lat_max_ego * self.__t_react
                obj_vel_t2 = obj_vel_t + a_acc_lat_max_obj * self.__t_react

                # calculate the distance, which the vehicle moves, separately
                d_t_ego = ((ego_vel_t + ego_vel_t2) * self.__t_react / 2
                           - pow(ego_vel_t2, 2) / (2 * a_dec_lat_max_ego))
                d_t_obj = ((obj_vel_t + obj_vel_t2) * self.__t_react / 2
                           + pow(obj_vel_t2, 2) / (2 * a_dec_lat_max_obj))

                # minimal safe lateral distance
                d_min_lat = max(d_t_obj - d_t_ego, 0.0)
            else:
                # map lateral acc. of the vehicles based on their relative position (acceleration and deceleration)
                a_acc_lat_max_ego = a_acc_left_max_ego
                a_dec_lat_max_ego = a_acc_right_max_ego
                a_acc_lat_max_obj = a_acc_right_max_obj
                a_dec_lat_max_obj = a_acc_left_max_obj

                # calculate the velocity after the reaction time
                ego_vel_t2 = ego_vel_t + a_acc_lat_max_ego * self.__t_react
                obj_vel_t2 = obj_vel_t - a_acc_lat_max_obj * self.__t_react

                # calculate the distance, which the vehicle moves, separately
                d_t_ego = ((ego_vel_t + ego_vel_t2) * self.__t_react / 2
                           + pow(ego_vel_t2, 2) / (2 * a_dec_lat_max_ego))
                d_t_obj = ((obj_vel_t + obj_vel_t2) * self.__t_react / 2
                           - pow(obj_vel_t2, 2) / (2 * a_dec_lat_max_obj))

                # minimal safe lateral distance
                d_min_lat = max(d_t_ego - d_t_obj, 0.0)

            # ----------------------------------------------------------------------------------------------------------
            # - OVERLAP - COMBINE LON. AND LAT. EVALUATION -------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------
            eff_veh_length = self.__veh_length / 2 + obj_length / 2
            eff_veh_width = self.__veh_width / 2 + obj_width / 2

            if not obj_veh_in_front:
                # only case where obj-vehicle is in front considered -> we are in front, situation assessed to be safe
                safety_rss = 1
            else:
                safety_rss = int(not (d_s_0 - eff_veh_length < d_min_lon and d_t_0 - eff_veh_width < d_min_lat))

            # assemble safety parameters dict
            safety_parameters.update({"d_lon_cur_" + key: d_s_0 - eff_veh_length,
                                      "d_lon_bound_" + key: d_min_lon,
                                      "d_lat_cur_" + key: d_t_0 - eff_veh_width,
                                      "d_lat_bound_" + key: d_min_lat})

            # ----------------------------------------------------------------------------------------------------------
            # - TERMINAL CONDITION -------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------
            # if situation is unsafe w.r.t. one single vehicle in the scene, abort further checks and return unsafe
            if safety_rss == 0:
                break

        # --------------------------------------------------------------------------------------------------------------
        # - TEMPORAL DEBOUNCE ------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # filter out single / short mis-classification or outliers
        if safety_rss == 1:
            # safe last time stamp of 'safe' output
            self.__timer_safety_overlap = time.time()

        elif not ((time.time() - self.__timer_safety_overlap) > self.__t_debounce):
            # if time interval is smaller than the threshold --> still considered safe (else leave it at '0')
            safety_rss = 1

        return bool(safety_rss), safety_parameters
