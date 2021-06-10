import os
import numpy as np
import logging
from trajectory_supervisor.helper_funcs.src import zone_loader_generic
import trajectory_supervisor


class RuleRoboraceSeasonAlpha:
    """
    Class holding LTL activation and reduction rules for the overtaking maneuvers in the Roborace Season Alpha.
    When a vehicle closed the gap to a lead vehicle to a parameterized temporal distance, the two vehicles are forced to
    stay on a dedicated side of the track. The zones where vehicles can trigger such a maneuver or overtake are defined
    in a separate file.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>
        * Yves Huberty

    :Created on:
        22.09.2020

    """

    def __init__(self,
                 t_trigger: float) -> None:
        """
        :param t_trigger:       trigger distance [in s]

        """

        # define internal map variables
        self.__ref_line = None
        self.__s_array = None
        self.__zones_dict = {}
        self.__closed = False

        # deletion patches
        self.__del_patches = None

        # container of determined vehicle status
        self.__veh_status_dict = {'ego': VehicleState()}

        self.__t_trigger = t_trigger

    def update_map_info(self,
                        ref_line: np.ndarray,
                        s_array: np.ndarray,
                        tw_left: np.ndarray,
                        tw_right: np.ndarray,
                        normal_vectors: np.ndarray,
                        zone_file_path: str = None,
                        closed: bool = False) -> None:
        """
        Function updates the internal representation of the map and corresponding information. May be called once on
        init or multiple times in case of a moving map representation.

        :param ref_line:            reference-line used to measure s-coordinates (columns x and y [in m])
        :param s_array:             s-coordinate along the ref_line, used for definition of zones [in m]
        :param tw_left:             track-width left [in m]
        :param tw_right:            track-width right [in m]
        :param normal_vectors:      normalized normal vector based on the ref_line [in m]
        :param zone_file_path:      path pointing to a .json-zone-file (if 'None', no zone-based reduction)
        :param closed:              'True' if current track is closed circuit, 'False' else
        :return:

        """

        # store internal map variables
        self.__ref_line = ref_line
        self.__s_array = s_array
        self.__closed = closed

        # -- import overtaking zones -----------------------------------------------------------------------------------
        zones = []

        if zone_file_path is not None and os.path.isfile(zone_file_path):
            zones, pitlane, map_ref_point = zone_loader_generic. \
                zone_loader_generic(path_map=zone_file_path,
                                    raceline=np.hstack([np.reshape(s_array, (-1, 1)), self.__ref_line]))

        elif zone_file_path is None:
            logging.getLogger('supervisor_logger').info("supmod_rule_reach_sets | No zone-information provided. "
                                                        "Zone-based rules are not applied for reduction of reachable"
                                                        " sets in this run!")

        else:
            logging.getLogger('supervisor_logger').warning("supmod_rule_reach_sets | Could not load provided "
                                                           "zone-information file! Zone-based rules are not applied "
                                                           "for reduction of reachable sets!")

        # -- update track zones ----------------------------------------------------------------------------------------
        for zone in zones:
            zone_type = dict.get(zone, 'type')
            zone_start = zone['s_start']
            zone_end = dict.get(zone, 's_end')
            if zone_type == 1:
                # Trigger zone
                self.__zones_dict[str(zone['id']) + '_trigger'] = {'s_start': zone_start,
                                                                   's_end': zone_end,
                                                                   'type': 'trigger'}
            elif zone_type in [2, 3]:
                # Overtaking zone
                if zone_type == 2:
                    overtake_side = 'left'
                else:
                    overtake_side = 'right'

                self.__zones_dict[str(zone['id']) + '_overtake'] = {'s_start': zone_start,
                                                                    's_end': zone_end,
                                                                    'type': 'overtake',
                                                                    'overtake_side': overtake_side}

        # -- (re)init deletion patches ---------------------------------------------------------------------------------
        self.__del_patches = prep_deletion_patches(ref_line=ref_line,
                                                   s_array=s_array,
                                                   tw_left=tw_left,
                                                   tw_right=tw_right,
                                                   normal_vectors=normal_vectors,
                                                   zones_dict=self.__zones_dict)

    def status(self):
        """
        Returns "True" if a valid representation of roborace zones is loaded
        """

        return bool(self.__zones_dict)

    def prep_rule_eval(self,
                       pos_ego: np.ndarray,
                       vel_ego: float,
                       object_list: dict) -> None:
        """
        Prepare for rule evaluation. Especially evaluations requiring the information of all vehicles are processed in
        this step. Here, the main part is updating the vehicle poses and determining the status of the vehicles (e.g.
        overtaking).

        :param pos_ego:         position of the ego vehicle
        :param vel_ego:         velocity of the ego vehicle
        :param object_list:     object list holding all vehicles in the scene / surrounding

        """

        # if zones exist
        if self.__zones_dict:
            # check if map was initialized
            if self.__ref_line is None:
                raise ValueError("Map data must be provided before rule evaluation.")

            # -- UPDATE POS AND VEL OF VEHICLES ------------------------------------------------------------------------
            # update ego vehicle state
            self.__veh_status_dict['ego'].position = tuple(pos_ego)
            self.__veh_status_dict['ego'].velocity = vel_ego

            # check if all objects are represented in vehicle status dict and update position and vel
            for obj_key in object_list.keys():
                if obj_key not in self.__veh_status_dict:
                    self.__veh_status_dict[obj_key] = VehicleState()

                    logging.getLogger("supervisor_logger").debug("supmod_rule_reach_set | Vehicle '" + str(obj_key)
                                                                 + "' initialized!")

                self.__veh_status_dict[obj_key].position = (object_list[obj_key]['X'], object_list[obj_key]['Y'])
                self.__veh_status_dict[obj_key].velocity = object_list[obj_key]['v_x']

            # -- UPDATE STATUS OF VEHICLES (e.g overtaking) ------------------------------------------------------------
            update_status(veh_status_dict=self.__veh_status_dict,
                          ref_line=self.__ref_line,
                          s_array=self.__s_array,
                          zones_dict=self.__zones_dict,
                          t_trigger=self.__t_trigger,
                          closed=self.__closed)

    def rule_eval(self,
                  obj_key: str) -> dict:
        """
        :param obj_key:             id of the object to be evaluated
        :returns:
            * **del_patches** -     avoidance patches with keys holding relevant time-stamps and values a list of
                                    patches defined by outlining coordinates as np.ndarray with columns [x, y]
                                    NOTE: patches applicable for all time-stamps are provided with the key 'all'
                                    NOTE: temporal patches must hold the EXACT matching key as in "reach_set"

        """

        # init deletion patches
        del_patches = {'all': []}

        # if zones exist
        if self.__zones_dict:
            # if vehicle is in offence status
            if self.__veh_status_dict[obj_key].status == 1:

                # loop through all zones and extract offence side
                for del_patch in self.__del_patches.values():
                    del_patches['all'].append(del_patch['offence'])

                logging.getLogger("supervisor_logger").debug("supmod_rule_reach_set | Reduction of reach-set of "
                                                             + str(obj_key) + " for offence.")

            # if vehicle is in defence status
            if self.__veh_status_dict[obj_key].status == 2:

                # loop through all zones and extract offence side
                for del_patch in self.__del_patches.values():
                    del_patches['all'].append(del_patch['defence'])

                logging.getLogger("supervisor_logger").debug("supmod_rule_reach_set | Reduction of reach-set of "
                                                             + str(obj_key) + " defence.")

        return del_patches


def update_status(veh_status_dict: dict,
                  ref_line: np.ndarray,
                  s_array: np.ndarray,
                  zones_dict: dict,
                  t_trigger: float = 1.5,
                  closed: bool = True) -> None:
    """
    Determines the current status (neutral, offence, defence according to Roborace-rules) for all vehicles in the scene.
    Therefore, it is checked, which vehicles are within the trigger-zone and if the trigger conditions hold.

    :param veh_status_dict: dict of keys for all vehicles ('ego' and obj-keys), each holding a vehicle_state instance
    :param ref_line:        reference-line used to measure s-coordinates (columns x and y [in m])
    :param s_array:         s-coordinate along the ref_line [in m]
    :param zones_dict:      dict holding all zones along the track (key: zone-id, value: dict incl. s_start and s_end)
    :param t_trigger:       trigger distance [in s]
    :param closed:          'True' if current track is closed, 'False' else

    .. note:: Changes are written to the input variable '`veh_status_dict`'.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>
        * Yves Huberty

    :Created on:
        08.07.2020
    """

    # init list for keys of vehicles in trigger zone
    triggerzone_veh = []

    # -- DETERMINE CURRENT ZONES FOR ALL VEHICLES ----------------------------------------------------------------------
    # calculate s-coordinate for all-vehicles
    s_veh = dict()
    for veh_key in veh_status_dict.keys():
        s_veh[veh_key] = trajectory_supervisor.helper_funcs.src.path_matching.\
            get_s_coord(ref_line=ref_line,
                        pos=veh_status_dict[veh_key].position,
                        s_array=s_array,
                        closed=closed)[0]

        veh_status_dict[veh_key].s = s_veh[veh_key]

        # check if in any zone
        veh_status_dict[veh_key].zone = 0
        for zone_key in zones_dict.keys():
            if s_in_range(s=s_veh[veh_key],
                          s_start=zones_dict[zone_key]['s_start'],
                          s_end=zones_dict[zone_key]['s_end']):
                veh_status_dict[veh_key].zone = zones_dict[zone_key]['type']

                if zones_dict[zone_key]['type'] == 'trigger':
                    triggerzone_veh.append(veh_key)

    # -- DETERMINE STATUS (NEUTRAL, OFFENCE, DEFENCE) ------------------------------------------------------------------
    # for vehicles in trigger-zone: check if fulfilling trigger distance, else reset status (status only changed there)
    # NOTE: according to regulations all vehicles fulfilling distance to next vehicle ahead will get offence status,
    #       if multiple vehicles fulfill this, then the first vehicle is the only with defence status
    # NOTE: trigger distance is assumed NOT to overlap start-finish line

    # for each vehicle in zone
    for veh_key in triggerzone_veh:

        logging.getLogger("supervisor_logger").debug("supmod_rule_reach_set | Vehicle '" + veh_key + "' in trigger zone"
                                                                                                     ".")

        # find closest vehicle ahead and behind
        closest_lead = None
        closest_follow = None

        for veh_key2 in veh_status_dict.keys():
            if veh_key2 != veh_key:
                if s_veh[veh_key] < s_veh[veh_key2]:
                    # if veh_key is behind veh_key2
                    if closest_lead is None or s_veh[veh_key2] - s_veh[veh_key] < s_veh[closest_lead] - s_veh[veh_key]:
                        # if no lead determined yet, or new lead is closer than currently stored
                        closest_lead = veh_key2
                else:
                    # if veh_key is in front of veh_key2
                    if (closest_follow is None
                            or s_veh[veh_key] - s_veh[veh_key2] < s_veh[veh_key] - s_veh[closest_follow]):
                        # if no follow determined yet, or new follow is closer than currently stored
                        closest_follow = veh_key2

        # check trigger-distance for closest vehicles
        triggered_lead = False
        if closest_lead is not None:
            triggered_lead = trigger_dist_roborace(s_lead=s_veh[closest_lead],
                                                   s_chase=s_veh[veh_key],
                                                   v_chase=veh_status_dict[veh_key].velocity,
                                                   t_trigger=t_trigger)

        triggered_follow = False
        if closest_follow is not None:
            triggered_follow = trigger_dist_roborace(s_lead=s_veh[veh_key],
                                                     s_chase=s_veh[closest_follow],
                                                     v_chase=veh_status_dict[closest_follow].velocity,
                                                     t_trigger=t_trigger)

        # set status accordingly
        if triggered_lead:
            # if triggered lead --> status offence (1)
            veh_status_dict[veh_key].status = 1

            logging.getLogger("supervisor_logger").info("supmod_rule_reach_set | Vehicle '" + veh_key + "' triggered "
                                                        "lead vehicle in trigger zone --> offence status!")

        elif triggered_follow:
            # if triggered by follow (but not triggered lead) --> status defence (2)
            veh_status_dict[veh_key].status = 2

            logging.getLogger("supervisor_logger").info("supmod_rule_reach_set | Vehicle '" + veh_key + "' triggered "
                                                        "by follow vehicle in trigger zone --> defence status!")

        else:
            # else: neutral (0)
            veh_status_dict[veh_key].status = 0

            logging.getLogger("supervisor_logger").info("supmod_rule_reach_set | Vehicle '" + veh_key + "' in trigger "
                                                        "zone, but no vehicle in reach --> neutral status!")


def prep_deletion_patches(ref_line: np.ndarray,
                          s_array: np.ndarray,
                          tw_left: np.ndarray,
                          tw_right: np.ndarray,
                          normal_vectors: np.ndarray,
                          zones_dict: dict) -> dict:
    """
    Prepare deletion patches once. Based on the track and zone specification, patches to be avoided are generated.

    :param ref_line:            reference-line used to measure s-coordinates (columns x and y [in m])
    :param s_array:             s-coordinate along the ref_line [in m]
    :param tw_left:             track-width left [in m]
    :param tw_right:            track-width right [in m]
    :param normal_vectors:      normalized normal vector based on the ref_line [in m]
    :param zones_dict:          dict holding zones along the track (key: zone-id, value: dict incl. s_start and s_end)
    :returns:
        * **del_patches** -     returned avoidance patches for offending and defending car, dict with zone keys, where
          each entry holds an 'offence' and 'defence' key with matching coordinates (col x & y)

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>
        * Yves Huberty

    :Created on:
        08.07.2020
    """

    del_patches = dict()

    # for every zone
    for zone_key in zones_dict.keys():
        if zones_dict[zone_key]['type'] == 'overtake':
            # calculate deletion patch for left and right track side
            # NOTE: currently the patch is drawn 50m to the side of the centerline --> might cause issues on sharp turns

            # calculate centerline
            bound_left = ref_line - normal_vectors * tw_left
            bound_right = ref_line + normal_vectors * tw_right

            centerline = (bound_right + bound_left) / 2

            # get index in ref_line / normal_vectors for given zone
            idx_start_tmp = (np.abs(s_array - zones_dict[zone_key]['s_start'])).argmin()

            # roll all arrays (start index is first element (in order to tackle closed tracks properly)
            centerline_r = np.roll(centerline, -idx_start_tmp, axis=0)
            normal_vectors_r = np.roll(normal_vectors, -idx_start_tmp, axis=0)
            s_array_r = np.roll(s_array, -idx_start_tmp, axis=0)

            idx_end = (np.abs(s_array_r - zones_dict[zone_key]['s_end'])).argmin()

            left_patch = np.vstack((np.flipud(centerline_r[0:idx_end, :]),
                                    centerline_r[0:idx_end, :] - normal_vectors_r[0:idx_end, :] * 50.0))

            right_patch = np.vstack((np.flipud(centerline_r[0:idx_end, :]),
                                    centerline_r[0:idx_end, :] + normal_vectors_r[0:idx_end, :] * 50.0))

            # depending on
            if zones_dict[zone_key]['overtake_side'] == 'left':
                del_patches[zone_key] = {'offence': right_patch,
                                         'defence': left_patch}
            elif zones_dict[zone_key]['overtake_side'] == 'right':
                del_patches[zone_key] = {'offence': left_patch,
                                         'defence': right_patch}
            else:
                raise ValueError("Unsupported 'overtake_side' provided (" + zones_dict[zone_key]['overtake_side']
                                 + ")!")

            logging.getLogger("supervisor_logger").info("supmod_rule_reach_set | Finished generation of rule-based"
                                                        "deletion patches.")

    return del_patches


def trigger_dist_roborace(s_lead: float,
                          s_chase: float,
                          v_chase: float,
                          t_trigger: float) -> bool:
    """
    This function calculates the trigger distance as defined by the Roborace overtaking regulations. This method of
    calculation is only an approximation. The distance between the vehicles is calculated along the reference line which
    is not the center line of the track!

    :param s_lead:           s coordinate of the lead car
    :param s_chase:          s coordinate of the chase car
    :param v_chase:          velocity of the chase car
    :param t_trigger:        predefined minimum safety parameter by Roborace
    :returns:
        * **trigger** -      boolean value if safe distance is respected or not

    :Authors:
        * Yves Huberty

    :Created on:
        17.04.2020
    """
    # Check if the lead car is in front of the chase car
    if s_lead - s_chase < 0:
        raise ValueError('s_lead < s_chase -> the lead car is behind the chase car!')
    else:
        delta_t = (s_lead - s_chase) / v_chase

        # Check if delta_t is greater or smaller than t_safe
        if delta_t < t_trigger:
            trigger = True
        else:
            trigger = False

    return trigger


def s_in_range(s: float,
               s_start: float,
               s_end: float) -> bool:
    """
    Checks if coordinate s is in between s_start and s_end. If s_end < s_start, a closed track is assumed.

    :param s:           s-value of interest
    :param s_start:     start s-value of range
    :param s_end:       end s-value of range
    :returns:
        * **rating** -  returns "True" if 's' in range, else "False"

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        08.07.2020
    """

    if s_start < s_end:
        return s_start <= s <= s_end

    else:
        return s <= s_end or s_start <= s


class VehicleState:
    """
    This class stores vehicle states regarding overtaking maneuvers for a vehicle.

        * position:       x, y coordinate in m
        * velocity:       vx in m/s
        * s:              s coordinate in m
        * zone
            * 0 = no zone
            * 1 = trigger zone
            * 2 = overtaking zone
        * status
            * 0 = neutral
            * 1 = offence
            * 2 = defence

    :Authors:
        * Yves Huberty
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        10.04.2020
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, pos=(0.0, 0.0), vel=0.0, s=0.0, zone=0, status=0):
        # generic properties
        self.__position = pos
        self.__velocity = vel
        self.__s = s

        # roborace race rules states
        self.__zone = zone
        self.__status = status

    # ------------------------------------------------------------------------------------------------------------------
    # DESTRUCTOR -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __del__(self):
        pass

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, pos):
        self.__position = pos

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self, vel):
        self.__velocity = vel

    @property
    def s(self):
        return self.__s

    @s.setter
    def s(self, s):
        if s < 0.0:
            raise ValueError('Negative s-coordinate not allowed!')

        self.__s = s

    @property
    def zone(self):
        return self.__zone

    @zone.setter
    def zone(self, zone):
        self.__zone = zone

    @property
    def status(self):
        return self.__status

    @status.setter
    def status(self, stat):
        if stat < 0:
            raise ValueError('Vehicle status out of range (< 0)!')
        elif stat > 2:
            raise ValueError('Vehicle status out of range (> 2)!')

        self.__status = stat
