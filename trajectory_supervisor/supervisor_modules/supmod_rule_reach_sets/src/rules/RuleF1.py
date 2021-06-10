import numpy as np
import trajectory_planning_helpers as tth
import trajectory_supervisor


class RuleF1:
    """
    Class holding LTL activation and reduction rules for the overtaking maneuvers in the Formula 1 motorsport.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        28.09.2020

    """

    def __init__(self,
                 veh_length: float,
                 veh_width: float,
                 dict_config_alongside: dict,
                 dict_config_defense: dict) -> None:
        """
        :param veh_length:              vehicle length [in m]
        :param veh_width:               vehicle width [in m]
        :param dict_config_alongside    dictionary holding relevant parameters for the "alongside" rule
        :param dict_config_defense      dictionary holding relevant parameters for the "defensd off-line" rule

        """

        # define internal map variables
        self.__ref_line = None
        self.__s_array = None
        self.__tw_left = None
        self.__tw_right = None
        self.__normal_vectors = None
        self.__closed = False

        # vehicle parameters
        self.__veh_length = veh_length
        self.__veh_width = veh_width

        # corner data
        self.__s_corner = None
        self.__left_corner = None

        # rule memory
        self.__rule_memory = dict()

        # configuration parameters
        self.__config_alongside = dict_config_alongside
        self.__config_defense = dict_config_defense

    def update_map_info(self,
                        ref_line: np.ndarray,
                        s_array: np.ndarray,
                        tw_left: np.ndarray,
                        tw_right: np.ndarray,
                        normal_vectors: np.ndarray,
                        closed: bool = False) -> None:
        """
        Function updates the internal representation of the map and corresponding information. May be called once on
        init or multiple times in case of a moving map representation.

        :param ref_line:            reference-line used to measure s-coordinates (columns x and y [in m])
        :param s_array:             s-coordinate along the ref_line, used for definition of zones [in m]
        :param tw_left:             track-width left [in m]
        :param tw_right:            track-width right [in m]
        :param normal_vectors:      normalized normal vector based on the ref_line [in m]
        :param closed:              'True' if current track is closed circuit, 'False' else
        :return:

        """

        # -- store internal map variables ------------------------------------------------------------------------------
        self.__ref_line = ref_line
        self.__s_array = s_array
        self.__tw_left = tw_left
        self.__tw_right = tw_right
        self.__normal_vectors = normal_vectors
        self.__closed = closed

        # -- detect track corners --------------------------------------------------------------------------------------
        el_length = np.diff(s_array)
        if closed:
            el_length = np.concatenate((np.diff(s_array), [np.hypot(ref_line[0, 0] - ref_line[-1, 0],
                                                                    ref_line[0, 1] - ref_line[-1, 0])]))

        curv = tth.calc_head_curv_num.calc_head_curv_num(path=ref_line,
                                                         el_lengths=el_length,
                                                         is_closed=closed)[1]

        self.__s_corner = s_array[np.abs(curv) > self.__config_defense['kappa_thr']]
        self.__left_corner = curv[np.abs(curv) > self.__config_defense['kappa_thr']] > 0.0

    def rule_eval(self,
                  ego_pos: np.ndarray,
                  ego_s_pos: float,
                  obj_pos: np.ndarray,
                  obj_s_pos: float,
                  obj_key: str = 'default') -> dict:
        """
        Evaluate LTL-rules and trigger calculation of patches, if conditions are fulfilled.

        :param ego_pos:             position of the ego vehicle
        :param ego_s_pos:           s-coordinate of ego vehicle
        :param obj_pos:             position of the object vehicle
        :param obj_s_pos:           s-coordinate of object vehicle
        :param obj_key:             object key, if multiple vehicle are present - required for past memory
        :returns:
            * **del_patches** -     avoidance patches with keys holding relevant time-stamps and values a list of
                                    patches defined by outlining coordinates as np.ndarray with columns [x, y]
                                    NOTE: patches applicable for all time-stamps are provided with the key 'all'
                                    NOTE: temporal patches must hold the EXACT matching key as in "reach_set"

        """

        # check if map was initialized
        if self.__ref_line is None:
            raise ValueError("Map data must be provided before rule evaluation.")

        # check if provided object is already present in memory, else init
        if self.__rule_memory.get(obj_key, None) is None:
            self.__rule_memory[obj_key] = dict()

        # init deletion patches
        del_patches = {'all': []}

        # handle closed tracks (since vehicles behind the ego are not considered --> project them to the next lap)
        if obj_s_pos < ego_s_pos and self.__closed:
            obj_s_pos += self.__s_array[-1]

        # check if approaching corner (required by multiple rules)
        nxt_corner_idx = 0
        approach_corner = False
        if len(self.__s_corner) > 0:
            nxt_corner_idx = np.argmax(self.__s_corner >= obj_s_pos)
            approach_corner = in_front_of_s_coord(target_s=self.__s_corner[nxt_corner_idx],
                                                  target_range=self.__config_defense['d_corner'],
                                                  current_s=obj_s_pos,
                                                  s_closed_max=(self.__s_array[-1] if self.__closed else None))

        # -- CHECK IF EGO VEHICLE IS IN CERTAIN CONSTELLATION WITH OTHER VEHICLE (LTL-RULES) ---------------------------

        # racing alongside another car ("3." in https://f1metrics.wordpress.com/2014/08/28/the-rules-of-racing/)
        #  "20.4 Any driver defending his position on a straight, and before any braking area, may use the full width of
        #   the track during his first move, provided no significant portion of the car attempting to pass is alongside
        #   his. For the avoidance of doubt, if any part of the front wing of the car attempting to pass is alongside
        #   the rear wheel of the car in front this will be deemed to be a 'significant portion'."

        # if not approaching corner or already alongside before approaching corner and overlapping vehicles
        if ((not approach_corner or (approach_corner and self.__rule_memory[obj_key].get('alongside', False)))
                and (obj_s_pos - ego_s_pos) < (1 - self.__config_alongside['d_overlap']) * self.__veh_length):
            # -- CALCULATE DELETION PATCH FOR THIS TYPE ----------------------------------------------------------------
            # (ego's side of track, divided by current lateral displacement on track)
            del_patches['all'].append(self.__patch_alongside(ego_pos=ego_pos,
                                                             ego_s_pos=ego_s_pos,
                                                             obj_pos=obj_pos,
                                                             obj_s_pos=obj_s_pos))
            self.__rule_memory[obj_key]['alongside'] = True

        else:
            self.__rule_memory[obj_key]['alongside'] = False

        # defending off-line
        # "Any driver moving back towards the racing line, having earlier defended his position off-line, should leave
        # at least one car width between his own car and the edge of the track on the approach to the corner" Apdx. L 2b
        # NOTE: if combination of curves (closer together than d_corner), apply rule only in first corner of the comb.
        if not self.__rule_memory[obj_key]['alongside'] and approach_corner and\
                not in_front_of_s_coord(target_s=self.__s_corner[nxt_corner_idx],
                                        target_range=self.__config_defense['d_corner'],
                                        current_s=self.__s_corner[nxt_corner_idx - 1],
                                        s_closed_max=(self.__s_array[-1] if self.__closed else None)):
            # if not once in defense off-line on this corner approach
            if not self.__rule_memory[obj_key].get('defense_on_corner_approach', False):
                # check if in defense
                defense = in_front_of_s_coord(target_s=obj_s_pos,
                                              target_range=self.__config_defense['d_defend'],
                                              current_s=ego_s_pos)

                # check if off-line (here on off-raceline track half, i.e. corner inside)
                bound_pos = bound_points_at_pos(ref_line_track=self.__ref_line,
                                                norm_vecs_track=self.__normal_vectors,
                                                tw_left_track=self.__tw_left,
                                                tw_right_track=self.__tw_right,
                                                s_array_track=self.__s_array,
                                                closed_track=self.__closed,
                                                s_pos=obj_s_pos,
                                                pos=obj_pos)

                track_width = np.hypot(bound_pos[0][0] - bound_pos[1][0], bound_pos[0][1] - bound_pos[1][1])
                left_width = np.hypot(bound_pos[0][0] - obj_pos[0], bound_pos[0][1] - obj_pos[1])
                left_half = (track_width / left_width) > 2.0

                # if on left side and left corner or right side and right corner (XNOR)
                off_line = not (left_half ^ self.__left_corner[nxt_corner_idx])

                # if defense and offline --> update defense on corner approach
                self.__rule_memory[obj_key]['defense_on_corner_approach'] = defense and off_line

            else:
                # -- CALCULATE DELETION PATCH FOR THIS TYPE --------------------------------------------------------
                # (ego's side of track, divided by current lateral displacement on track)
                del_patches['all'].append(self.__patch_defend(ego_s_pos=ego_s_pos,
                                                              left_bound=not self.__left_corner[nxt_corner_idx]))

        else:
            # reset defense on corner approach, if not approaching a corner
            self.__rule_memory[obj_key]['defense_on_corner_approach'] = False

        return del_patches

    def __patch_alongside(self,
                          ego_pos: np.ndarray,
                          ego_s_pos: float,
                          obj_pos: np.ndarray,
                          obj_s_pos: float) -> np.ndarray:
        """
        Calculate the patch for the "alongside" F1 racing rule
        ("3." in https://f1metrics.wordpress.com/2014/08/28/the-rules-of-racing/)

        :param ego_pos:             position of the ego vehicle
        :param ego_s_pos:           s-coordinate of ego vehicle
        :param obj_pos:             position of the object vehicle
        :param ego_s_pos:           s-coordinate of object vehicle
        :returns:
            * **patch_coords** -    numpy array holding patch coordinates with columns x, y

        """

        # -- GET LATERAL DISPLACEMENT ON TRACK (RELATIVE TO BOUND) -----------------------------------------------------
        # calculate bound positions next to ego-position (index '1' for right bound, in order to align 'right_bound')
        bound_pos = bound_points_at_pos(ref_line_track=self.__ref_line,
                                        norm_vecs_track=self.__normal_vectors,
                                        tw_left_track=self.__tw_left,
                                        tw_right_track=self.__tw_right,
                                        s_array_track=self.__s_array,
                                        closed_track=self.__closed,
                                        s_pos=ego_s_pos,
                                        pos=ego_pos)

        bound_pos_o = bound_points_at_pos(ref_line_track=self.__ref_line,
                                          norm_vecs_track=self.__normal_vectors,
                                          tw_left_track=self.__tw_left,
                                          tw_right_track=self.__tw_right,
                                          s_array_track=self.__s_array,
                                          closed_track=self.__closed,
                                          s_pos=obj_s_pos,
                                          pos=obj_pos)

        # get side of other vehicle via direction of normal vector (retrieve bound of opposite side)
        # -> calculate angle between vector from ego to obj and normal vector
        ang_offset = trajectory_supervisor.helper_funcs.src.path_matching.\
            angle3pt(a=self.__normal_vectors[bound_pos[2], :],
                     b=[0.0, 0.0],
                     c=ego_pos - obj_pos)

        # if relevant bound is on side where normal vector points (right_bound = True); False else
        right_bound = bool(-np.pi / 2 < ang_offset < np.pi / 2)

        # calculate lateral distance from ego vehicle and object vehicle to bound
        d_lat_ego_wall = np.hypot(ego_pos[0] - bound_pos[right_bound][0], ego_pos[1] - bound_pos[right_bound][1])
        d_lat_obj_wall = np.hypot(obj_pos[0] - bound_pos_o[right_bound][0], obj_pos[1] - bound_pos_o[right_bound][1])

        # select middle point between two vehicles
        d_lat_wall = (d_lat_ego_wall + d_lat_obj_wall) / 2

        # effective lateral displacement to wall (minimum of object- and ego-vehicle's relevant edges)
        d_lat_wall = min(d_lat_wall,
                         d_lat_obj_wall - self.__veh_width / 2)

        # calculate percentage of track width
        track_width = np.hypot(bound_pos[0][0] - bound_pos[1][0], bound_pos[0][1] - bound_pos[1][1])
        perc_lat = d_lat_wall / track_width

        # -- CALCULATE BOUNDS OF DELETION PATCH ------------------------------------------------------------------------
        # (calculate bound points, until the stop condition - covered longitudinal distance - is met

        # init variables (including bound coordinate arrays)
        len_tmp = 0.0
        idx_tmp = bound_pos[3]
        prv_pos = bound_pos[right_bound]

        ptch_bound_coords = [bound_pos[right_bound]]
        if right_bound:
            ptch_track_coords = [np.array(bound_pos[right_bound]) - self.__normal_vectors[bound_pos[2], :] * d_lat_wall]
        else:
            ptch_track_coords = [np.array(bound_pos[right_bound]) + self.__normal_vectors[bound_pos[2], :] * d_lat_wall]

        # loop until length is reached
        while len_tmp < self.__config_alongside['len_patch']:
            # check if idx_tmp is valid
            if idx_tmp > self.__ref_line.shape[0] - 1:
                if self.__closed:
                    idx_tmp = 0

                else:
                    break

            # calculate next bound position along relevant bound (left / right)
            if right_bound:
                nxt_pos = self.__ref_line[idx_tmp, :] + self.__normal_vectors[idx_tmp, :] * self.__tw_right[idx_tmp]
            else:
                nxt_pos = self.__ref_line[idx_tmp, :] - self.__normal_vectors[idx_tmp, :] * self.__tw_left[idx_tmp]

            # calculate distance to next point (in order to track, if required length is reached)
            len_tmp += np.hypot(prv_pos[0] - nxt_pos[0], prv_pos[1] - nxt_pos[1])

            # calculate new offset on track (based on percentage), while ensuring that:
            # 1. ego vehicle is not pushed of track - max(...)
            # 2. object vehicle must not leave track - min(...) [prioritized, if bot conditions can not be met]
            track_width = self.__tw_right[idx_tmp] + self.__tw_left[idx_tmp]
            d_lat_wall = min(max(track_width * perc_lat, min(d_lat_wall, self.__veh_width)),
                             track_width - self.__veh_width)

            # append patch bound coordinates (fixed lateral displacement to wall)
            if right_bound:
                ptch_bound_coords.append(nxt_pos + self.__normal_vectors[idx_tmp, :]
                                         * self.__config_alongside['width_extra_bound'])

                ptch_track_coords.append(self.__ref_line[idx_tmp, :] + self.__normal_vectors[idx_tmp, :]
                                         * (self.__tw_right[idx_tmp] - d_lat_wall))
            else:
                ptch_bound_coords.append(nxt_pos - self.__normal_vectors[idx_tmp, :]
                                         * self.__config_alongside['width_extra_bound'])
                ptch_track_coords.append(self.__ref_line[idx_tmp, :] - self.__normal_vectors[idx_tmp, :]
                                         * (self.__tw_left[idx_tmp] - d_lat_wall))

            prv_pos = np.copy(nxt_pos)
            idx_tmp += 1

        # -- CONVERT EXTRACTED SHAPE BOUNDS TO SHAPELY PATCH -----------------------------------------------------------
        patch_coords = np.vstack((np.flipud(np.row_stack(ptch_bound_coords)), np.row_stack(ptch_track_coords)))

        return patch_coords

    def __patch_defend(self,
                       ego_s_pos: float,
                       left_bound: bool) -> np.ndarray:
        """
        Calculate the patch for the "defense off-line" F1 racing rule

        :param ego_s_pos:           s-coordinate of object vehicle
        :param left_bound:          True, if the deletion patch should be generated on the left bound
        :returns:
            * **patch_coords** -    numpy array holding patch coordinates with columns x, y

        """

        # determine start index (at s-value before ego_s_pos)
        idx_tmp = max(np.argmax(self.__s_array > ego_s_pos) - 1, 0)

        # -- CALCULATE BOUNDS OF DELETION PATCH ------------------------------------------------------------------------
        # (calculate bound points, until the stop condition - covered longitudinal distance - is met

        # init variables (including bound coordinate arrays)
        len_tmp = 0.0
        ptch_bound_coords = []
        ptch_track_coords = []
        prv_pos = None

        # loop until length is reached
        while len_tmp < self.__config_defense['len_patch']:
            # check if idx_tmp is valid
            if idx_tmp > self.__ref_line.shape[0] - 1:
                if self.__closed:
                    idx_tmp = 0

                else:
                    break

            # calculate next bound position along relevant bound (left / right)
            if left_bound:
                nxt_pos = self.__ref_line[idx_tmp, :] - self.__normal_vectors[idx_tmp, :] * self.__tw_left[idx_tmp]

                ptch_bound_coords.append(nxt_pos - self.__normal_vectors[idx_tmp, :]
                                         * self.__config_defense['width_extra_bound'])
                ptch_track_coords.append(self.__ref_line[idx_tmp, :] - self.__normal_vectors[idx_tmp, :]
                                         * (self.__tw_left[idx_tmp]
                                            - self.__veh_width * self.__config_defense['veh_width_factor']))

            else:
                nxt_pos = self.__ref_line[idx_tmp, :] + self.__normal_vectors[idx_tmp, :] * self.__tw_right[idx_tmp]

                ptch_bound_coords.append(nxt_pos + self.__normal_vectors[idx_tmp, :]
                                         * self.__config_defense['width_extra_bound'])

                ptch_track_coords.append(self.__ref_line[idx_tmp, :] + self.__normal_vectors[idx_tmp, :]
                                         * (self.__tw_right[idx_tmp]
                                            - self.__veh_width * self.__config_defense['veh_width_factor']))

            # calculate distance to next point (in order to track, if required length is reached)
            if prv_pos is not None:
                len_tmp += np.hypot(prv_pos[0] - nxt_pos[0], prv_pos[1] - nxt_pos[1])

            prv_pos = np.copy(nxt_pos)
            idx_tmp += 1

        # -- CONVERT EXTRACTED SHAPE BOUNDS TO SHAPELY PATCH -----------------------------------------------------------
        patch_coords = np.vstack((np.flipud(np.row_stack(ptch_bound_coords)), np.row_stack(ptch_track_coords)))

        return patch_coords


def bound_points_at_pos(ref_line_track: np.ndarray,
                        norm_vecs_track: np.ndarray,
                        tw_left_track: np.ndarray,
                        tw_right_track: np.ndarray,
                        s_array_track: np.ndarray,
                        closed_track: bool,
                        s_pos: float,
                        pos: np.ndarray) -> tuple:
    """
    Calculate bound points to the left / right of the provided position.

    :param ref_line_track:      reference-line used to measure s-coordinates (columns x and y [in m])
    :param norm_vecs_track:     normalized normal vector based on the ref_line [in m]
    :param tw_left_track:       track-width left [in m]
    :param tw_right_track:      track-width right [in m]
    :param s_array_track:       s-coordinate along the ref_line, used for definition of zones [in m]
    :param closed_track:        'True' if current track is closed circuit, 'False' else
    :param s_pos:               s-coordinate of the provided position [in m]
    :param pos:                 coordinate of interest
    :returns:
        * **pos_l** -           position on the left bound, closest to "pos"
        * **pos_r** -           position on the right bound, closest to "pos"
        * **idx1** -            index within refline (and associated values), that lies before 'pos'
        * **idx2** -            index within refline (and associated values), that lies after 'pos'
    """

    # get coordinate _before_ provided s-coordinate (reference line, normal vector, bounds)
    s_diff = np.array(s_array_track - s_pos)
    idx_nb = int(np.argmax(s_diff[s_diff < 0]))

    # get index pair (before and behind provided position)
    if closed_track:
        idx1 = idx_nb
        idx2 = idx_nb + 1
        if idx2 > (ref_line_track.shape[0] - 1):
            idx2 = 0
    else:
        idx1 = idx_nb
        idx2 = min(idx_nb + 1, ref_line_track.shape[0] - 1)

    # calculate track bound coordinate to the right and left of given position
    # left bound
    pos_l = perpendicular_point(a_pos=ref_line_track[idx1, :] - norm_vecs_track[idx1, :] * tw_left_track[idx1],
                                b_pos=ref_line_track[idx2, :] - norm_vecs_track[idx2, :] * tw_left_track[idx2],
                                c_pos=list(pos))

    # right bound
    pos_r = perpendicular_point(a_pos=ref_line_track[idx1, :] + norm_vecs_track[idx1, :] * tw_right_track[idx1],
                                b_pos=ref_line_track[idx2, :] + norm_vecs_track[idx2, :] * tw_right_track[idx2],
                                c_pos=list(pos))

    return pos_l, pos_r, idx1, idx2


def perpendicular_point(a_pos: list,
                        b_pos: list,
                        c_pos: list) -> list:
    """
    Get point projected from c_pos perpendicular on the line passing the points a_pos, b_pos.

    :param a_pos:           position a
    :param b_pos:           position b
    :param c_pos:           position c
    :returns:
        * **line_pos** -    position resulting when c_pos is projected perpendicular on line passing a_pos and b_pos

    """

    # https://stackoverflow.com/questions/10301001/perpendicular-on-a-line-segment-from-a-given-point
    numerator = (np.power(b_pos[0] - a_pos[0], 2) + np.power(b_pos[1] - a_pos[1], 2))
    if numerator > 0.1:
        t = ((c_pos[0] - a_pos[0]) * (b_pos[0] - a_pos[0]) + (c_pos[1] - a_pos[1]) * (b_pos[1] - a_pos[1])) / numerator
    else:
        t = 0.0

    return [a_pos[0] + t * (b_pos[0] - a_pos[0]), a_pos[1] + t * (b_pos[1] - a_pos[1])]


def in_front_of_s_coord(target_s: float,
                        target_range: float,
                        current_s: float,
                        s_closed_max: float = None) -> bool:
    """
    Check, whether the current s-coordinate (current_s) is within the target range (target_range) in front (i.e. smaller
    s-coordinate) of the target s-coordinate (target_s). If the track is closed, the maximum s-coordinate (s_closed_max)
    must be provided.

    :param target_s:        target s-coordinate
    :param target_range:    range in front of target s-coordinate to be rated as valid
    :param current_s:       current s-coordinate
    :param s_closed_max:    maximum s-coordinate for closed tracks
    :returns:
        * **in_range** -    True, if the current s-coordinate falls within the specified range

    """

    if s_closed_max is None or (target_s - target_range) > 0:
        in_range = ((target_s - target_range) <= current_s <= target_s)

    else:
        current_s_closed = current_s
        if 0 <= current_s <= target_s:
            current_s_closed += s_closed_max

        in_range = ((target_s - target_range + s_closed_max) <= current_s <= target_s + s_closed_max)

    return in_range
