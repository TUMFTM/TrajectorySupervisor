import os
import glob
import json
import sys
import numpy as np
import copy
import datetime
import PlotHandler
import SafetyParamInspector
import plot_timing_histogram
import scenario_testing_tools as stt

"""
Script which allows to visualize safety logs recorded during a real or simulated run. In order to visualize a log, call
this script with the path to a log file as argument. When the run was executed locally, a call of the script without any
argument will display the latest run.

:Authors:
    * Tim Stahl <tim.stahl@tum.de>

:Created on:
    24.01.2019
"""

# -- PARAMETERS FOR TEMPORAL STAMP VISUALIZATION (DOCUMENTATION PLOTS) -------------------------------------------------
# if this parameter is set to "True" vehicle stamps are generated in a configured way along the logged trajectories
PLOT_VEH = False
PLOT_VEH_PARAM = {"temporal_increment": 1.0,
                  "plot_text": True,
                  "alpha": 0.7,
                  "zorder": 8,
                  "plot_text_distance": 10.0,
                  "plot_text_every_ith_element": 2,
                  "initial_t_offset": 0.0}


def get_data_from_line(_file_path: str,
                       _line_num: int) -> tuple:
    """
    Retrieve data in a certain line ('`line_num`') from a log-file ('`file_path_in`').

    :param _file_path:      string holding path to the log-file
    :param _line_num:       line number from which the data should be retrieved
    :returns:
        * **ego_traj** -    retrieved ego-trajectory
        * **objects** -     retrieved objects
        * **safety_base** - retrieved 'safety_base' (holding safety score and further relevant parameters for the score)

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        24.01.2019

    """

    # skip header
    _line_num = _line_num + 1

    # extract a certain line number (based on time_stamp)
    with open(_file_path) as _file:
        # get to top of file (1st line)
        _file.seek(0)

        # handle comments
        _line = _file.readline()
        while _line[0] == '#':
            _line = _file.readline()

        # get header (":-1" in order to remove tailing newline character)
        _header = _line[:-1]

        # extract line
        _line = ""
        for _ in range(_line_num):
            _line = _file.readline()

        # parse the data objects we want to retrieve from that line
        _data = dict(zip(_header.split(";"), _line.split(";")))

        # decode
        # time;traj_stamp;traj_perf_ref;traj_perf;traj_emerg_ref;traj_emerg;objects_stamp;objects_ref;objects;
        # safety_static;safety_dynamic;safety_base
        ego_traj = {}

        for traj_type in ['perf', 'emerg']:
            ego_traj[traj_type] = {'stamp': json.loads(_data['traj_stamp']),
                                   'data_intern': np.array(json.loads(_data['traj_' + traj_type]))}

            # add reference data
            if json.loads(_data['traj_' + traj_type + '_ref']) is not None:
                ego_traj[traj_type]['data_ref'] = np.array(json.loads(_data['traj_' + traj_type + '_ref']))
            else:
                ego_traj[traj_type]['data_ref'] = np.copy(ego_traj[traj_type]['data_intern'])

        objects = {'stamp': json.loads(_data['objects_stamp']),
                   'data_intern': json.loads(_data['objects'])}

        # add reference data
        if json.loads(_data['objects_ref']) is not None:
            objects['data_ref'] = json.loads(_data['objects_ref'])
        else:
            objects['data_ref'] = json.loads(_data['objects'])

    return ego_traj, objects, json.loads(_data['safety_base'])


def get_map_from_line(_file_path: str,
                      _line_num: int) -> tuple:
    """
    Retrieve the map from a safety-log-file.

    :param _file_path:              string holding path to the log-file
    :param _line_num:                line number from which the data should be retrieved
    :returns:
        * **bound_l** -             left bound coordinates
        * **bound_r** -             right bound coordinates
        * **localgg** -             local gg diagram (if present in file, else "None")
        * **ax_max_mach** -         maximum acceleration potential of the machine (if present in file, else "None")
        * **acc_limit_factor** -    acceleration limit factor [1.0, if localgg is used as is, without safety factor]

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        24.01.2019

    """

    # skip header
    _line_num = _line_num + 1

    # extract a certain line number (based on time_stamp)
    with open(_file_path) as _file:
        # get to top of file (1st line)
        _file.seek(0)

        # get header (":-1" in order to remove tailing newline character)
        _header = _file.readline()[:-1]

        # extract line
        _line = ""
        for _ in range(_line_num):
            _line = _file.readline()

        # parse the data objects we want to retrieve from that line
        map_data = dict(zip(_header.split(";"), _line.split(";")))

        # decode
        # time;bound_l;bound_r;localgg;ax_max_machines
        _bound_l = np.array(json.loads(map_data['bound_l']))
        _bound_r = np.array(json.loads(map_data['bound_r']))

        if 'localgg' in map_data.keys():
            _localgg = np.array(json.loads(map_data['localgg']))
            _ax_max_machines = np.array(json.loads(map_data['ax_max_machines']))
        else:
            _localgg = None
            _ax_max_machines = None

        if 'acc_limit_factor' in map_data.keys():
            _acc_limit_factor = np.array(json.loads(map_data['acc_limit_factor']))
        else:
            _acc_limit_factor = 1.0

    return _bound_l, _bound_r, _localgg, _ax_max_machines, _acc_limit_factor


class DebugHandler(object):
    """
    Class that interfaces the plot-handler for the visualization of a safety-log-file. The mouse-pointer is continuously
    tracked and whenever it is moved on top of the timeline plot, all information is updated according to the selected
    temporal timestamp.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        24.01.2019

    """

    def __init__(self,
                 _time_stamps_data: list,
                 _time_stamps_msgs: list,
                 _time_stamps_map: list,
                 _time_msgs_types: list,
                 _time_msgs_content: list,
                 _veh_params: dict = None) -> None:

        self.__n_store = None
        self.__type_store = None
        self.__working = False

        self.__time_stamps_data = _time_stamps_data
        self.__time_stamps_msgs = _time_stamps_msgs
        self.__time_stamps_map = _time_stamps_map
        self.__time_msgs_types = _time_msgs_types
        self.__time_msgs_content = _time_msgs_content

        self.__active_map_stamp = _time_stamps_map[0]

        # vehicle parameters
        self.__veh_params = _veh_params
        self.__localgg = None
        self.__ax_max_machines = None

        # calculate thresholds to set the maximum allowed cursor offset
        self.__time_threshold_data = max(np.mean(np.diff(self.__time_stamps_data)) * 2, 0.01)
        self.__time_threshold_msgs = max(np.mean(np.diff(self.__time_stamps_msgs)) * 2, 0.01)

    def plot_timestamp_n(self,
                         _plot_handler,
                         _n: int,
                         _traj_type: str = 'perf') -> None:
        """
        Update the data in the main plot for a specified time-stamp (integer from stamp in log-file).

        :param _plot_handler:        plot-handler object
        :param _n:                   integer time-stamp (in log-file) to be visualized
        :param _traj_type:           trajectory type to be displayed, either 'perf' (performance) or 'emerg' (emergency)

        """

        # setup order list (in some cases, 'perf' and 'emerg' are both always plotted, this list sets the order of these
        # two depending on the priorized selection
        if _traj_type == 'perf':
            traj_order = ['perf', 'emerg']
        elif _traj_type == 'emerg':
            traj_order = ['emerg', 'perf']
        else:
            raise ValueError("Requested unsupported _traj_type (" + _traj_type + ")!")

        if not self.__n_store == _n or not self.__type_store == _traj_type:
            self.__n_store = _n
            self.__type_store = _traj_type

            ego_traj, objects, _safety_base = get_data_from_line(file_path_data, _n)

            # -- calculate time axis for path dependent velocity profile -----------------------------------------------
            t_coord = dict()
            for tmp_tt in traj_order:
                # calculate s-coordinates
                s_coord = ego_traj[tmp_tt]['data_ref'][:, 0]

                v_avg = 0.5 * (ego_traj[tmp_tt]['data_ref'][1:, 5] + ego_traj[tmp_tt]['data_ref'][:-1, 5])
                t_coord[tmp_tt] = np.cumsum(np.concatenate(([0],
                                                            np.diff(s_coord) / np.where(v_avg > 0.01, v_avg, 0.01))))

            # -- plot gained information about ego vehicle -------------------------------------------------------------
            # NOTE: If multiple lines to be plotted, simply add further entries to lists
            vel_info_list = [np.column_stack((t_coord[traj_order[0]], ego_traj[traj_order[0]]['data_ref'][:, 5])),
                             np.column_stack((t_coord[traj_order[1]], ego_traj[traj_order[1]]['data_ref'][:, 5]))]
            kappa_info_list = [np.column_stack((t_coord[traj_order[0]], ego_traj[traj_order[0]]['data_ref'][:, 4])),
                               np.column_stack((t_coord[traj_order[1]], ego_traj[traj_order[1]]['data_ref'][:, 4]))]
            psi_info_list = [np.column_stack((t_coord[traj_order[0]],
                                              ego_traj[traj_order[0]]['data_ref'][:, 3] / (10 * np.pi))),
                             np.column_stack((t_coord[traj_order[1]],
                                              ego_traj[traj_order[1]]['data_ref'][:, 3] / (10 * np.pi)))]

            # plot course of temporal information (= non-path)
            _plot_handler.plot_time_rel_line(line_coords_list=[vel_info_list, kappa_info_list, psi_info_list])

            # update reference ego vehicle pose
            _plot_handler.plot_vehicle(pos=[ego_traj[_traj_type]['data_ref'][0, 1:3]],
                                       heading=[ego_traj[_traj_type]['data_ref'][0, 3]],
                                       width=self.__veh_params.get('veh_width', 2.8),
                                       length=self.__veh_params.get('veh_length', 4.7),
                                       zorder=12,
                                       color_str='TUM_grey_dark',
                                       id_in='ego_ref')

            # update intern ego vehicle pose
            _plot_handler.plot_vehicle(pos=[ego_traj[_traj_type]['data_intern'][0, 1:3]],
                                       heading=[ego_traj[_traj_type]['data_intern'][0, 3]],
                                       width=self.__veh_params.get('veh_width', 2.8),
                                       length=self.__veh_params.get('veh_length', 4.7),
                                       zorder=13,
                                       color_str='TUM_orange',
                                       id_in='ego')

            # plot (original) path
            if _traj_type == 'emerg' and min(ego_traj[_traj_type]['data_ref'][:, 5]) < 0.5:
                idx_standstill = np.where(ego_traj[_traj_type]['data_ref'][:, 5] < 0.5)[0][0]
            else:
                idx_standstill = None

            _plot_handler.highlight_path(path_coords=ego_traj[_traj_type]['data_ref'][:idx_standstill, 1:3],
                                         id_in='log',
                                         zorder=9,
                                         color_str='TUM_orange')

            # -- plot object vehicles in the scene ---------------------------------------------------------------------
            # update reference vehicle poses
            if objects['data_ref']:
                _plot_handler.plot_vehicle(pos=[[obj['X'], obj['Y']] for obj in objects['data_ref'].values()],
                                           heading=[obj['theta'] for obj in objects['data_ref'].values()],
                                           width=list(objects['data_ref'].values())[0]['width'],
                                           length=list(objects['data_ref'].values())[0]['length'],
                                           zorder=11,
                                           color_str='TUM_grey_dark',
                                           id_in='objects_ref')

            # update internal vehicle poses
            if objects['data_intern']:
                _plot_handler.plot_vehicle(pos=[[obj['X'], obj['Y']] for obj in objects['data_intern'].values()],
                                           heading=[obj['theta'] for obj in objects['data_intern'].values()],
                                           width=list(objects['data_intern'].values())[0]['width'],
                                           length=list(objects['data_intern'].values())[0]['length'],
                                           zorder=12,
                                           color_str='TUM_blue',
                                           id_in='objects')

            # -- plot RSS bounds ---------------------------------------------------------------------------------------
            coord_array = [[], []]
            for p_key in _safety_base.keys():
                if 'd_lon_bound_' in p_key:
                    ego_pos = ego_traj[_traj_type]['data_ref'][0, 1:3]
                    ego_head = ego_traj[_traj_type]['data_ref'][0, 3]

                    d_lon = _safety_base[p_key] + self.__veh_params.get('veh_length', 4.7) / 2
                    d_lat = _safety_base[p_key.replace('_lon_', '_lat_')] + self.__veh_params.get('veh_width', 2.8) / 2

                    # calculate line holding points:
                    # * center to the left side of the vehicle (d_alt_bound*)
                    coord_array[0].append(ego_pos[0] + np.cos(ego_head - np.pi) * d_lat)
                    coord_array[1].append(ego_pos[1] + np.sin(ego_head - np.pi) * d_lat)

                    # * coordinate crossing d_lon_bound_* and d_lat_bound* left (assume that lat bound is present)
                    coord_array[0].append(ego_pos[0] + np.cos(ego_head + np.pi / 2) * d_lon
                                          + np.cos(ego_head - np.pi) * d_lat)
                    coord_array[1].append(ego_pos[1] + np.sin(ego_head + np.pi / 2) * d_lon
                                          + np.sin(ego_head - np.pi) * d_lat)

                    # * coordinate crossing d_lon_bound_* and d_lat_bound* right (assume that lat bound is present)
                    coord_array[0].append(ego_pos[0] + np.cos(ego_head + np.pi / 2) * d_lon + np.cos(ego_head) * d_lat)
                    coord_array[1].append(ego_pos[1] + np.sin(ego_head + np.pi / 2) * d_lon + np.sin(ego_head) * d_lat)

                    # * center to the right side of vehicle (d_lat_bound*)
                    coord_array[0].append(ego_pos[0] + np.cos(ego_head) * d_lat)
                    coord_array[1].append(ego_pos[1] + np.sin(ego_head) * d_lat)

                    # insert "None" between non-adjacent points (in order to separate plotted lines)
                    coord_array[0].append(None)
                    coord_array[1].append(None)

            _plot_handler.highlight_path(path_coords=np.array(coord_array).T,
                                         id_in='RSS',
                                         color_str='red',
                                         linewidth=2.0,
                                         zorder=8)

            # -- plot guaranteed occupation sets -----------------------------------------------------------------------
            polygon_list = []
            for p_key in _safety_base.keys():
                if 'gocc_' in p_key and _traj_type in p_key:
                    polygon_list.append(np.column_stack(_safety_base[p_key]))

            _plot_handler.plot_polygon(polygon_list=polygon_list,
                                       color_str='TUM_blue',
                                       id_in='guaranteed_occupation',
                                       zorder=10,
                                       alpha=0.2)

            # collision sets for guaranteed occupation sets (only when collision occurred)
            polygon_list = []
            for p_key in _safety_base.keys():
                if 'gocccol' in p_key and _traj_type in p_key:
                    polygon_list.append(np.column_stack(_safety_base[p_key]))

            _plot_handler.plot_polygon(polygon_list=polygon_list,
                                       color_str=None,
                                       color_e_str='r',
                                       id_in='guaranteed_occupation_collision',
                                       zorder=10,
                                       alpha=0.8)

            # -- plot rule-based reachable sets of object vehicles -----------------------------------------------------
            polygon_list = []
            for p_key in _safety_base.keys():
                if 'rrese_' in p_key and _traj_type in p_key:
                    polygon_list.append(np.column_stack(_safety_base[p_key]))

            _plot_handler.plot_polygon(polygon_list=polygon_list,
                                       color_str='TUM_green',
                                       id_in='rule_based_set',
                                       zorder=10,
                                       alpha=0.2)

            # collision sets for rule-based reachable sets (only when collision occurred)
            polygon_list = []
            for p_key in _safety_base.keys():
                if 'rresecol' in p_key and _traj_type in p_key:
                    polygon_list.append(np.column_stack(_safety_base[p_key]))

            _plot_handler.plot_polygon(polygon_list=polygon_list,
                                       color_str=None,
                                       color_e_str='r',
                                       id_in='rule_based_set_collision',
                                       zorder=10,
                                       alpha=0.8)

            # -- highlight static collision areas ----------------------------------------------------------------------
            pos_list = []
            for p_key in _safety_base.keys():
                if 'stat_intersect' in p_key and _traj_type in p_key:
                    pos_list = [[p[0] for p in _safety_base[p_key]], [p[1] for p in _safety_base[p_key]]]
                    break

            _plot_handler.highlight_pos(pos_coords=pos_list,
                                        color_str='r',
                                        zorder=10,
                                        marker='x',
                                        id_in='stat_intersect')

            # -- highlight safe end state ------------------------------------------------------------------------------
            pos_list = []
            time_list = []
            for p_key in _safety_base.keys():
                if 'safe_end_state' in p_key:
                    if not _safety_base[p_key]:
                        # highlight unsafe end state for active trajectory in position plot
                        if _traj_type in p_key:
                            pos_list = [[ego_traj[_traj_type]['data_ref'][-1, 1]],
                                        [ego_traj[_traj_type]['data_ref'][-1, 2]]]

                        # highlight unsafe end state in temporal plot - for selected type or (if not present) for other
                        if 'perf' in p_key and (not time_list or _traj_type in p_key):
                            time_list = [[t_coord['perf'][-1]], [ego_traj['perf']['data_ref'][-1, 5]]]
                        elif 'emerg' in p_key and (not time_list or _traj_type in p_key):
                            time_list = [[t_coord['emerg'][-1]], [ego_traj['emerg']['data_ref'][-1, 5]]]

            _plot_handler.highlight_pos(pos_coords=pos_list,
                                        color_str='r',
                                        zorder=10,
                                        marker='X',
                                        id_in='safe_end_state')

            _plot_handler.highlight_time(time_coords=time_list,
                                         color_str='r',
                                         zorder=10,
                                         marker='X',
                                         id_in='safe_end_state')

            # -- highlight turn radius violation -----------------------------------------------------------------------
            coord_array = [[], []]
            for p_key in _safety_base.keys():
                if 'stat_idx_turn_rad_err' in p_key and _traj_type in p_key:

                    prev_idx = None
                    for idx in _safety_base[p_key]:
                        # extract coordinate in line
                        coord_array[0].append(ego_traj[_traj_type]['data_ref'][idx, 1])
                        coord_array[1].append(ego_traj[_traj_type]['data_ref'][idx, 2])

                        # insert "None" between non-adjacent points (in order to separate plotted lines)
                        if prev_idx is not None and prev_idx != (idx - 1):
                            coord_array[0].append(None)
                            coord_array[1].append(None)

                        prev_idx = idx

                    break

            _plot_handler.highlight_path(path_coords=np.array(coord_array).T,
                                         id_in='turn_violation',
                                         color_str='red',
                                         linewidth=3.0,
                                         zorder=8)

            # -- reach set bound cut -----------------------------------------------------------------------------------
            coord_array = [[], []]
            for p_key in _safety_base.keys():
                if 'bound_reach_set_outline_' in p_key and _traj_type in p_key:

                    outline = np.array(_safety_base[p_key])
                    coord_array[0] = np.concatenate((outline[:, 0], [outline[0, 0]]))
                    coord_array[1] = np.concatenate((outline[:, 1], [outline[0, 1]]))

                    break

            _plot_handler.highlight_path(path_coords=np.array(coord_array).T,
                                         id_in='bound_reach_set_outline',
                                         color_str='red',
                                         linewidth=2.0,
                                         zorder=8)

            # -- acceleration data -------------------------------------------------------------------------------------
            for p_key in _safety_base.keys():
                if 'a_lat_used' in p_key and _traj_type in p_key:
                    a_lat_used = np.array(_safety_base[p_key])
                    a_lon_used = np.array(_safety_base[p_key.replace('a_lat_used', 'a_lon_used')])
                    a_comb_used_perc = np.array(_safety_base[p_key.replace('a_lat_used', 'a_comb_used_perc')])

                    _plot_handler.update_acc_plot(acc_limit_valid=[a_lat_used[a_comb_used_perc <= 1.0],
                                                                   a_lon_used[a_comb_used_perc <= 1.0]],
                                                  acc_limit_invalid=[a_lat_used[a_comb_used_perc > 1.0],
                                                                     a_lon_used[a_comb_used_perc > 1.0]])

            # plot text to top left
            # plot_handler.update_text_field(text_str="empty",
            #                                color_str='r')

            # plot text to top right (time stamps of trajectory and objects)
            plot_handler.update_text_field(text_str="Processed Time-Stamps:\n - Traj.: %.3f"
                                                    "\n - Obj.:  %.3f" % (ego_traj['perf']['stamp'], objects['stamp']),
                                           text_field_id=2)

            self.__working = False
            _plot_handler.show_plot(non_blocking=True)

        self.__working = False

    def get_closest_timestamp(self,
                              _plot_handler,
                              _safety_ins,
                              _time_stamp: float,
                              _traj_type: str = 'perf') -> None:
        """
        Searches for closest time-stamp in log and triggers visualization of corresponding log-entry.

        :param _plot_handler:       plot-handler object
        :param _safety_ins:         safety-inspector object
        :param _time_stamp:         float time-stamp to be visualized
        :param _traj_type:          trajectory type to be displayed (either 'perf' or 'emerg')

        """

        # move cursors in safety inspector
        _safety_ins.move_cursor(t=_time_stamp)

        # check if currently still processing (avoid lags)
        if not self.__working:
            self.__working = True

            dists = abs(np.array(self.__time_stamps_msgs) - _time_stamp)

            # if cursor is in range of valid measurements
            if np.min(dists) < self.__time_threshold_msgs:
                # get index of closest time stamp
                _, idx = min((val, idx) for (idx, val) in enumerate(dists))

                # trigger plot update
                _plot_handler.highlight_timeline(time_stamp=self.__time_stamps_msgs[idx],
                                                 type_in=self.__time_msgs_types[idx],
                                                 message=self.__time_msgs_content[idx])

            dists = abs(np.array(self.__time_stamps_data) - _time_stamp)

            # if cursor is in range of valid measurements
            if np.min(dists) < self.__time_threshold_data:
                # get index of closest time stamp
                _, idx = min((val, idx) for (idx, val) in enumerate(dists))

                # trigger plot update
                self.plot_timestamp_n(_plot_handler,
                                      idx,
                                      _traj_type=_traj_type)
            else:
                self.__working = False

            # check if reached different map region
            i = 0
            while i < len(self.__time_stamps_map) and self.__time_stamps_map[i] < _time_stamp:
                i += 1

            # update map, if the map changed compared to the last time-stamp
            i -= 1
            if self.__time_stamps_map[i] != self.__active_map_stamp:
                self.__active_map_stamp = self.__time_stamps_map[i]

                _bound_l, _bound_r, _localgg, _, _acc_limit_factor = get_map_from_line(_file_path=file_path_map,
                                                                                       _line_num=max(i, 0))

                if self.__veh_params is not None and localgg is not None:
                    plot_handler.init_acc_plot(a_lat_max_tires=max(_localgg[:, 4]) * _acc_limit_factor,
                                               a_lon_max_tires=max(_localgg[:, 3]) * _acc_limit_factor,
                                               dyn_model_exp=self.__veh_params['dyn_model_exp'])

                plot_handler.plot_map(bound_l=_bound_l,
                                      bound_r=_bound_r)
                self.__working = False


def plot_all_poses(_plot_handler,
                   _file_path: str,
                   _t_stamps: list):

    # extract relevant poses
    num_steps = int((_t_stamps[-1] - _t_stamps[0]
                     - PLOT_VEH_PARAM['initial_t_offset']) / PLOT_VEH_PARAM["temporal_increment"]) + 1
    data_dict_template = {'pos': [],
                          'psi': [],
                          'text_list': []}
    data_container = {'ego': copy.deepcopy(data_dict_template)}

    for i in range(num_steps):
        # get time stamp
        t = _t_stamps[0] + PLOT_VEH_PARAM['initial_t_offset'] + i * PLOT_VEH_PARAM["temporal_increment"]

        # get data
        t_close = int(np.argmax(_t_stamps > t))
        ego_traj_prev, objects_prev = get_data_from_line(_file_path=_file_path,
                                                         _line_num=max(t_close - 1, 0))[:2]
        ego_traj_post, objects_post = get_data_from_line(_file_path=_file_path,
                                                         _line_num=t_close)[:2]

        data_container['ego']['pos'].append([np.interp(t, _t_stamps[t_close - 1:t_close + 1],
                                                       [ego_traj_prev['perf']['data_intern'][0, 1],
                                                        ego_traj_post['perf']['data_intern'][0, 1]]),
                                             np.interp(t, _t_stamps[t_close - 1:t_close + 1],
                                                       [ego_traj_prev['perf']['data_intern'][0, 2],
                                                        ego_traj_post['perf']['data_intern'][0, 2]])])

        data_container['ego']['psi'].append(stt.interp_heading.interp_heading(
            heading=np.array([ego_traj_prev['perf']['data_intern'][0, 3], ego_traj_post['perf']['data_intern'][0, 3]]),
            t_series=_t_stamps[t_close - 1:t_close + 1],
            t_in=t)
        )

        data_container['ego']['text_list'].append("$t^{" + "veh1" + "}_{" + str(i) + "}$")

        # for all objects persistent in the two selected time-stamps
        for obj in list(set(objects_prev['data_intern'].keys()).intersection(objects_post['data_intern'].keys())):
            if obj not in data_container:
                data_container[obj] = copy.deepcopy(data_dict_template)

            data_container[obj]['pos'].append([np.interp(t, _t_stamps[t_close - 1:t_close + 1],
                                                         [objects_prev['data_intern'][obj]['X'],
                                                          objects_post['data_intern'][obj]['X']]),
                                               np.interp(t, _t_stamps[t_close - 1:t_close + 1],
                                                         [objects_prev['data_intern'][obj]['Y'],
                                                          objects_post['data_intern'][obj]['Y']])])

            data_container[obj]['psi'].append(stt.interp_heading.interp_heading(
                heading=np.array([objects_prev['data_intern'][obj]['theta'],
                                  objects_post['data_intern'][obj]['theta']]),
                t_series=_t_stamps[t_close - 1:t_close + 1], t_in=t))

            data_container[obj]['text_list'].append("$t^{" + obj.replace("_", "") + "}_{" + str(i) + "}$")

    # plot all vehicles
    for veh_key in data_container.keys():
        _plot_handler.plot_vehicle(pos=np.array(data_container[veh_key]['pos']),
                                   heading=data_container[veh_key]['psi'],
                                   width=2.8,
                                   length=4.7,
                                   color_str=('TUM_orange' if veh_key == 'ego' else 'TUM_blue'),
                                   zorder=PLOT_VEH_PARAM['zorder'],
                                   id_in=veh_key + "_all",
                                   alpha=PLOT_VEH_PARAM['alpha'])

        # print text if enabled and not currently selected (performance issues)
        if PLOT_VEH_PARAM['plot_text']:
            # gather positions of all other vehicles
            avoid_pos = None
            for tmp_key in data_container.keys():
                if tmp_key != veh_key:
                    if avoid_pos is None:
                        avoid_pos = np.array(data_container[tmp_key]['pos'])
                    else:
                        avoid_pos = np.vstack((avoid_pos, np.array(data_container[tmp_key]['pos'])))

            # print text next to each pose
            _plot_handler.plot_text(pos=np.array(data_container[veh_key]['pos']),
                                    heading=data_container[veh_key]['psi'],
                                    text_list=data_container[veh_key]['text_list'],
                                    text_dist=PLOT_VEH_PARAM['plot_text_distance'],
                                    plot_ith=PLOT_VEH_PARAM['plot_text_every_ith_element'],
                                    avoid_pos=avoid_pos,
                                    id_in=veh_key + "_all")


# ----------------------------------------------------------------------------------------------------------------------
# MAIN SCRIPT ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    toppath = os.path.dirname(os.path.dirname(os.path.realpath(__file__ + "/../../")))
    sys.path.append(toppath)

    # ----- get log-file name (via arguments or most recent one) -----
    if len(sys.argv) == 2:
        # if one argument provided, assume provided file name
        file_path = sys.argv[1]  # First argument
    else:
        # use most recent file if no arguments provided
        list_of_files = glob.glob(
            os.path.expanduser(toppath + '/logs/'
                               + datetime.datetime.now().strftime("%Y_%m_%d") + '/*_safety_log_data.csv'))
        if list_of_files:
            file_path = max(list_of_files, key=os.path.getctime)
        else:
            raise ValueError("Could not find any logs in the specified folder! Please provide a file path argument.")

        # specific file
        # file_path = os.path.expanduser(toppath + '/logs/2019_03_22/11_00_47_data.csv')

    # extract common file parent
    file_path = file_path[:file_path.rfind("_")]

    # get file paths for all sub-files
    file_path_data = file_path + "_data.csv"
    file_path_msg = file_path + "_msg.csv"
    file_path_map = file_path + "_map.csv"

    # get first line without comments in data file
    with open(file_path_data) as file:
        line = file.readline()
        n_skip = 0
        while line[0] == '#':
            line = file.readline()
            n_skip += 1

    # -- get path to relevant graph-base object (for now: assuming the one in the repo) --------------------------------
    # get time stamps from file
    time_stamps = list(np.genfromtxt(file_path_data, delimiter=';', skip_header=n_skip, names=True)['time'])

    # get message data
    time_stamps_msgs = list(np.genfromtxt(file_path_msg, delimiter=';', skip_header=0, names=True)['time'])
    msgs_type = list(np.genfromtxt(
        file_path_msg, delimiter=';', skip_header=0, dtype=None, encoding=None, names=True)['type'].astype(str))
    msgs_content = list(np.genfromtxt(
        file_path_msg, delimiter=';', skip_header=0, dtype=None, encoding=None, names=True)['message'].astype(str))

    # get map data
    time_stamps_map = list(np.atleast_1d(np.genfromtxt(file_path_map, delimiter=';',
                                                       skip_header=0, names=True)['time']))

    # get safety (title) data
    safety_stat_course = np.genfromtxt(
        file_path_data, delimiter=';', skip_header=n_skip, dtype=bool, encoding=None, names=True)['safety_static']
    safety_dyn_course = np.genfromtxt(
        file_path_data, delimiter=';', skip_header=n_skip, dtype=bool, encoding=None, names=True)['safety_dynamic']
    safety_overall_course = safety_stat_course * safety_dyn_course

    safety_stat_course = np.column_stack((time_stamps, safety_stat_course))
    safety_dyn_course = np.column_stack((time_stamps, safety_dyn_course + 1.5))
    safety_overall_course = np.column_stack((time_stamps, safety_overall_course + 3.0))

    # get safety base for safety inspector
    veh_params = None
    with open(file_path_data) as file:
        # get to top of file
        file.seek(0)

        # handle comments
        line = file.readline()
        while line[0] == '#':
            # check if line holds vehicle parameters
            if "veh_width" in line:
                # load vehicle parameters (omit leading '#' and tailing '\n')
                veh_params = json.loads(line[1:-1])

            line = file.readline()

        # get header (":-1" in order to remove tailing newline character)
        header = line[:-1]
        content = file.readlines()

    safety_base = {}
    calc_times = {}
    for line in content:
        data = dict(zip(header.split(";"), line.split(";")))

        temp_safety_base = json.loads(data['safety_base'])
        for key in temp_safety_base.keys():
            if key not in safety_base.keys():
                safety_base[key] = []

            safety_base[key].append(temp_safety_base[key])

            # extract calc times
            if key == "mod_calctime":
                for calckey in temp_safety_base[key].keys():

                    if calckey not in calc_times:
                        calc_times[calckey] = []

                    calc_times[calckey].append(temp_safety_base[key][calckey])

    # -- init main debug handler and safety parameter inspector --------------------------------------------------------

    # timing plot
    plot_timing_histogram.timing_histogram(calc_times=calc_times)

    # init safety param inspector
    safety_ins = SafetyParamInspector.SafetyParamInspector(time_stamps=time_stamps,
                                                           safety_combined=list(safety_overall_course[:, 1] - 3.0),
                                                           safety_base=safety_base)

    # init debug handler
    dh = DebugHandler(_time_stamps_data=time_stamps,
                      _time_stamps_msgs=time_stamps_msgs,
                      _time_stamps_map=time_stamps_map,
                      _time_msgs_types=msgs_type,
                      _time_msgs_content=msgs_content,
                      _veh_params=veh_params)

    # -- initialize main plot ------------------------------------------------------------------------------------------

    # plot time stamps
    plot_handler = PlotHandler.PlotHandler(plot_title="Safety Log Visualization")

    # plot initial map
    bound_l, bound_r, localgg, _, acc_limit_factor = get_map_from_line(_file_path=file_path_map,
                                                                       _line_num=0)
    plot_handler.plot_map(bound_l=bound_l,
                          bound_r=bound_r)

    if veh_params is not None and localgg is not None:
        plot_handler.init_acc_plot(a_lat_max_tires=max(localgg[:, 4]) * acc_limit_factor,
                                   a_lon_max_tires=max(localgg[:, 3]) * acc_limit_factor,
                                   dyn_model_exp=veh_params['dyn_model_exp'])

    # get combined time stamps
    time_stamps_comb = time_stamps + time_stamps_msgs
    types_comb = (["DATA"] * len(time_stamps)) + msgs_type

    # plot time line
    plot_handler.plot_timeline_stamps(time_stamps=time_stamps_comb,
                                      types=types_comb,
                                      lambda_fct=lambda time_stamp, traj_type: dh.get_closest_timestamp(plot_handler,
                                                                                                        safety_ins,
                                                                                                        time_stamp,
                                                                                                        traj_type))

    plot_handler.plot_timeline_course(line_coords_list=[safety_stat_course, safety_dyn_course, safety_overall_course])

    # initialize track plot with first entry in log
    dh.plot_timestamp_n(plot_handler, 1)

    # plot all vehicles, if activated
    if PLOT_VEH:
        plot_all_poses(_plot_handler=plot_handler,
                       _file_path=file_path_data,
                       _t_stamps=time_stamps)

    plot_handler.show_plot()
