"""
Main script minimal realization example launching the online verification framework.

:Authors:
    * Tim Stahl <tim.stahl@tum.de>

:Created on:
    14.06.2019
"""

import os
# -- Limit number of OPENBLAS library threads --
# On linux based operation systems, we observed a occupation of all cores by the underlying openblas library. Often,
# this slowed down other processes, as well as the planner itself. Therefore, it is recommended to set the number of
# threads to one. Note: this import must happen before the import of any openblas based package (e.g. numpy)
os.environ['OPENBLAS_NUM_THREADS'] = str(1)

import sys
import json
import time
import configparser
import logging
import numpy as np
import scenario_testing_tools as stt

# custom modules
import trajectory_supervisor

SCENARIO_REL_PATH = "/sample_files/modena_T1_cutin_collision.saa"
# SCENARIO_REL_PATH = "/sample_files/modena_T3_T4_catchingup_collision.saa"


def get_t_step_from_scenario(file_path_: str,
                             t_: float,
                             time_f_: np.ndarray = None) -> tuple:
    """
    Retrieve a time step from a provided scenario file.
    This scenario file is used as an example application of the Supervisor, to be replaced by your trajectory planner.

    :param file_path_:      File path pointing to a scenario file
    :param t_:              Time step to be retrieved from scenario file [in s]
    :param time_f_:         Time array to be used to locate time step in file (speed up, provide None on first call)
    :returns:
        * **traj_perf** -   Performance trajectory in form of dict with keys: 'id', 'time', 'traj'
        * **traj_em** -     Emergency trajectory in form of dict with keys: 'id', 'time', 'traj'
        * **object_list** - Object list hosting dicts of objects
        * **time_f** -      Retrieved  time array from file (to be provided on next call for speedup)
    """

    t_step, _, _, _, _, _, traj_perf, traj_em, tmp_objects, time_f_ = stt.get_scene_timesample. \
        get_scene_timesample(file_path=file_path_,
                             t_in=t_,
                             time_f=time_f_)

    # calculate s-coordinate and prepend to retrieved trajectory
    s_coord = np.cumsum(np.sqrt(np.power(np.diff(traj_perf[:, 0]), 2)
                                + np.power(np.diff(traj_perf[:, 1]), 2)))
    traj_perf = np.column_stack((list(np.insert(s_coord, 0, 0.0)), traj_perf))

    s_coord_em = np.cumsum(np.sqrt(np.power(np.diff(traj_em[:, 0]), 2)
                                   + np.power(np.diff(traj_em[:, 1]), 2)))
    traj_em = np.column_stack((list(np.insert(s_coord_em, 0, 0.0)), traj_em))

    # assemble requested trajectory format
    traj_perf = {'id': int(t_ * 100),       # generate exemplary id
                 'time': t_step,
                 'traj': traj_perf}

    traj_em = {'id': int(t_ * 100) + 1,  # generate exemplary id
               'time': t_step,
               'traj': traj_em}

    # prepare object list
    # object-list data
    obj_list = {}
    if tmp_objects is not None and tmp_objects:
        for obj_key in tmp_objects.keys():
            obj_list[obj_key] = {'time': t_step,
                                 'type': 'car',
                                 'form': 'rectangle',
                                 'X': tmp_objects[obj_key]['X'],
                                 'Y': tmp_objects[obj_key]['Y'],
                                 'theta': tmp_objects[obj_key]['psi'],
                                 'v_x': tmp_objects[obj_key]['vel'],
                                 'length': tmp_objects[obj_key]['length'],
                                 'width': tmp_objects[obj_key]['width']}

    return traj_perf, traj_em, obj_list, time_f_


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # HANDLE INPUTS ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    target = 'sim'           # when no target provided, assume sim
    log_identifier = ""

    if len(sys.argv) >= 2:
        # if first argument provided, assume provided target name {'sim', 'ci'}
        target = sys.argv[1]                        # first argument

    # top level path (module directory)
    toppath = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(toppath)

    # ------------------------------------------------------------------------------------------------------------------
    # SPECIFY FILE PATHS .----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # get target specific supmod config path
    supmod_config_path = toppath + "/params/supmod_config_" + target + ".ini"

    # check if target specific config file exists, else use generic one
    if not os.path.isfile(supmod_config_path):
        supmod_config_path = toppath + "/params/supmod_config.ini"

    # load interface config file
    interface_param = configparser.ConfigParser()
    if not interface_param.read(toppath + "/params/interface_config_" + target + ".ini"):
        raise ValueError('Specified interface config file does not exist or is empty!')

    # ------------------------------------------------------------------------------------------------------------------
    # INIT SUPERVISOR --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # get Supervisor config parameters
    module_enabled = json.loads(interface_param.get('GENERAL_SAFETY', 'module_enabled'))
    veh_params = json.loads(interface_param.get('GENERAL_SAFETY', 'veh_params'))

    # init Supervisor object
    supervisor = trajectory_supervisor.supervisor.Supervisor(module_enabled=module_enabled,
                                                             supmod_config_path=supmod_config_path,
                                                             veh_params=veh_params)

    # fetch logger (initialized in Supervisor class)
    logger = logging.getLogger("supervisor_logger")

    # ------------------------------------------------------------------------------------------------------------------
    # LOAD AND SET ENVIRONMENT DATA ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # import map
    bound_l, bound_r = stt.get_scene_track.get_scene_track(file_path=toppath + SCENARIO_REL_PATH)

    # import localgg and ax_max_machines
    ggv, ax_max_machines = stt.get_scene_veh_param.get_scene_veh_param(file_path=toppath + SCENARIO_REL_PATH)

    # convert ggv to localgg (dimension-wise)
    localgg = np.column_stack((np.zeros((ggv.shape[0], 3)), ggv[:, 1:]))

    # set map for Supervisor
    supervisor.set_environment(bound_left=bound_l,
                               bound_right=bound_r,
                               ax_max_machines=ax_max_machines,
                               localgg=localgg)

    # ------------------------------------------------------------------------------------------------------------------
    # ONLINE EXECUTION -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # loop through process
    time_f = None
    tic = time.time()
    while True:
        tic_iter = time.time()

        # -- get sample trajectory and sample object list to be passed to the Supervisor -------------------------------
        # NOTE: exemplary retrieval of trajectory and object list from scenario file, to be replaced by your system data
        ego_traj, ego_traj_em, object_list, time_f = get_t_step_from_scenario(file_path_=toppath + SCENARIO_REL_PATH,
                                                                              t_=time.time() - tic,
                                                                              time_f_=time_f)

        # break if end of scenario file reached
        if ego_traj['time'] == time_f[-1]:
            break

        # -- process data ----------------------------------------------------------------------------------------------
        result_perf, result_em = supervisor.process_data_simple(traj_perf=ego_traj,
                                                                traj_em=ego_traj_em,
                                                                objects=object_list)

        # -- publish results - here printed as example -----------------------------------------------------------------
        print("Results for step t=%.2fs:  Performance trajectory: %s;  Emergency trajectory: %s"
              % (ego_traj['time'], "SAFE" if result_perf else "UNSAFE", "SAFE" if result_em else "UNSAFE"))

        # -- publish resulting safe trajectory - here printed as example -----------------------------------------------
        safe_traj = supervisor.get_safe_trajectory()
        print(" -> ID of safe trajectory: " + str(safe_traj["id"]) + "\n")

        # wait for simulation purposes to complete 100ms in this iteration
        if time.time() - tic_iter < 0.1:
            time.sleep(0.1 - (time.time() - tic_iter))
