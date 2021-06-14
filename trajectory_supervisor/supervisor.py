import os
import sys
import time
import numpy as np
import json
import configparser
import logging
import platform
import multiprocessing
import scenario_testing_tools as stt

import trajectory_supervisor

# required vehicle parameters
REQ_VEH_PARAM = ('veh_width', 'veh_length', 'turn_rad', 'dyn_model_exp', 'drag_coeff', 'm_veh')
LOG_HEADER = ("time;traj_stamp;traj_perf_ref;traj_perf;traj_emerg_ref;traj_emerg;objects_stamp;objects_ref;"
              "objects;safety_static;safety_dynamic;safety_base")
MAP_LOG_HEADER = "time;bound_l;bound_r;localgg;ax_max_machines;acc_limit_factor"


class Supervisor(object):
    """
    Supervisor class, holding the core functionality and execution steps for an online verification of motion
    primitives. This class handles the following core tasks:

    * Holds methods to retrieve the latest parameter updates (e.g. trajectories, object list, ...)
    * Handles the temporal synchronisation of trajectories and object-list
    * Triggers calculation and retrieves results of safety scores

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        14.06.2019

    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 module_enabled: dict,
                 supmod_config_path: str,
                 veh_params: dict,
                 logging_param: dict = None,
                 zone_file_path: str = None,
                 use_mp: bool = False) -> None:
        """
        :param module_enabled:       dict of dicts specifying which modules of the supervisor are enabled
        :param supmod_config_path:   path pointing to the config file of the Supervisor modules
        :param logging_param:        dict of logging parameters (e.g. file paths), must hold the following keys:
                                      file_log_level -    string spec. file log level (e.g. INFO, WARNING, CRITICAL)
                                      console_log_level - string spec. console log level (e.g. INFO, WARNING, CRITICAL)
                                      log_path_data -     string for path of file where the data log should be generated
                                      log_path_msg -      string for path of file where the msg log should be generated
                                      log_path_map -      string for path of file where the map log should be generated
                                      NOTE: - all log files should reside in the same folder, for the log viewer to
                                              automatically locate the most recent log, stick to the example on github
                                            - if "None" is provided, logging is disabled; ignoring setting in config
        :param veh_params:           dict of vehicle parameters; must hold the following keys:
                                      veh_width -         width of the ego-vehicle [in m]
                                                          NOTE: also used for offline pre-calculations of other vehicles
                                      veh_length -        length of the ego-vehicle [in m]
                                                          NOTE: also used for offline pre-calculations of other vehicles
                                      turn_rad -          turning radius of vehicle [in m]
                                      dyn_model_exp -     exponent used in vehicle dynamics model (range [1.0, 2.0])
                                                          NOTE: 2.0 -> ideal friction circle; 1.0 -> clean diamond shape
                                      drag_coeff -        drag coeff. incl. all constants (0.5*c_w*A_front*rho_air)
                                                          set zero to disable [in m2*kg/m3]
                                      m_veh -             vehicle mass [in kg]
        :param zone_file_path:       (optional) path pointing to a Roborace zone spec. (e.g. for reach set reduct.)
        :param use_mp:               (optional) if set to true and executed on a Linux machine, multiprocessing is used

        """

        # -- define local variables ------------------------------------------------------------------------------------
        # dict holding dicts for performance and emergency trajectory
        self.__traj = {'perf': None,
                       'emerg': None}
        self.__safe_emerg_traj = None

        # time stamp at which the object list was generated (in order to relate to rest of surrounding)
        self.__objects = {}

        self.__bound_l = None
        self.__bound_r = None
        self.__ref_line = None
        self.__norm_vec = None
        self.__tw_left = None
        self.__tw_right = None
        self.__occ_map = None
        self.__ego_path = None
        self.__ax_max_machines = None
        self.__localgg = None
        self.__valid_dyn = None
        self.__valid_stat = None
        self.__param_dict = None
        self.__fired_modules = []

        self.__zone_file_path = zone_file_path
        self.__module_enabled = module_enabled
        self.__use_mp = use_mp

        # read frequently used parameters from configuration file
        self.__supmod_config_path = supmod_config_path
        supmod_configparser = configparser.ConfigParser()
        if not supmod_configparser.read(supmod_config_path):
            raise ValueError('Specified config file does not exist or is empty!')

        self.__t_warn = supmod_configparser.getfloat('GENERAL', 't_warn')
        self.__acc_limit_factor = supmod_configparser.getfloat('FRICTION', 'allowed_acc')

        # module dict
        self.__mod_dict = dict()
        self.__mod_calctime = dict()

        # check and store vehicle parameters
        if all(k in veh_params for k in REQ_VEH_PARAM):
            self.__veh_params = veh_params
        else:
            raise ValueError("Provided vehicle parameters are missing key(s)! The following parameter(s) is/are"
                             " missing: " + str(list((x for x in REQ_VEH_PARAM if x not in veh_params))))

        # --------------------------------------------------------------------------------------------------------------
        # INIT LOGGING -------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        self.__fp_log_data = None
        if logging_param is not None and logging_param['log_path_data'] is not None:
            self.__fp_log_data = logging_param['log_path_data']
            with open(self.__fp_log_data, "w+") as fh:
                # write vehicle params in form of comment
                data = json.dumps(self.__veh_params)
                fh.write("# " + data + "\n")

                # write header to logging file
                header = LOG_HEADER
                fh.write(header)

        self.__fp_log_map = None
        if logging_param is not None and logging_param['log_path_map'] is not None:
            self.__fp_log_map = logging_param['log_path_map']
            # write header to logging file
            with open(self.__fp_log_map, "w+") as fh:
                header = MAP_LOG_HEADER
                fh.write(header)

        # init logger
        self.__log = logging.getLogger("supervisor_logger")

        # Configure console output
        # normal - stdout
        hdlr = logging.StreamHandler(sys.stdout)
        hdlr.setFormatter(logging.Formatter('%(levelname)s [%(asctime)s]: %(message)s', '%H:%M:%S'))
        hdlr.addFilter(lambda record: record.levelno < logging.CRITICAL)
        hdlr.setLevel(os.environ.get("LOGLEVEL",
                                     "WARNING" if logging_param is None else logging_param['console_log_level']))
        self.__log.addHandler(hdlr)

        # error - stderr
        hdlr_e = logging.StreamHandler()
        hdlr_e.setFormatter(logging.Formatter('%(levelname)s [%(asctime)s]: %(message)s', '%H:%M:%S'))
        hdlr_e.setLevel(logging.CRITICAL)
        self.__log.addHandler(hdlr_e)

        # Configure file output
        if logging_param is not None and logging_param['log_path_msg'] is not None:
            with open(logging_param['log_path_msg'], "w+") as fh:
                header = "time;type;message\n"
                fh.write(header)

            fhdlr = logging.FileHandler(logging_param['log_path_msg'])
            fhdlr.setFormatter(logging.Formatter('%(created)s;%(levelname)s;%(message)s'))
            fhdlr.setLevel(os.environ.get("LOGLEVEL", logging_param['file_log_level']))
            self.__log.addHandler(fhdlr)

        # set the global logger level (should be the lowest of all individual streams --> leave at DEBUG!)
        self.__log.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # CLASS METHODS ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def set_environment(self,
                        bound_left: np.ndarray,
                        bound_right: np.ndarray,
                        ax_max_machines: np.ndarray,
                        localgg: np.ndarray):
        """
        Set map information including performance limits applicable on the given map.

        Depending on the enabled modules this triggers an (re)initialization that may take some time. For online update
        of maps, evaluate performance first.

        :param bound_left:      left track boundary coordinates as numpy array with columns [x, y]
        :param bound_right:     right track boundary coordinates as numpy array with columns [x, y]
        :param ax_max_machines: acceleration profile for the machine of the used vehicles, if multiple types use fastest
                                the acc. for a given velocity is extracted - provide numpy array with columns [v, a]
        :param localgg:         track specific acceleration limits as numpy array with columns [x, y, s_m, ax, ay]
                                NOTE: currently only the maximum of ax and ay is used globally (worst case estimate)
        """

        # receive map and other data
        self.__bound_l = bound_left
        self.__bound_r = bound_right
        self.__ax_max_machines = ax_max_machines
        self.__localgg = localgg

        # log and print (for docker compose)
        print("Supervisor received new map - triggering initialization")
        self.__log.info("New map provided.")

        # calculate reference-line
        if any(list(self.__module_enabled['dynamic_rule_reach_sets'].values())):
            self.__log.info("Calculating reference-line...")
            self.__ref_line, self.__norm_vec, self.__tw_left, self.__tw_right = stt.generate_refline.\
                generate_refline(bound_l=self.__bound_l,
                                 bound_r=self.__bound_r,
                                 resolution=1.0)
            self.__log.info("Calculation of reference-line done.")

        # initialize supervisor modules
        self.init_supmods()
        print("Supervisor finished initialization.")

        # log map, if logging enabled
        if self.__fp_log_map is not None:
            with open(self.__fp_log_map, "a") as fh:
                fh.write("\n"
                         + str(time.time()) + ";"
                         + json.dumps(self.__bound_l, default=default) + ";"
                         + json.dumps(self.__bound_r, default=default) + ";"
                         + json.dumps(self.__localgg, default=default) + ";"
                         + json.dumps(self.__ax_max_machines, default=default) + ";"
                         + json.dumps(self.__acc_limit_factor, default=default)
                         )

        return True

    def init_supmods(self) -> None:
        """
        Initialize all activated SUPervisor MODules.
        """

        # -- CALL SUPMOD INIT FUNCTIONS --------------------------------------------------------------------------------
        # dummy module
        if any(self.__module_enabled['static_dummy'].values()):
            self.__mod_dict['dummy'] = trajectory_supervisor.supervisor_modules.supmod_dummy.src.supmod_dummy.\
                SupModDummy()

        # static environment safety
        if any(self.__module_enabled['static_safe_end_state'].values()):
            self.__mod_dict['static_safe_end_state'] = trajectory_supervisor.supervisor_modules.supmod_safe_end_state.\
                src.supmod_safe_end_state.SupModSafeEndState()

        # static collision checks
        if any(self.__module_enabled['static_collision_check'].values()):
            self.__mod_dict['static_collision_check'] = trajectory_supervisor.supervisor_modules.\
                supmod_static_collision.src.supmod_static_collision.\
                SupModStaticCollision(supmod_config_path=self.__supmod_config_path,
                                      veh_params=self.__veh_params)

        # friction checks
        if any(self.__module_enabled['static_friction_ellipse'].values()):
            self.__mod_dict['static_friction_ellipse'] = trajectory_supervisor.supervisor_modules.supmod_friction.src.\
                supmod_friction.SupModFriction(supmod_config_path=self.__supmod_config_path,
                                               veh_params=self.__veh_params,
                                               localgg=self.__localgg)

        # kinematic and dynamic of ego vehicle
        if any(self.__module_enabled['static_kinematic_dynamic'].values()):
            self.__mod_dict['static_kinematic_dynamic'] = trajectory_supervisor.supervisor_modules.\
                supmod_kinematic_dynamic.src.supmod_kinematic_dynamic.\
                SupModKinematicDynamic(veh_params=self.__veh_params,
                                       ax_max_machines=self.__ax_max_machines,
                                       supmod_config_path=self.__supmod_config_path)

        # static rule evaluation
        if any(self.__module_enabled['static_rules'].values()):
            self.__mod_dict['static_rules'] = trajectory_supervisor.supervisor_modules.supmod_rules.src.supmod_rules.\
                SupModRules(supmod_config_path=self.__supmod_config_path)

        # static integrity checks
        if any(self.__module_enabled['static_integrity'].values()):
            self.__mod_dict['static_integrity'] = trajectory_supervisor.supervisor_modules.supmod_integrity.src.\
                supmod_integrity.SupModIntegrity(supmod_config_path=self.__supmod_config_path)

        # RSS
        if any(self.__module_enabled['dynamic_RSS'].values()):
            self.__mod_dict['RSS'] = trajectory_supervisor.supervisor_modules.supmod_RSS.src.supmod_RSS. \
                SupModRSS(supmod_config_path=self.__supmod_config_path,
                          veh_params=self.__veh_params,
                          localgg=self.__localgg)

        # guaranteed occ area
        if any(self.__module_enabled['dynamic_guar_occupation'].values()):
            self.__mod_dict['dynamic_guar_occupation'] = trajectory_supervisor.supervisor_modules.\
                supmod_guaranteed_occupancy_area.src.supmod_guaranteed_occupancy_area. \
                SupModGuaranteedOccupancyArea(localgg=self.__localgg,
                                              ax_max_machines=self.__ax_max_machines,
                                              supmod_config_path=self.__supmod_config_path,
                                              veh_params=self.__veh_params)

        # rule-based reachable sets
        if any(self.__module_enabled['dynamic_rule_reach_sets'].values()):
            self.__mod_dict['rule_reach_set'] = trajectory_supervisor.supervisor_modules.supmod_rule_reach_sets.src.\
                supmod_rule_reach_sets.SupModReachSets(supmod_config_path=self.__supmod_config_path,
                                                       veh_params=self.__veh_params,
                                                       a_max=np.max(self.__localgg[:, 3:5]))

        # -- PROVIDE ADDITIONAL INITIALIZATION DATA --------------------------------------------------------------------
        # provide map / reference-line to modules
        self.__log.info("Forwarding map to modules...")
        if 'RSS' in self.__mod_dict.keys():
            self.__mod_dict['RSS'].update_map(bound_l=self.__bound_l,
                                              bound_r=self.__bound_r)

        if 'static_collision_check' in self.__mod_dict.keys():
            self.__mod_dict['static_collision_check'].update_map(bound_l=self.__bound_l,
                                                                 bound_r=self.__bound_r)

        if 'rule_reach_set' in self.__mod_dict.keys():
            self.__mod_dict['rule_reach_set'].update_map(ref_line=self.__ref_line,
                                                         norm_vec=self.__norm_vec,
                                                         tw_left=self.__tw_left,
                                                         tw_right=self.__tw_right,
                                                         zone_file_path=self.__zone_file_path,
                                                         localgg=self.__localgg,
                                                         ax_max_machines=self.__ax_max_machines,
                                                         turn_radius=self.__veh_params['turn_rad'])

    # ------------------------------------------------------------------------------------------------------------------

    def set_inputs(self,
                   traj_perf: dict,
                   traj_em: dict,
                   objects: dict) -> None:
        """
        Set new input data.
        Note: if provided "None" instead of a dict, the previous value remains. However, the trajectories must always
              be provided in pairs.

        :param traj_perf:       performance trajectory in form of a dict with the following entries:
                                - 'traj':  trajectory data as numpy array with columns: s, x, y, head, curv, vel, acc
                                - 'id':    unique id of the trajectory (int)
                                - 'time':  time stamp of the trajectory in seconds
        :param traj_em:         emergency trajectory in form of a dict (same entries / format as traj_perf)
        :param objects:         object-list dict with each key being a dedicated object id and each object hosting a
                                dict with (at least) the following information:
                                - 'X': x position of cg
                                - 'Y': y position of cg
                                - 'theta': heading (north = 0, +pi -pi)
                                - 'v_x': x velocity
                                - 'type': "car", ...
                                - 'form': "rectangle" / "circle"
                                - 'width': width (if rectangle)
                                - 'length': length (if rectangle)
                                - 'diameter': diameter (if circle)
                                - 'time':  time-stamp

        """

        # -- init ratings and store trajectories -----------------------------------------------------------------------
        if traj_perf is not None and traj_em is not None:
            self.__traj['perf'] = traj_perf
            self.__traj['emerg'] = traj_em

            self.__log.info("Received new trajectory with (perf) stamp " + str(self.__traj['perf']['time']) + "!")

            # add new dict keys for safety rating
            self.__traj['perf']['valid'] = None
            self.__traj['perf']['valid_dyn'] = None
            self.__traj['perf']['valid_stat'] = None
            self.__traj['perf']['time_safety'] = None

            self.__traj['emerg']['valid'] = None
            self.__traj['emerg']['valid_dyn'] = None
            self.__traj['emerg']['valid_stat'] = None
            self.__traj['emerg']['time_safety'] = None

        # -- store objects ---------------------------------------------------------------------------------------------
        if objects is not None:
            self.__objects = objects

        return

    # ------------------------------------------------------------------------------------------------------------------

    def process_data(self) -> tuple:
        """
        Triggers the calculation of the safety metric with all activated supervisor modules. If parallelization is
        activated in the configuration, the safety score for the performance and emergency trajectory is each calculated
        in an separate thread.

        This method returns trajectory-dicts with embedded safety rating (key 'valid').

        :returns:
            * **traj_perf** -  performance trajectory in form of a dict with the following entries:
                                - 'traj':        trajectory data numpy array with columns: s, x, y, head, curv, vel, acc
                                - 'id':          unique id of the trajectory (int)
                                - 'time':        time stamp of the trajectory in seconds
                                - 'valid':       'True' if the trajectory is rated as safe, else 'False'
                                - 'valid_dyn':   'True' if rated as safe w.r.t. the dynamic environment, else 'False'
                                - 'valid_stat':  'True' if rated as safe w.r.t. the static environment, else 'False'
                                - 'time_safety': time.time() stamp at which safety score was concluded
            * **traj_em** -    emergency trajectory in form of a dict (same entries / format as traj_perf)

        .. note:: The returned values default to 'None', if no (new) data to process was handed to the class.
        """

        # check if map data is present, else raise value error
        if self.__bound_l is None or self.__bound_r is None:
            raise ValueError("Map initialization was not finished before first processing call!")

        # if no (new) trajectory present or no object coms, skip iteration
        if self.__traj['perf'] is None or self.__traj['perf']['valid'] is not None or self.__objects is None:
            self.__log.warning("Triggered processing step without defined data! Skipped processing iteration.")
            return None, None

        # --------------------------------------------------------------------------------------------------------------
        # - CALCULATE SCORE --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        tic = time.time()
        mp_queue_results = multiprocessing.Queue()
        mod_dict_queue_out = multiprocessing.Queue()

        # add modules dict twice to queue
        mod_dict_queue_in = multiprocessing.Queue()
        mod_dict_queue_in.put(self.__mod_dict)
        mod_dict_queue_in.put(self.__mod_dict)

        jobs = []
        pending_results = 0
        pending_mod_dicts = 0
        for traj_type in ['perf', 'emerg']:
            traj = self.__traj[traj_type]['traj']

            # use parallel calculation, when enabled and not windows
            # NOTE: windows does not support "fork" initialization and uses "spawn" which is slow
            if platform.system() == 'Linux' and self.__use_mp:
                # call trajectory clipping in parallel manner
                p = multiprocessing.Process(target=safety_rating,
                                            args=(mod_dict_queue_in, self.__module_enabled, traj, self.__objects,
                                                  traj_type, mod_dict_queue_out, mp_queue_results))
                jobs.append(p)
                p.start()
            else:
                # call trajectory clipping
                safety_rating(mod_dict_q_in=mod_dict_queue_in,
                              mod_enabled=self.__module_enabled,
                              traj=traj,
                              objects_sync=self.__objects,
                              traj_type=traj_type,
                              mod_dict_q_out=mod_dict_queue_out,
                              mp_queue=mp_queue_results)

            pending_results += 1
            pending_mod_dicts += 1

        # sync module dict (supervisor module classes)
        while pending_mod_dicts != 0 or not mod_dict_queue_in.empty():
            msg = mod_dict_queue_out.get(timeout=1.0)

            self.__mod_dict.update(msg)

            pending_mod_dicts -= 1

        mod_dict_queue_in.close()

        # extract results from queue
        self.__fired_modules = []
        self.__param_dict = dict()
        while pending_results != 0 or not mp_queue_results.empty():
            msg = mp_queue_results.get(timeout=0.1)

            for traj_type in msg['traj_dict'].keys():
                self.__traj[traj_type].update(msg['traj_dict'][traj_type])
                pending_results -= 1

            if "mod_calctime" in self.__param_dict and "mod_calctime" in msg['params_dict']:
                self.__param_dict['mod_calctime'].update(msg['params_dict']["mod_calctime"])
                del msg['params_dict']["mod_calctime"]

            self.__param_dict.update(msg['params_dict'])
            self.__fired_modules.extend(msg['fired_modules'])

        # extract calc-times
        for module in self.__param_dict['mod_calctime'].keys():
            if module not in self.__mod_calctime:
                self.__mod_calctime[module + "_count"] = 1
                self.__mod_calctime[module] = self.__param_dict['mod_calctime'][module]
            else:
                self.__mod_calctime[module + "_count"] += 1
                n = self.__mod_calctime[module + "_count"] - 1
                self.__mod_calctime[module] = (self.__mod_calctime[module] * n
                                               + self.__param_dict['mod_calctime'][module]) / (n + 1)

        mp_queue_results.close()

        # merge ratings of perf and emerg for log and return
        self.__valid_stat = self.__traj['perf']['valid_stat'] and self.__traj['emerg']['valid_stat']
        self.__valid_dyn = self.__traj['perf']['valid_dyn'] and self.__traj['emerg']['valid_dyn']

        # calculate and store overall execution time
        if self.__param_dict is not None:
            if "mod_calctime" not in self.__param_dict:
                self.__param_dict['mod_calctime'] = dict()
            self.__param_dict['mod_calctime']['overall'] = time.time() - tic

        # warn if calculation time exceeded threshold
        if time.time() - tic > self.__t_warn:
            self.__log.warning("Supvervisor | One iteration of the trajectory assessment took more than "
                               "%.3fs (actual %.3fs)!" % (self.__t_warn, time.time() - tic))

        return self.__traj['perf'], self.__traj['emerg']
    # ------------------------------------------------------------------------------------------------------------------

    def process_data_simple(self,
                            traj_perf: dict,
                            traj_em: dict,
                            objects: dict) -> tuple:
        """
        Calls internal methods "set_inputs()" and "process_data()" and returns simplified safety rating (boolean T/F for
        performance and emergency trajectory respectively.

        Note: if provided "None" instead of a dict, the previous value remains. However, the trajectories must always
              be provided in pairs.

        :param traj_perf:       performance trajectory in form of a dict with the following entries:
                                - 'traj':  trajectory data as numpy array with columns: s, x, y, head, curv, vel, acc
                                - 'id':    unique id of the trajectory (int)
                                - 'time':  time stamp of the trajectory in seconds
        :param traj_em:         emergency trajectory in form of a dict (same entries / format as traj_perf)
        :param objects:         object-list dict with each key being a dedicated object id and each object hosting a
                                dict with (at least) the following information:
                                - 'id': unique identifier
                                - 'X': x position of cg
                                - 'Y': y position of cg
                                - 'theta': heading (north = 0, +pi -pi)
                                - 'v_x': x velocity
                                - 'type': "car", ...
                                - 'form': "rectangle" / "circle"
                                - 'width': width (if rectangle)
                                - 'length': length (if rectangle)
                                - 'diameter': diameter (if circle)
                                - 'time':  time-stamp

        :returns:
            * **safety_perf** - 'True' if provided performance trajectory is safe, 'False' else - None if not provided
            * **safety_em** -   'True' if provided performance trajectory is safe, 'False' else - None if not provided

        .. note:: The returned values default to 'None', if no (new) data to process was handed to the class.
        """

        self.set_inputs(traj_perf=traj_perf,
                        traj_em=traj_em,
                        objects=objects)

        temp_traj_perf, temp_traj_em = self.process_data()

        return (temp_traj_perf['valid'] if temp_traj_perf is not None else None),\
               (temp_traj_em['valid'] if temp_traj_em is not None else None)

    # ------------------------------------------------------------------------------------------------------------------

    def get_safe_trajectory(self,
                            traj_perf: dict = None,
                            traj_em: dict = None) -> dict:
        """
        Selects a safety trajectory based on the provided new performance and emergency trajectory and a previously
        stored emergency trajectory. If in the first call no safe emergency trajectory is provided, an error is raised.

        .. note:: If no trajectory is provided, the last calculated internal set of trajectories is used

        :param traj_perf:       (optional) performance trajectory in form of a dict with the following entries:
                                - 'traj':  trajectory data as numpy array with columns: s, x, y, head, curv, vel, acc
                                - 'id':    unique id of the trajectory (int)
                                - 'time':  time stamp of the trajectory in seconds
                                - 'valid': boolean flag indicating safety
        :param traj_em:         (optional) emergency trajectory in form of a dict (same entries / format as traj_perf)
        :returns:
            * **safe_traj** -   safe trajectory to be executed (either performance or emergency) in form of dict
        """

        if traj_perf is None:
            traj_perf = self.__traj['perf']

        if traj_em is None:
            traj_em = self.__traj['emerg']

        # if new emergency trajectory is safe
        if traj_em['valid']:
            self.__safe_emerg_traj = dict(traj_em)

            # if new performance trajectory is safe -> return this as safe option
            if traj_perf['valid']:
                return traj_perf

        # (try to) load stored safe emergency trajectory
        if self.__safe_emerg_traj is None:
            raise ValueError("The first time step did not host a safe emergency trajectory!")

        return self.__safe_emerg_traj

    def get_fired_modules(self) -> list:
        """
        Return a list of strings for the modules fired in the preceding processing step.

        :return:
        """

        return self.__fired_modules

    def log(self,
            traj_perf_ref: np.ndarray = None,
            traj_em_ref: np.ndarray = None,
            objects_ref: dict = None) -> None:
        """
        Log relevant data to file (if parameterized).
        A reference trajectory (to the ones provided via "set_inputs") for the performance and emergency trajectory can
        be provided. This could be helpful when using the "sync_data" helper-function. That way one can log the synced
        as well as the original trajectory. The same scheme applies for a reference object-list

        :param traj_perf_ref:       (optional) reference trajectory for the internal performance trajectory
        :param traj_em_ref:         (optional) reference trajectory for the internal emergency trajectory
        :param objects_ref:         (optional) reference object-list for the internal object list
        :return:
        """

        if self.__fp_log_data is not None:
            # determine objects_stamp
            objects_stamp = time.time()
            if self.__objects:
                objects_stamp = list(self.__objects.values())[0]['time']

            # "time;traj_stamp;traj_perf;traj_perf_sync;traj_emerg;traj_emerg_sync;objects_stamp;objects_ref;"
            # "objects;safety_static;safety_dynamic;safety_base"
            # append data to file
            with open(self.__fp_log_data, "a") as fh:
                fh.write("\n"
                         + str(time.time()) + ";"
                         + str(self.__traj['perf']['time']) + ";"
                         + json.dumps(traj_perf_ref, default=default) + ";"
                         + json.dumps(self.__traj['perf']['traj'], default=default) + ";"
                         + json.dumps(traj_em_ref, default=default) + ";"
                         + json.dumps(self.__traj['emerg']['traj'], default=default) + ";"
                         + str(objects_stamp) + ";"
                         + json.dumps(objects_ref, default=default) + ";"
                         + json.dumps(self.__objects, default=default) + ";"
                         + json.dumps(self.__valid_stat, default=default) + ";"
                         + json.dumps(self.__valid_dyn, default=default) + ";"
                         + json.dumps(self.__param_dict, default=default))
        return

    # ------------------------------------------------------------------------------------------------------------------


def default(obj):
    # handle numpy arrays when converting to json
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError('Not serializable (type: ' + str(type(obj)) + ')')


def safety_rating(mod_dict_q_in: multiprocessing.Queue,
                  mod_enabled: dict,
                  traj: np.ndarray,
                  objects_sync: dict,
                  traj_type: str,
                  mod_dict_q_out: multiprocessing.Queue,
                  mp_queue: multiprocessing.Queue) -> None:
    """
    This function handles the generation of safety score for a given trajectory. Therefore, all activated supervisor
    modules for the trajectory type at hand ('`traj_type`') are called. The final safety rating is a conjunction of the
    returned scores of all passed supervisor modules.

    :param mod_dict_q_in:   queue holding dictionary with initialized module classes (used for calculations)
    :param mod_enabled:     dictionary specifying whether a module is activated for the 'perf' and / or 'emerg' traj.
    :param traj:            trajectory data with columns [s, x, y, heading, curv, vel, acc]
    :param objects_sync:    synced object dictionary
    :param traj_type:       string describing the trajectory type ('perf' or 'emerg')
    :param mod_dict_q_out:  queue holding dictionary with initialized module classes (updated with executed calculation)
    :param mp_queue:        queue that will receive all results

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        30.03.2020

    """

    tic = time.time()

    # extract module dict from queue
    mod_dict = mod_dict_q_in.get()

    # ----------------------------------------------------------------------------------------------------------
    # - CHECK DYNAMIC ENVIRONMENT (trajectory and objects must be present) -------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    # init dict used to return all generated data
    return_dict = {'traj_dict': {traj_type: dict()},
                   'params_dict': {'mod_calctime': dict()},
                   'fired_modules': []}

    # init, i.e. if no objects present, dynamic environment is safe
    return_dict['traj_dict'][traj_type]['valid_dyn'] = True

    # if objects in object-list
    if objects_sync:
        # - dummy module (example module - executed when activated and initialized) ------------------------------------
        if 'dummy' in mod_dict.keys() and mod_enabled['static_dummy'][traj_type]:
            valid_dummy = mod_dict['dummy'].calc_score()

            # append to list of fired modules if unsafe
            if not valid_dummy:
                return_dict['fired_modules'].append('static_dummy__' + traj_type)

        else:
            mod_dict.pop('dummy', None)
            valid_dummy = True

        # - RSS module -------------------------------------------------------------------------------------------------
        if 'RSS' in mod_dict.keys() and mod_enabled['dynamic_RSS'][traj_type]:
            tic_m = time.time()
            valid_rss, safety_params_rss = mod_dict['RSS'].calc_score(ego_pos=traj[0, 1:3],
                                                                      ego_heading=traj[0, 3],
                                                                      ego_curv=traj[0, 4],
                                                                      ego_vel=traj[0, 5],
                                                                      objects=objects_sync)

            # store params dict (append trajectory-type to all keys)
            return_dict['params_dict']['mod_calctime']['RSS_' + traj_type] = (time.time() - tic_m)
            return_dict['params_dict'].update({str(k) + '_' + traj_type: v for k, v in safety_params_rss.items()})

            # append to list of fired modules if unsafe
            if not valid_rss:
                return_dict['fired_modules'].append('dynamic_RSS__' + traj_type)
        else:
            mod_dict.pop('RSS', None)
            valid_rss = True

        # -- guaranteed occupation -------------------------------------------------------------------------------------
        if 'dynamic_guar_occupation' in mod_dict.keys() and mod_enabled['dynamic_guar_occupation'][traj_type]:
            tic_m = time.time()
            valid_gu_occ, safety_params_gu_occ = mod_dict['dynamic_guar_occupation'].calc_score(ego_traj=traj,
                                                                                                objects=objects_sync)

            # store params dict (append trajectory-type to all keys)
            return_dict['params_dict']['mod_calctime']['dynamic_guar_occupation_' + traj_type] = (time.time() - tic_m)
            return_dict['params_dict'].update({str(k) + '_' + traj_type: v for k, v in safety_params_gu_occ.items()})

            # append to list of fired modules if unsafe
            if not valid_gu_occ:
                return_dict['fired_modules'].append('dynamic_guar_occupation__' + traj_type)
        else:
            mod_dict.pop('dynamic_guar_occupation', None)
            valid_gu_occ = True

        # -- rule-based reachable sets ---------------------------------------------------------------------------------
        if 'rule_reach_set' in mod_dict.keys() and mod_enabled['dynamic_rule_reach_sets'][traj_type]:
            tic_m = time.time()
            valid_rule_rs, safety_params_rule_rs = mod_dict['rule_reach_set'].calc_score(traj=traj,
                                                                                         objects=objects_sync)

            # store params dict (append trajectory-type to all keys)
            return_dict['params_dict']['mod_calctime']['rule_reach_set_' + traj_type] = (time.time() - tic_m)
            return_dict['params_dict'].update({str(k) + '_' + traj_type: v for k, v in safety_params_rule_rs.items()})

            # append to list of fired modules if unsafe
            if not valid_rule_rs:
                return_dict['fired_modules'].append('dynamic_rule_reach_sets__' + traj_type)
        else:
            mod_dict.pop('rule_reach_set', None)
            valid_rule_rs = True

        # -- fuse score of all active dynamic env. assessment modules (add more via conjunction) -----------------------
        valid_dyn_env = valid_dummy and valid_rss and valid_gu_occ and valid_rule_rs

        # store to return dict
        return_dict['traj_dict'][traj_type]['valid_dyn'] = valid_dyn_env

    # ------------------------------------------------------------------------------------------------------------------
    # - CHECK STATIC ENVIRONMENT (only trajectory must be present) -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # -- static collision checks ---------------------------------------------------------------------------------------
    if 'static_collision_check' in mod_dict.keys() and mod_enabled['static_collision_check'][traj_type]:
        tic_m = time.time()
        valid_static_col, safety_params_stat_col = mod_dict['static_collision_check'].calc_score(ego_data=traj)

        # store params dict (append trajectory-type to all keys)
        return_dict['params_dict']['mod_calctime']['static_collision_' + traj_type] = (time.time() - tic_m)
        return_dict['params_dict'].update({str(k) + '_' + traj_type: v for k, v in safety_params_stat_col.items()})

        # append to list of fired modules if unsafe
        if not valid_static_col:
            return_dict['fired_modules'].append('static_collision_check__' + traj_type)
    else:
        mod_dict.pop('static_collision_check', None)
        valid_static_col = True

    # -- friction / acceleration ---------------------------------------------------------------------------------------
    if 'static_friction_ellipse' in mod_dict.keys() and mod_enabled['static_friction_ellipse'][traj_type]:
        tic_m = time.time()
        valid_friction, safety_params_friction = mod_dict['static_friction_ellipse'].calc_score(ego_data=traj)

        # store params dict (append trajectory-type to all keys)
        return_dict['params_dict']['mod_calctime']['static_friction_ellipse_' + traj_type] = (time.time() - tic_m)
        return_dict['params_dict'].update({str(k) + '_' + traj_type: v for k, v in safety_params_friction.items()})

        # append to list of fired modules if unsafe
        if not valid_friction:
            return_dict['fired_modules'].append('static_friction_ellipse__' + traj_type)
    else:
        mod_dict.pop('static_friction_ellipse', None)
        valid_friction = True

    # -- vehicle kinematics and dynamics -------------------------------------------------------------------------------
    if 'static_kinematic_dynamic' in mod_dict.keys() and mod_enabled['static_kinematic_dynamic'][traj_type]:
        tic_m = time.time()
        valid_kin_dyn, safety_params_kin_dyn = mod_dict['static_kinematic_dynamic'].calc_score(ego_data=traj)

        # store params dict (append trajectory-type to all keys)
        return_dict['params_dict']['mod_calctime']['static_kinematic_dynamic_' + traj_type] = (time.time() - tic_m)
        return_dict['params_dict'].update({str(k) + '_' + traj_type: v for k, v in safety_params_kin_dyn.items()})

        # append to list of fired modules if unsafe
        if not valid_kin_dyn:
            return_dict['fired_modules'].append('static_kinematic_dynamic__' + traj_type)
    else:
        mod_dict.pop('static_kinematic_dynamic', None)
        valid_kin_dyn = True

    # -- safe end state ------------------------------------------------------------------------------------------------
    if 'static_safe_end_state' in mod_dict.keys() and mod_enabled['static_safe_end_state'][traj_type]:
        tic_m = time.time()
        valid_safe_end_state, safety_param_safe_end_state = mod_dict['static_safe_end_state'].calc_score(ego_data=traj)

        # store params dict (append trajectory-type to all keys)
        return_dict['params_dict']['mod_calctime']['static_safe_end_state_' + traj_type] = (time.time() - tic_m)
        return_dict['params_dict'].update({str(k) + '_' + traj_type: v for k, v in safety_param_safe_end_state.items()})

        # append to list of fired modules if unsafe
        if not valid_safe_end_state:
            return_dict['fired_modules'].append('static_safe_end_state__' + traj_type)

    else:
        mod_dict.pop('static_safe_end_state', None)
        valid_safe_end_state = True

    # -- rules ---------------------------------------------------------------------------------------------------------
    if 'static_rules' in mod_dict.keys() and mod_enabled['static_rules'][traj_type]:
        tic_m = time.time()
        valid_rules, safety_params_rules = mod_dict['static_rules'].calc_score(ego_data=traj)

        # store params dict (append trajectory-type to all keys)
        return_dict['params_dict']['mod_calctime']['static_rules_' + traj_type] = (time.time() - tic_m)
        return_dict['params_dict'].update({str(k) + '_' + traj_type: v for k, v in safety_params_rules.items()})

        # append to list of fired modules if unsafe
        if not valid_rules:
            return_dict['fired_modules'].append('static_rules__' + traj_type)

    else:
        mod_dict.pop('static_rules', None)
        valid_rules = True

    # -- integrity -----------------------------------------------------------------------------------------------------
    if 'static_integrity' in mod_dict.keys() and mod_enabled['static_integrity'][traj_type]:
        tic_m = time.time()
        valid_integrity = mod_dict['static_integrity'].calc_score(traj=traj)

        # store params dict (append trajectory-type to all keys)
        return_dict['params_dict']['mod_calctime']['static_integrity_' + traj_type] = (time.time() - tic_m)

        # append to list of fired modules if unsafe
        if not valid_integrity:
            return_dict['fired_modules'].append('static_integrity__' + traj_type)

    else:
        mod_dict.pop('static_integrity', None)
        valid_integrity = True

    # -- fuse score of all active static env. assessment modules (add more via conjunction) ----------------------------
    valid_static_env = (valid_static_col and valid_friction and valid_kin_dyn and valid_safe_end_state and valid_rules
                        and valid_integrity)

    # store to return dict
    return_dict['traj_dict'][traj_type]['valid_stat'] = valid_static_env

    # ------------------------------------------------------------------------------------------------------------------
    # set global safety rating and safety score time-stamp
    return_dict['traj_dict'][traj_type]['valid'] = (return_dict['traj_dict'][traj_type]['valid_dyn']
                                                    and return_dict['traj_dict'][traj_type]['valid_stat'])
    return_dict['traj_dict'][traj_type]['time_safety'] = time.time()
    return_dict['traj_dict'][traj_type]['calc_time'] = time.time() - tic

    # push dict to queue
    mp_queue.put(return_dict)

    # push modified mod_dict to queue
    mod_dict_q_out.put(mod_dict)

    # wait to ensure proper handling of queue
    time.sleep(0.00001)


# -- TESTING -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
