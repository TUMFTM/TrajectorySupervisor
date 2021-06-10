import numpy as np
import logging
import json
import configparser
import trajectory_planning_helpers as tph


class SupModIntegrity(object):
    """
    Sample supervisor module class, holding the basic layout. In order to set up a new supervisor module, simply copy
    this folder ('`supmod_dummy`'), change all occurrences of 'dummy' to your module name and add your code to the
    function skeleton.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        31.05.2021

    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 supmod_config_path: str) -> None:
        """
        Init the rule-based reachable set SupMod.

        :param supmod_config_path:  path to Supervisor config file

        """

        self.__log = logging.getLogger('supervisor_logger')

        # -- read configuration file -----------------------------------------------------------------------------------
        param = configparser.ConfigParser()
        if not param.read(supmod_config_path):
            raise ValueError('Specified config file does not exist or is empty!')

        self.__basic_s_max = param.getfloat('INTEGRITY', 's_max')
        self.__basic_head_int = json.loads(param.get('INTEGRITY', 'head_int'))
        self.__basic_curv_int = json.loads(param.get('INTEGRITY', 'curv_int'))
        self.__basic_vel_int = json.loads(param.get('INTEGRITY', 'vel_int'))
        self.__basic_acc_int = json.loads(param.get('INTEGRITY', 'acc_int'))
        self.__s_err = param.getfloat('INTEGRITY', 's_err')
        self.__head_err = param.getfloat('INTEGRITY', 'head_err')
        self.__curv_err = param.getfloat('INTEGRITY', 'curv_err')
        self.__ax_err = param.getfloat('INTEGRITY', 'ax_err')

    # ------------------------------------------------------------------------------------------------------------------
    # DESTRUCTOR -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __del__(self) -> None:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # CALC SCORE -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def calc_score(self,
                   traj: np.ndarray) -> bool:
        """
        Calculate safety score (boolean value) based on integrity checks.

        This module checks whether the values specified in the trajectory match each other. In this sense, the following
        checks are performed:
        * basic data checks (format of provided trajectory, physical range of values)
        * s-coordinate matches the x-y-coordinates - calculation and adjustment of the distances between the points
        * heading matches the x-y coordinates - rough adjustment of the orientation given by the points
        * curvature based on angle change over the given distance (s-coordinate)
        * acceleration based on velocity data and s-coordinate


        :param traj:                        trajectory of the ego-veh. with the following columns:
                                            [s, pos_x, pos_y, heading, curvature, velocity, acceleration]
        :returns:
            * **safety** -                  safety score based on integrity. True = safe, False = possibly not safe
            * **safety_params** -           parameter dict

        """

        # init safety score to true (safe)
        safety = True

        # -- basic checks (data shape, range of values) ----------------------------------------------------------------
        # check for proper amount of columns
        if traj.shape[1] != 7:
            raise ValueError("Expected trajectory array with 7 columns, but got " + str(traj.shape[1]) + "!")

        # check s-coordinate (untypical large jumps / negative s-coord)
        if any(np.diff(traj[:, 0]) > self.__basic_s_max):
            self.__log.warning("supmod_integrity | Faced a jump of >30.0 m in the s-coordinate of the ego trajectory!")
            safety = False
        elif any(np.diff(traj[:, 0]) < 0.0):
            self.__log.warning("supmod_integrity | Faced a negative the s-coordinate gradient in the ego trajectory!")
            safety = False

        # check heading (values not within specified range)
        if any(np.less(traj[:, 3], self.__basic_head_int[0])) or any(np.greater(traj[:, 3], self.__basic_head_int[1])):
            self.__log.warning("supmod_integrity | Faced a heading value exceeding the specified range!")
            safety = False

        # check curvature (values not within specified range)
        if any(np.less(traj[:, 4], self.__basic_curv_int[0])) or any(np.greater(traj[:, 4], self.__basic_curv_int[1])):
            self.__log.warning("supmod_integrity | Faced a curvature value exceeding the specified range!")
            safety = False

        # check velocity (values not within specified range)
        if any(np.less(traj[:, 5], self.__basic_vel_int[0])) or any(np.greater(traj[:, 5], self.__basic_vel_int[1])):
            self.__log.warning("supmod_integrity | Faced a velocity value exceeding the specified range!")
            safety = False

        # check acceleration (values not within specified range)
        if any(np.less(traj[:, 6], self.__basic_acc_int[0])) or any(np.greater(traj[:, 6], self.__basic_acc_int[1])):
            self.__log.warning("supmod_integrity | Faced an acceleration value exceeding the specified range!")
            safety = False

        # -- check s-coordinate ----------------------------------------------------------------------------------------
        el_length = np.sqrt(np.sum(np.diff(traj[:, 1:3], axis=0) ** 2, axis=1))
        s_ref = np.insert(np.cumsum(el_length), 0, 0.0)

        ba = np.greater(np.abs(np.divide(traj[:, 0], s_ref, out=np.ones(traj.shape[0]), where=s_ref > 0.1) - 1.0),
                        self.__s_err)
        if any(ba):
            i = int(np.argmax(ba))
            self.__log.warning("supmod_integrity | The calculated s-coordinate reference exceeded the spec. tolerance! "
                               "E.g. %.3f in trajectory and %.3f in reference (index %d)" % (traj[i, 0], s_ref[i], i))
            safety = False

        # -- check heading ---------------------------------------------------------------------------------------------
        head, _ = tph.calc_head_curv_num.calc_head_curv_num(path=traj[:, 1:3],
                                                            el_lengths=el_length,
                                                            is_closed=False,
                                                            calc_curv=False)

        # calculate angle difference
        # exclude first and last value in check, since previous and following conditions are not known
        ang_err = np.minimum(np.abs(traj[:, 3] - head), 2 * np.pi - np.abs(traj[:, 3] - head))
        ba = np.greater(ang_err[1:-1], self.__head_err)
        if any(ba):
            i = int(np.argmax(ba)) + 1
            self.__log.warning("supmod_integrity | The calculated heading reference exceeded the spec. tolerance! "
                               "E.g. %.3f in trajectory and %.3f in reference (index %d)" % (traj[i, 3], head[i], i))
            safety = False

        # -- check curvature -------------------------------------------------------------------------------------------
        # calculate curvature based on trajectory heading and s-coord (excluding first and last point)
        delta_head = tph.normalize_psi.normalize_psi(traj[2:, 3] - traj[:-2, 3])
        curv = delta_head / (np.diff(traj[:, 0])[1:] + np.diff(traj[:, 0])[:-1])

        # exclude first and last value in check, since previous and following conditions are not known
        ba = np.greater(np.abs(traj[1:-1, 4] - curv), self.__curv_err)
        if any(ba):
            i = int(np.argmax(ba)) + 1
            j = i - 1
            self.__log.warning("supmod_integrity | The calculated curvature reference exceeded the spec. tolerance! "
                               "E.g. %.3f in trajectory and %.3f in reference (index %d)" % (traj[i, 4], curv[j], i))
            safety = False

        # -- check acceleration ----------------------------------------------------------------------------------------
        # get element length of trajectory
        el_len_traj = np.diff(traj[:, 0])

        # filter regions of no progress (s ~ 0)
        f_s = el_len_traj > 0.01
        f_l = np.concatenate((el_len_traj, [1.0])) > 0.01

        ax = tph.calc_ax_profile.calc_ax_profile(vx_profile=traj[f_l, 5],
                                                 el_lengths=el_len_traj[f_s])

        # exclude first and last value in check, since previous and following conditions are not known
        ba = np.greater(np.abs(traj[f_l, 6][1:-1] - ax[1:]), self.__ax_err)
        if any(ba):
            i = int(np.argmax(ba)) + 1
            self.__log.warning("supmod_integrity | The calculated acceleration reference exceeded the spec. tolerance! "
                               "E.g. %.3f in trajectory and %.3f in reference (index %d)" % (traj[f_l, 6][i], ax[i], i))
            safety = False

        return safety
