import numpy as np
import logging
import configparser


class SupModKinematicDynamic(object):
    """
    Class handling safety checks regarding kinematic and/or dynamic properties of the ego vehicle.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        04.09.2020

    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 veh_params: dict,
                 ax_max_machines: np.ndarray,
                 supmod_config_path: str):
        """
        Init the SupMod.

        :param veh_params:          dict of vehicle parameters; must hold the following key:
                                      turn_rad -      turning radius of vehicle [in m]
                                      drag_coeff -    drag coeff. incl. all constants (0.5 * c_w * A_front * rho_air)
                                                      set zero to disable [in m2*kg/m3]
                                      m_veh -         vehicle mass [in kg]
        :param ax_max_machines:     long. acceleration limits by the electrical motors, columns [vx, ax_max_machines];
                                    velocity in m/s, accelerations in m/s2. They should be handed in without considering
                                    drag resistance, i.e. simply by calculating F_x_drivetrain / m_veh
        :param supmod_config_path:  path to Supervisor config file
        """

        # check shape of ax_max_machines
        if ax_max_machines.shape[1] != 2:
            raise RuntimeError("ax_max_machines must consist of the two columns [vx, ax_max_machines]!")

        self.__log = logging.getLogger("supervisor_logger")
        self.__turn_rad = veh_params['turn_rad']
        self.__drag_coeff = veh_params['drag_coeff']
        self.__m_veh = veh_params['m_veh']
        self.__ax_max_machines = ax_max_machines

        # read configuration file
        safety_param = configparser.ConfigParser()
        if not safety_param.read(supmod_config_path):
            raise ValueError('Specified config file does not exist or is empty!')

        # -- get parameters for the static safety assessment from configuration file -----------------------------------
        self.__enable_motor_limits = safety_param.getboolean('KINEMATIC_DYNAMIC', 'enable_motor_limits')
        self.__allowed_acc = safety_param.getfloat('KINEMATIC_DYNAMIC', 'allowed_acc')

    # ------------------------------------------------------------------------------------------------------------------
    # DESTRUCTOR -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __del__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # CALC SCORE -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def calc_score(self,
                   ego_data: np.ndarray) -> tuple:
        """
        Asses the trajectory safety for a given time instance regarding kinematic and/or dynamic properties of the
        ego vehicle.

        :param ego_data:        data of the ego vehicle (s, pos_x, pos_y, heading, curvature, velocity, acceleration)
                                for a given time instance
        :returns:
            * **safety** -            binary value indicating the safety state. 'False' = unsafe and 'True' = safe
            * **safety_parameters** - parameter dict

        """

        # init safety score and params
        safety = True
        safety_params = dict()

        # check if turn radius is violated at any point in trajectory
        if any(abs(ego_data[:, 4]) > (1 / self.__turn_rad)):
            safety = False

            self.__log.warning(
                'supmod_kinematic_dynamic | Curvature of ego trajectory violates turn radius! E.g. at position ('
                + str(ego_data[np.where(abs(ego_data[:, 4]) > (1 / self.__turn_rad))[0][0], 1:3]) + ').')

            # extract indexes violating turn radius
            safety_params["stat_idx_turn_rad_err"] = np.where(abs(ego_data[:, 4]) > (1 / self.__turn_rad))[0]

        # check if requested acceleration is below machine limit
        if self.__enable_motor_limits:
            # get true ax for regions with positive velocity (reversing and standstill may be handled differently)
            ax_cur = np.where(ego_data[:, 5] > 0.01, np.copy(ego_data[:, 6]), 0.0)
            ax_cur += np.power(ego_data[:, 5], 2) * self.__drag_coeff / self.__m_veh

            ax_max_lim = np.interp(ego_data[:, 5], self.__ax_max_machines[:, 0], self.__ax_max_machines[:, 1])

            if any(np.greater(ax_cur, ax_max_lim * self.__allowed_acc)):
                safety = False

                idx = np.where(np.greater(ax_cur, ax_max_lim))[0][0]
                self.__log.warning('supmod_kinematic_dynamic | Requested acceleration (incl. drag) exceeds machine '
                                   'limits! E.g. {:.2f}m/s2 while the limit is at {:.2f}m/s2 (for v={:.2f}m/s)'
                                   '.'.format(ax_cur[idx], ax_max_lim[idx], ego_data[idx, 5]))

                # extract indexes violating acceleration limits
                safety_params["stat_idx_acc_machine"] = np.where(np.greater(ax_cur, ax_max_lim))[0]

        return safety, safety_params
