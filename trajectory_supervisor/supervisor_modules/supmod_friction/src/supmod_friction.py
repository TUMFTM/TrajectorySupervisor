import numpy as np
import configparser
import trajectory_supervisor


class SupModFriction(object):
    """
    Class handling safety checks regarding given friction / acceleration limits.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>
        * Maroua Ben Lakhal

    :Created on:
        14.11.2019

    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 supmod_config_path: str,
                 veh_params: dict,
                 localgg: np.ndarray):
        """
        Init the SupMod.

        :param supmod_config_path:  path to Supervisor config file
        :param veh_params:          dict of vehicle parameters; must hold the following keys:
                                      dyn_model_exp - exponent used in vehicle dynamics model (usual range [1.0, 2.0])
                                                      NOTE: 2.0 -> ideal friction circle; 1.0 -> perfect diamond shape
                                      drag_coeff -    drag coeff. incl. all constants (0.5 * c_w * A_front * rho_air)
                                                      set zero to disable [in m2*kg/m3]
                                      m_veh -         vehicle mass [in kg]
        :param localgg:             track specific acceleration limits as numpy array with columns [x, y, s_m, ax, ay]
                                    NOTE: currently only the maximum of ax and ay is used globally (worst case estimate)

        """

        # read configuration file
        safety_param = configparser.ConfigParser()
        if not safety_param.read(supmod_config_path):
            raise ValueError('Specified config file does not exist or is empty!')

        # -- get parameters for the static safety assessment from configuration file -----------------------------------
        self.__long_acc_max = max(localgg[:, 3]) * safety_param.getfloat('FRICTION', 'allowed_acc')
        self.__lat_acc_max = max(localgg[:, 4]) * safety_param.getfloat('FRICTION', 'allowed_acc')

        self.__dyn_model_exp = veh_params['dyn_model_exp']
        self.__drag_coeff = veh_params['drag_coeff']
        self.__m_veh = veh_params['m_veh']

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
        Asses the trajectory safety for a given time instance regarding given acceleration limits
        :param ego_data:        data of the ego vehicle(s, pos_x, pos_y, heading, curvature, velocity, acceleration)
                                for a given time instance
        :returns:
            * **safety** -            binary value indicating the safety state. 'False' = unsafe and 'True' = safe
            * **safety_parameters** - parameter dict

        """

        # -- check safety regarding friction limit ---------------------------------------------------------------------
        safety, safety_parameters = trajectory_supervisor.supervisor_modules.supmod_friction.src.check_friction.\
            friction_check(ego_data=ego_data,
                           a_lon_max_tires=self.__long_acc_max,
                           a_lat_max_tires=self.__lat_acc_max,
                           dyn_model_exp=self.__dyn_model_exp,
                           drag_coeff=self.__drag_coeff,
                           m_veh=self.__m_veh)

        return safety, safety_parameters
