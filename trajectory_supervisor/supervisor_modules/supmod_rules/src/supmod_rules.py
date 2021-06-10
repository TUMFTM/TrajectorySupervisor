import numpy as np
import logging
import configparser


class SupModRules(object):
    """
    Class handling safety checks regarding applicable rules for the ego-vehicle.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        12.08.2020

    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 supmod_config_path: str):
        """
        Init the SupMod.

        :param supmod_config_path:  path to Supervisor config file
        """

        self.__log = logging.getLogger("supervisor_logger")

        # read configuration file
        safety_param = configparser.ConfigParser()
        if not safety_param.read(supmod_config_path):
            raise ValueError('Specified config file does not exist or is empty!')

        # -- get parameters for the static safety assessment from configuration file -----------------------------------
        self.__v_max = safety_param.getfloat('RULES', 'v_max')
        self.__a_max_dec = safety_param.getfloat('RULES', 'a_max_dec')

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
        Asses whether the given trajectory adheres to formalized rules (e.g. maximum velocity).

        :param ego_data:        data of the ego vehicle(s, pos_x, pos_y, heading, curvature, velocity, acceleration)
                                for a given time instance
        :returns:
            * **safety** -            binary value indicating the safety state. 'False' = unsafe and 'True' = safe
            * **safety_parameter** -  parameter dict

        """

        # -- check safety regarding obstacle collision with wall bounds or static obstacles ----------------------------
        # init safety score
        safety = True

        # check if maximum velocity is violated
        if max(ego_data[:, 5]) > self.__v_max:
            safety = False

            self.__log.warning('supmod_rules | Maximum velocity violated! The velocity ({:.2f}m/s) '
                               'exceeds v_max = {:.2f}m/s.'.format(max(ego_data[:, 5]), self.__v_max))

        # check if reversing
        if min(ego_data[:, 5]) < 0.0:
            safety = False

            self.__log.warning('supmod_rules | Requested negative acceleration! The velocity ({:.2f}m/s) '
                               'is below 0.0m/s.'.format(min(ego_data[:, 5])))

        # check if maximum deceleration is violated
        if min(ego_data[:, 6]) < self.__a_max_dec:
            safety = False

            self.__log.warning('supmod_rules | Maximum deceleration violated! The acceleration ({:.2f}m/s) '
                               'exceeds a_max = {:.2f}m/s.'.format(min(ego_data[:, 6]), self.__a_max_dec))

        safety_parameter = {"stat_rules_cur": safety}

        return safety, safety_parameter
