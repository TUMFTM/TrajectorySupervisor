import numpy as np
import logging


class SupModSafeEndState(object):
    """
    Class handling safety checks regarding a safe end state for the given trajectory.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        12.08.2020

    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        """
        Init the SupMod.
        """

        self.__log = logging.getLogger("supervisor_logger")

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
        Asses the trajectory safety for a given time instance  if the end state of a trajectory is safe
        (here: v_end=0.0, other constraints (in track, valid acc. profile, ...) are checked by other modules.

        :param ego_data:        data of the ego vehicle(s, pos_x, pos_y, heading, curvature, velocity, acceleration)
                                for a given time instance
        :returns:
            * **safety** -            binary value indicating the safety state. 'False' = unsafe and 'True' = safe
            * **safety_parameter** -  parameter dict

        """

        # -- check safety regarding obstacle collision with wall bounds or static obstacles ----------------------------
        # init safety score
        safety = True

        # check if end state is safe (here: v_end=0.0)
        if abs(ego_data[-1, 5]) > 0.01:
            safety = False

            self.__log.warning('supmod_safe_end_state | End state of trajectory is not safe! The velocity ({:.2f}m/s) '
                               'differs from 0.0m/s.'.format(ego_data[-1, 5]))

        safety_parameter = {"safe_end_state_cur": safety}

        return safety, safety_parameter
