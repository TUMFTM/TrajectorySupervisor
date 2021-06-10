import numpy as np
import math
import os
import trajectory_planning_helpers as tph


def acceleration_profile(a_lat_max: float,
                         a_long_max: float,
                         velocity: float,
                         velocity_angle: float,
                         curvature: float,
                         acc_type: str) -> tuple:
    """
    This function deal with the allowed acceleration profile according to the current velocity and the
    curvature. The original rough profile has been imported in the function and it will be further refined with
    interpolation. The friction ellipse and the limit of the motor are utilized to decide the boundary of the
    acceleration and deceleration. Based on the input value "acc_type", the acceleration and deceleration can be output
    accordingly.

    :param a_lat_max:       lateral acceleration limit of the tire
    :param a_long_max:      longitudinal acceleration limit of the tire
    :param velocity:        the current velocity of the vehicle
    :param velocity_angle:  the angle of the current velocity
    :param curvature:       the 1/radius of the current path
    :param acc_type:        whether the longitudinal or the lateral acceleration('long', 'lat')
    :returns:
        * **acc_max** -     if acc_type == 'long': allowed max. long. acceleration at current vel; if acc_type == 'lat':
          allowed max. lat. acceleration to the left at current vel. and curv. (in Frenet-Frame)
        * **dec_max** -     if acc_type == 'long': allowed max. long. decceleration at curr. vel; if acc_type == 'lat':
          allowed max. lat. acceleration to the right at current vel. and curv. (in Frenet-Frame)

    :Authors:
        * Yujie Lian
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        06.07.2019

    """

    # friction ellipse equation: a_long^2/A_LONG_MAX^2 + a_lat^2/A_LAT_MAX^2 = 1

    # ------------------------------------------------------------------------------------------------------------------
    # - CALCULATE CENTRIFUGAL FORCE ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # the curvature value can be negative for right turn and left turn
    a_centri = abs(pow(velocity, 2) * curvature)
    if a_centri > a_lat_max:
        a_centri = a_lat_max

    # ------------------------------------------------------------------------------------------------------------------
    # - LONGITUDINAL ACCELERATION --------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    if acc_type == 'long':
        # import ax_max_machines
        module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        ax_max_machines = tph.import_veh_dyn_info.\
            import_veh_dyn_info(ax_max_machines_import_path=(module_path
                                                             + '/params/veh_dyn_info/ax_max_machines.csv'))[1]

        # interpolate motor acceleration based on current velocity
        a_acc_long_max_motor = np.interp(velocity, ax_max_machines[:, 0], ax_max_machines[:, 1])

        # -- calculate max. lon. acc. (compare tire and motor) ---------------------------------------------------------

        # check if the centrifugal force exceed the friction ellipse, this can happen when the path are not smooth
        if 1 - pow(a_centri, 2) / pow(a_lat_max, 2) >= 0:
            a_acc_long_max_tire = np.sqrt(pow(a_long_max, 2) * (1 - (pow(a_centri, 2) / pow(a_lat_max, 2))))
        else:
            # if the input curvature is too large (because of the noise), then just ignore the centrifugal force
            a_acc_long_max_tire = a_long_max

        # choose the smaller value of motor and tire acc., then projected to the lon. direction of the track
        a_acc_long_max = min(a_acc_long_max_tire, a_acc_long_max_motor) * math.cos(velocity_angle)

        # -- calculate max. lon. deceleration (tire only) --------------------------------------------------------------

        # check if the centrifugal force exceed the friction ellipse, this can happen when the path are not smooth
        if 1 - pow(a_centri, 2) / pow(a_lat_max, 2) >= 0:
            a_dec_long_max = (np.sqrt(pow(a_long_max, 2) * (1 - (pow(a_centri, 2) / pow(a_lat_max, 2))))
                              * math.cos(velocity_angle))
        else:
            # if the input curvature is too large(because of the noise), then just ignore the centrifugal force
            a_dec_long_max = a_long_max * math.cos(velocity_angle)

        return a_acc_long_max, a_dec_long_max

    # ------------------------------------------------------------------------------------------------------------------
    # - LATERAL ACCELERATION --------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    elif acc_type == 'lat':
        if curvature <= 0:
            # situation: vehicle is making a right turn (if the curvature is negative)
            # -> centrifugal force consuming acc. potential to the right -> addition should not exceed friction ellipse
            #    (assuming the lon. acc. is zero) and then project the acceleration to the lateral direction (Frenet)
            a_acc_right_max = abs((a_lat_max - a_centri) * math.cos(velocity_angle))
            a_acc_left_max = abs(a_lat_max * math.cos(velocity_angle))

        else:
            # situation: vehicle is making a right turn (if the curvature is positive)
            # ->  centrifugal force consuming acc. potential to the left -> addition should not exceed friction ellipse
            #     (assuming the lon. acc. is zero) and then project the acceleration to lateral direction (Frenet)
            a_acc_right_max = abs(a_lat_max * math.cos(velocity_angle))
            a_acc_left_max = abs((a_lat_max - a_centri) * math.cos(velocity_angle))

        return a_acc_left_max, a_acc_right_max

    else:
        raise ValueError("Requested calculation acc_type '" + acc_type + "' not supported!")
