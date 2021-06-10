import logging
import numpy as np


def friction_check(ego_data: np.ndarray,
                   a_lon_max_tires: float,
                   a_lat_max_tires: float,
                   dyn_model_exp: float = 2.0,
                   drag_coeff: float = 0.0,
                   m_veh: float = 1000.0) -> tuple:
    """
    This function asses the trajectory safety for a given time instance regarding friction limit.

    :param ego_data:              data of the ego vehicle(s, pos_x, pos_y, heading, curvature, velocity, acceleration)
                                  for a given time instance
    :param a_lon_max_tires:       maximal allowed longitudinal acceleration the ego vehicle can transfer via the tires
    :param a_lat_max_tires:       maximal allowed lateral acceleration of the ego vehicle can transfer via the tires
    :param dyn_model_exp:         exponent used in the vehicle dynamics model (usual range [1.0, 2.0])
                                  NOTE: 2.0 represents a ideal friction circle; 1.0 a perfect diamond shape
    :param drag_coeff:            drag coefficient incl. all constants (0.5 * c_w * A_front * rho_air) [in m2*kg/m3]
                                  NOTE: set to zero, in order to disable drag consideration
    :param m_veh:                 vehicle mass (required for drag calculation) [in kg]
    :returns:
        * **safety** -            boolean value - 'False' for unsafe, 'True' for safe
        * **safety_parameters** - parameter dict (here: currently used acceleration and acceleration limit at value in
          trajectory, at position where the requested acceleration is closest to the bounds)

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>
        * Maroua Ben Lakhal

    :Created on:
        30.07.2019

    """

    # init safety score
    safety = True

    # ------------------------------------------------------------------------------------------------------------------
    # -- GET USED / REQUESTED ACCELERATION VALUES ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # -- for each point on the planed trajectory, extract curvature, velocity and longitudinal acceleration ------------
    ego_curve = ego_data[:, 4]
    ego_velocity = ego_data[:, 5]
    a_lon_used = np.copy(ego_data[:, 6])

    # -- for each point on the planned trajectory, calculate the lateral acceleration based on curvature and velocity
    a_lat_used = np.power(ego_velocity[:], 2) * ego_curve[:]

    # ------------------------------------------------------------------------------------------------------------------
    # -- ADD DRAG INFLUENCE --------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # calculate equivalent longitudinal acceleration of drag force along velocity profile
    a_lon_drag = np.power(ego_velocity[:], 2) * drag_coeff / m_veh

    # drag reduces requested deceleration but increases requested acceleration at the tire
    a_lon_used += a_lon_drag

    # ------------------------------------------------------------------------------------------------------------------
    # -- CHECK IF RESULTING REQUIRED ACCELERATION IS WITHIN LIMITS -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get used percentage of allowed tire acceleration
    a_comb_used_perc = np.power((np.power(np.abs(a_lon_used) / a_lon_max_tires, dyn_model_exp)
                                 + np.power(np.abs(a_lat_used) / a_lat_max_tires, dyn_model_exp)),
                                1.0 / dyn_model_exp)

    # calculate used combined acceleration (for logging purposes)
    a_comb_used = np.power(np.power(np.abs(a_lon_used), dyn_model_exp) + np.power(np.abs(a_lat_used), dyn_model_exp),
                           1.0 / dyn_model_exp)

    # ------------------------------------------------------------------------------------------------------------------
    # -- WARNING IF VIOLATED -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    if any(a_comb_used_perc > 1.0):
        safety = False

        # get index of violation
        i = np.argmax(a_comb_used_perc > 1.0)

        log = logging.getLogger("supervisor_logger")
        log.warning('supmod_friction | Friction limit exceeded! Requested acceleration at tires ({:.2f}m/s2, '
                    'including {:.2f}m/s2 due to drag) at ({:.2f}m, {:.2f}m) is {:.1f}% above the allowed limit'
                    '.'.format(a_comb_used[np.argmax(a_comb_used_perc)], a_lon_drag[np.argmax(a_comb_used_perc)],
                               ego_data[i, 1], ego_data[i, 2],
                               (a_comb_used_perc[np.argmax(a_comb_used_perc)] - 1) * 100.0))

    # assemble safety parameters dict (current value and bound at location of value closest to bound)
    safety_parameters = {"a_lon_used": a_lon_used,
                         "a_lat_used": a_lat_used,
                         "a_comb_used_perc": a_comb_used_perc,
                         "a_comb_cur": a_comb_used[np.argmax(a_comb_used_perc)],
                         "a_comb_bound": a_comb_used[np.argmax(a_comb_used_perc)] / (np.max(a_comb_used_perc) + 0.01)
                         }

    return safety, safety_parameters
