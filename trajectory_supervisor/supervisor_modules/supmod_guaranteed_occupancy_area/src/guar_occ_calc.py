import os
import json
import numpy as np
import shapely.geometry
from functools import partial
import trajectory_planning_helpers as tph

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# TUM Colors
TUM_colors = {
    'TUM_blue': '#3070b3',
    'TUM_blue_dark': '#003359',
    'TUM_blue_medium': '#64A0C8',
    'TUM_blue_light': '#98C6EA',
    'TUM_grey_dark': '#9a9a9a',
    'TUM_orange': '#E37222',
    'TUM_green': '#A2AD00'
}


def guar_occ_calc(t_max: float,
                  d_t: float,
                  v_max: float,
                  d_v: float,
                  localgg: np.ndarray,
                  ax_max_mach: np.ndarray,
                  nmb_states: int = 101,
                  veh_length: float = 4.7,
                  veh_width: float = 2.0,
                  turn_rad: float = 11.0,
                  md5_key: str = '',
                  export_path: str = None) -> None:

    """
    This function calculates and exports (to file) the guaranteed occupation area for a given gg-diagram.

    :param t_max:       time-horizon the state set is computed for [in s]
    :param d_t:         temporal step-size in rendered set [in s]
    :param v_max:       maximum velocity of objects to be rendered in set [in m/s]
    :param d_v:         velocity step-size in rendered set [in m/s]
    :param localgg:     maximum g-g-values at a certain position on the map (currently, only the max values used)
    :param ax_max_mach: maximum acceleration the motors can provide (longitudinal)
    :param nmb_states:  number of reachable states, being computed to derive state set from. Creates quadruple of states
                        due to computation quadrant vice
    :param veh_length:  length of vehicle [in m]
    :param veh_width:   width of vehicle [in m]
    :param turn_rad:    turning radius of vehicle [in m]
    :param md5_key:     md5 key to be stored with the maps (e.g. md5-sum of parameter file)
    :param export_path: absolute path pointing to the location where the occupation set should be stored,
                        if set to 'None' an interactive visualization is shown

    :Authors:
        * Maximilian Bayerlein
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        20.01.2020
    """

    # number of temporal steps the reachable set is divided in
    nmb_t_steps = int(t_max / d_t)
    nmb_v_steps = int(v_max / d_v)

    export_dict = dict()
    plot_dict = dict()

    for v in range(nmb_v_steps + 1):
        "Initial data"
        v0 = v * d_v

        # --------------------------------------------------------------------------------------------------------------
        # -- CREATION OF REACHABLE-SET FOR COG -------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        reachable_set, phi = reachable_set_creator(t_r=t_max,
                                                   t_step=nmb_t_steps + 1,
                                                   number_of_states=nmb_states,
                                                   v0=v0,
                                                   turn_rad=turn_rad,
                                                   localgg=localgg,
                                                   acc_max=ax_max_mach)

        r_set_x = np.concatenate((reachable_set[:, :, 0, 0], reachable_set[:, :, 1, 0]), axis=0)
        r_set_y = np.concatenate((reachable_set[:, :, 0, 1], reachable_set[:, :, 1, 1]), axis=0)
        phi_set = np.concatenate((phi[:, :, 0], phi[:, :, 1]), axis=0)

        # --------------------------------------------------------------------------------------------------------------
        # -- OCCUPANCY-SET BASED ON 2D VEHICLE FOOTPRINT ---------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # turning of the front left corner of the vehicle
        x_fl = r_set_x + 0.5 * veh_length * np.cos(phi_set) + 0.5 * veh_width * np.sin(phi_set)
        y_fl = r_set_y + 0.5 * veh_length * np.sin(phi_set) - 0.5 * veh_width * np.cos(phi_set)
        # turning of the front right corner of the vehicle
        x_fr = r_set_x + 0.5 * veh_length * np.cos(phi_set) - 0.5 * veh_width * np.sin(phi_set)
        y_fr = r_set_y + 0.5 * veh_length * np.sin(phi_set) + 0.5 * veh_width * np.cos(phi_set)
        # turning of the rear left corner of the vehicle
        x_rl = r_set_x - 0.5 * veh_length * np.cos(phi_set) + 0.5 * veh_width * np.sin(phi_set)
        y_rl = r_set_y - 0.5 * veh_length * np.sin(phi_set) - 0.5 * veh_width * np.cos(phi_set)
        # turning of the rear right corner of the vehicle
        x_rr = r_set_x - 0.5 * veh_length * np.cos(phi_set) - 0.5 * veh_width * np.sin(phi_set)
        y_rr = r_set_y - 0.5 * veh_length * np.sin(phi_set) + 0.5 * veh_width * np.cos(phi_set)

        if export_path is not None:
            # generate sub dict for chosen velocity
            export_dict[v0] = dict()
        else:
            # generate sub dict for chosen velocity
            plot_dict[v0] = dict()

        # --------------------------------------------------------------------------------------------------------------
        # -- DEFINITELY OCCUPIED-SET -----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        for i in range(nmb_t_steps + 1):  # For every temporal step
            t = i * d_t

            # progressbar
            tph.progressbar.progressbar(1 + i + v * nmb_t_steps, (1 + nmb_v_steps) * (1 + nmb_t_steps),
                                        prefix="Simulating maneuvers ",
                                        suffix=" (currently calculating occupation area for v = %.2fm/s, t = %.2fs)"
                                               % (v0, t))

            # -- DERIVE STATE-SET --------------------------------------------------------------------------------------
            i_max_phi = np.argmax(phi_set[:, -1])         # Extract the index of the biggest angle of all states
            i_max_x = np.argmax(x_rl[:, -1])              # Extract the index of the biggest x-value of all states
            i_max_break = np.argmin(r_set_x[:, i])        # Extract the index of the state breaking the most
            i_max_acc = np.argmax(r_set_x[:, i])          # Extract the index of the state accelerating the most

            # -- CALCULATE 2D VEH-FOOTPRINTS BASED ON POSES ------------------------------------------------------------
            # rectangle defining the vehicle turning the most, left and right have the same x-value but opposite y-value
            rect_biggest_phi = [(x_rr[i_max_phi, i], y_rr[i_max_phi, i]),
                                (x_rl[i_max_phi, i], y_rl[i_max_phi, i]),
                                (x_fl[i_max_phi, i], y_fl[i_max_phi, i]),
                                (x_fr[i_max_phi, i], y_fr[i_max_phi, i]),
                                (x_rr[i_max_phi, i], y_rr[i_max_phi, i])]
            rect_biggest_phi = shapely.geometry.Polygon(rect_biggest_phi)

            rect_biggest_phi_neg = [(x_rr[i_max_phi, i], -y_rr[i_max_phi, i]),
                                    (x_rl[i_max_phi, i], -y_rl[i_max_phi, i]),
                                    (x_fl[i_max_phi, i], -y_fl[i_max_phi, i]),
                                    (x_fr[i_max_phi, i], -y_fr[i_max_phi, i]),
                                    (x_rr[i_max_phi, i], -y_rr[i_max_phi, i])]
            rect_biggest_phi_neg = shapely.geometry.Polygon(rect_biggest_phi_neg)

            # rectangle defining the vehicle its corner has the biggest x-value, x-value and y-value as in turning case
            rect_biggest_x = [(x_rr[i_max_x, i], y_rr[i_max_x, i]),
                              (x_rl[i_max_x, i], y_rl[i_max_x, i]),
                              (x_fl[i_max_x, i], y_fl[i_max_x, i]),
                              (x_fr[i_max_x, i], y_fr[i_max_x, i]),
                              (x_rr[i_max_x, i], y_rr[i_max_x, i])]
            rect_biggest_x = shapely.geometry.Polygon(rect_biggest_x)

            rect_biggest_x_neg = [(x_rr[i_max_x, i], -y_rr[i_max_x, i]),
                                  (x_rl[i_max_x, i], -y_rl[i_max_x, i]),
                                  (x_fl[i_max_x, i], -y_fl[i_max_x, i]),
                                  (x_fr[i_max_x, i], -y_fr[i_max_x, i]),
                                  (x_rr[i_max_x, i], -y_rr[i_max_x, i])]
            rect_biggest_x_neg = shapely.geometry.Polygon(rect_biggest_x_neg)

            # rectangle defining the vehicle breaking the most
            rect_biggest_break = [(x_rr[i_max_break, i], veh_width * 0.5),
                                  (x_rl[i_max_break, i], -veh_width * 0.5),
                                  (x_fl[i_max_break, i], -veh_width * 0.5),
                                  (x_fr[i_max_break, i], veh_width * 0.5),
                                  (x_rr[i_max_break, i], veh_width * 0.5)]
            rect_biggest_break = shapely.geometry.Polygon(rect_biggest_break)

            # rectangle defining the vehicle accelerating the most
            rect_biggest_acc = [(x_rr[i_max_acc, i], veh_width * 0.5),
                                (x_rl[i_max_acc, i], -veh_width * 0.5),
                                (x_fl[i_max_acc, i], -veh_width * 0.5),
                                (x_fr[i_max_acc, i], veh_width * 0.5),
                                (x_rr[i_max_acc, i], veh_width * 0.5)]
            rect_biggest_acc = shapely.geometry.Polygon(rect_biggest_acc)

            # -- COMPUTE INTERSECTION POINTS BETWEEN VEH-END-POSES -----------------------------------------------------
            intersection_coords = ([], [])
            try:
                intersection_1 = rect_biggest_phi.intersection(rect_biggest_phi_neg)
                intersection_2 = intersection_1.intersection(rect_biggest_x)
                intersection_3 = intersection_2.intersection(rect_biggest_x_neg)
                intersection_4 = intersection_3.intersection(rect_biggest_break)
                intersection_final = intersection_4.intersection(rect_biggest_acc)
                x, y = intersection_final.exterior.xy
                intersection_coords = (list(x), list(y))
            except AttributeError:
                # brake when no intersection was found
                if export_path is not None:
                    break

            # -- SAVE OCCUPATION AREA or STORE ADVANCED INFO FOR PLOT --------------------------------------------------
            if export_path is not None:
                export_dict[v0][t] = intersection_coords

            else:
                plot_dict[v0][t] = dict()

                x1, y1 = rect_biggest_phi.exterior.xy           # right turn - most extreme turn
                x2, y2 = rect_biggest_phi_neg.exterior.xy       # left turn - most extreme turn
                x3, y3 = rect_biggest_x.exterior.xy             # right turn - largest x
                x4, y4 = rect_biggest_x_neg.exterior.xy         # left turn - biggest x
                x5, y5 = rect_biggest_break.exterior.xy         # full deceleration
                x6, y6 = rect_biggest_acc.exterior.xy           # full acceleration

                x_extrema = (list(x1) + [None] + list(x2) + [None] + list(x3) + [None]
                             + list(x4) + [None] + list(x5) + [None] + list(x6))
                y_extrema = (list(y1) + [None] + list(y2) + [None] + list(y3) + [None]
                             + list(y4) + [None] + list(y5) + [None] + list(y6))

                plot_dict[v0][t]['extrema'] = [x_extrema, y_extrema]

                # store plot parameters for visualization of the reachable set
                x_reach_veh = []
                y_reach_veh = []
                for j in range(nmb_states * 2):
                    # vehicle for given state and right turn
                    x_reach_veh.extend([x_rr[j, i], x_rl[j, i], x_fl[j, i], x_fr[j, i], x_rr[j, i], None])
                    y_reach_veh.extend([y_rr[j, i], y_rl[j, i], y_fl[j, i], y_fr[j, i], y_rr[j, i], None])

                    # vehicle for given state and left turn
                    x_reach_veh.extend([x_rr[j, i], x_rl[j, i], x_fl[j, i], x_fr[j, i], x_rr[j, i], None])
                    y_reach_veh.extend([-y_rr[j, i], -y_rl[j, i], -y_fl[j, i], -y_fr[j, i], -y_rr[j, i], None])

                plot_dict[v0][t]['reach'] = [x_reach_veh, y_reach_veh]

                # store reduced intersection coordinates
                plot_dict[v0][t]['def_occ'] = intersection_coords

    # close progress-bar (even if aborted in last iteration)
    tph.progressbar.progressbar(1, 1, prefix="Simulating maneuvers ", suffix=" Done!")

    # ------------------------------------------------------------------------------------------------------------------
    # -- EXPORT or PLOT ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    if export_path is not None:
        with open(export_path, 'w') as fp:
            # add md5 key to dict
            export_dict['md5_key'] = md5_key

            # store as json formatted file
            json.dump(export_dict, fp)

    else:
        # setup plot
        fig, ax = plt.subplots()
        fig.canvas.set_window_title("Occupancy Analysis")
        plt.subplots_adjust(bottom=0.25)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        ax.autoscale(False)
        plt.tight_layout(rect=(0.05, 0.10, 1.0, 1.0))
        ax.axis('equal')
        ax.set_xlabel('x in m')
        ax.set_ylabel('y in m')

        ax_vel_sld = plt.axes([0.15, 0.08, 0.7, 0.03], facecolor=TUM_colors['TUM_blue_medium'])
        ax_t_sld = plt.axes([0.15, 0.02, 0.7, 0.03], facecolor=TUM_colors['TUM_blue_medium'])

        sld_vel = Slider(ax_vel_sld, 'v0 in m/s', 0.0, v_max, valinit=0.0, valstep=d_v)
        sld_t = Slider(ax_t_sld, 't in s', 0.0, t_max, valinit=t_max, valstep=d_t)

        sld_vel.on_changed(partial(change_plot, fig, ax, sld_vel, sld_t, plot_dict))
        sld_t.on_changed(partial(change_plot, fig, ax, sld_vel, sld_t, plot_dict))

        plt.show()

    return


def change_plot(fig, ax, vel_slider, t_slider, plot_dict, _):
    """
    Update the plot.

    """

    # clear content
    for artist in ax.lines + ax.collections + ax.texts:
        artist.remove()

    # get closest key to val
    vel_keys = list(plot_dict.keys())
    idx = min(range(len(vel_keys)), key=lambda i: abs(vel_keys[i] - vel_slider.val))
    vel_key = vel_keys[idx]

    # get closest key to val
    time_keys = list(plot_dict[vel_key].keys())
    idx = min(range(len(time_keys)), key=lambda i: abs(time_keys[i] - t_slider.val))
    t_key = time_keys[idx]

    # print reach for all time-steps
    for t_idx in time_keys:
        ax.plot(plot_dict[vel_key][t_idx]['reach'][1],
                plot_dict[vel_key][t_idx]['reach'][0], color=TUM_colors['TUM_blue_light'], zorder=1)

    # print cached information for selected v and t
    ax.plot(plot_dict[vel_key][t_key]['extrema'][1],
            plot_dict[vel_key][t_key]['extrema'][0], color=TUM_colors['TUM_green'])
    ax.plot(plot_dict[vel_key][t_key]['reach'][1],
            plot_dict[vel_key][t_key]['reach'][0], color=TUM_colors['TUM_blue_dark'])
    ax.plot(plot_dict[vel_key][t_key]['def_occ'][1],
            plot_dict[vel_key][t_key]['def_occ'][0], color=TUM_colors['TUM_orange'])

    # print vehicle at t = 0
    ax.plot(plot_dict[min(vel_keys)][min(time_keys)]['reach'][1],
            plot_dict[min(vel_keys)][min(time_keys)]['reach'][0], color=TUM_colors['TUM_green'])

    fig.canvas.draw_idle()


def reachable_set_creator(t_r: float,
                          t_step: int,
                          number_of_states: int,
                          v0: float,
                          turn_rad: float,
                          localgg: np.ndarray,
                          acc_max: np.ndarray) -> tuple:

    """
    Calculates a reachable set (with a time-horizon of '`t_r`') for the center of gravity (COG) by simulating multiple
    (edge-case) trajectories. The trajectories are sampled for each timestamp with a heading discretization fixed by the
    number of samples ('`number_of_states`').

    :param t_r:                 time range covered by the reachable-set in [s]
    :param t_step:              number of temporal steps t_r is divided in
    :param number_of_states:    number of states the reachable-set is computed for per quadrant
    :param v0:                  velocity the reachable set computation starts from in [m/s]
    :param turn_rad:            turning radius of vehicle [m]
    :param localgg:             maximum g-g-values at a certain position on the map (currently, only the max a used)
    :param acc_max:             velocity dependent maximum acceleration (motor limits)
    :returns:
        * **rule_zone** -       list in list returning the reachable set for every quadrant in x-position and y-position
        * **phi** -             list in list returning the total angle of turning in every time step

    :Authors:
        * Maximilian Bayerlein
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        20.01.2020

    """

    state_diff = 1 / (number_of_states - 1)

    # definition of a numpy.array holding time values from zero to t_r divided in
    t = np.linspace(0.0, t_r, num=t_step)

    # definition of the length of every t_step in [s]
    step_width = t[-1] / (t_step - 1)

    # creation of empty lists for the reachable set: rule_zone
    reach_set = np.zeros([number_of_states, t_step, 2, 2])

    # creation of empty lists for the phi list: phi
    phi = np.zeros([number_of_states, t_step, 2])

    # ------------------------------------------------------------------------------------------------------------------
    # ---- Calculation of the path the COG takes for a given ratio of a_y and a_x with changing ratio from 0 to inf ----
    # ------------------------------------------------------------------------------------------------------------------

    # calculation of the acceleration side for all states of ratio
    for i in range(number_of_states):
        xx, xy, phi[i, :, 0] = trajectory_def(proportion=state_diff * i,
                                              v0=v0,
                                              step=t_step,
                                              step_width=step_width,
                                              turn_rad=turn_rad,
                                              localgg=localgg,
                                              acc_max=acc_max)
        reach_set[i, :, 0, 0] = xx
        reach_set[i, :, 0, 1] = xy

    # calculation of the breaking side for all states of ratio
    for i in range(number_of_states):
        xx, xy, phi[-1 - i, :, 1] = trajectory_def(proportion=-state_diff * i,
                                                   v0=v0,
                                                   step=t_step,
                                                   step_width=step_width,
                                                   turn_rad=turn_rad,
                                                   localgg=localgg,
                                                   acc_max=acc_max)
        reach_set[-1 - i, :, 1, 0] = xx
        reach_set[-1 - i, :, 1, 1] = xy

    return reach_set, -phi


def trajectory_def(proportion: float,
                   v0: float,
                   step: int,
                   step_width: float,
                   turn_rad: float,
                   localgg: np.ndarray,
                   acc_max: np.ndarray) -> tuple:

    """
    This function simulates one single edge-case trajectory for the given velocity and share / proportion of lat. / lon.
    acceleration.

    :param proportion:          ratio between a_x and a_y; number in range [-1, 1]; negative numbers for braking
    :param v0:                  velocity at start of maneuver in [m/s]
    :param step:                number of steps the iteration is made on
    :param step_width:          length of every step in [s]
    :param turn_rad:            turning radius of vehicle [m]
    :param localgg:             maximum g-g-values at a certain position on the map (currently, only the max a used)
    :param acc_max:             velocity dependent maximum acceleration (motor limits)
    :returns:
        * **xx** -              x-value of turning vehicle in [m]
        * **xy** -              y-value of turning vehicle in [m]
        * **phi_c** -           turning angle of the vehicle in [rad]

    :Authors:
        * Maximilian Bayerlein
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        20.01.2020

    """

    acc_c = np.zeros(step)          # array defining the acceleration in every time step
    vel_c = np.zeros(step + 1)      # array defining the velocity in every time step
    vel_c = vel_c + v0
    x_c = np.zeros((step + 1))      # array defining the path in x-direction in every time step
    kappa_c = np.zeros(step)        # array defining the recipocal of radius of the trajectory in every time step
    phi_c = np.zeros(step)          # array defining the absolute angle of turing in every time step
    d_phi_c = np.zeros(step)        # array defining the relative angle of turing in every time step

    for i in range(step):
        # acceleration depending on a_y / a_x ratio
        acc_c[i], a_y = proportion_longitudinal_acc(proportion=proportion,
                                                    velocity=vel_c[i],
                                                    localgg=localgg,
                                                    acc_mach_max=acc_max)
        # actual velocity following Kamm's circle of forces in the actual time step
        # velocity in the next time step
        vel_c[i + 1] = step_width * acc_c[i] + vel_c[i]

        # if velocity is smaller zero -> set actual acc and velocity zero as well as in the next step
        if vel_c[i] < 0:
            vel_c[i] = 0
            acc_c[i] = 0
            vel_c[i + 1] = 0

        # x-position in next step
        x_c[i + 1] = 0.5 * step_width ** 2 * acc_c[i] + vel_c[i] * step_width + x_c[i]

        # maximum possible radius of the actual trajectory element (expressed via kappa)
        kappa_c[i] = -a_y / (np.where(abs(vel_c[i]) < 0.001, 0.001, vel_c[i]) ** 2)

        # limit kappa by respecting turn radius
        if kappa_c[i] > 1 / turn_rad:
            kappa_c[i] = 1 / turn_rad
        elif kappa_c[i] < -1 / turn_rad:
            kappa_c[i] = -1 / turn_rad

    # remove "0.0" values in kappa array, in order to avoid division by zero
    kappa_c = np.where(abs(kappa_c) < 0.001, 0.001, kappa_c)

    # reduce x-position array to the number of steps
    x_c = x_c[:step]

    # define starting angles as zero
    phi_c[0] = 0
    d_phi_c[0] = phi_c[0]

    # compute angles executed in every time step, adding the to the absolute angle
    for i in range(1, step):
        d_phi_c[i] = (x_c[i] - x_c[i - 1]) * kappa_c[i]
        phi_c[i] = phi_c[i - 1] + d_phi_c[i - 1]

    # progress in x/y-direction respecting driven radius (kappa)
    x_circle = np.sin(d_phi_c) / kappa_c
    y_circle = (1 - np.cos(d_phi_c)) / kappa_c

    rot_mat = np.zeros([2, 2, step])
    xy = np.zeros(step)
    xx = np.zeros(step)

    for i in range(1, step):
        rot_mat[:, :, i] = [[np.cos(phi_c[i]), np.sin(phi_c[i])], [-np.sin(phi_c[i]), np.cos(phi_c[i])]]  # RotationalM
        xx[i] = x_circle[i] * rot_mat[0, 0, i] - y_circle[i] * rot_mat[0, 1, i] + xx[i - 1]  # Rotating circle elements
        xy[i] = x_circle[i] * rot_mat[1, 0, i] - y_circle[i] * rot_mat[1, 1, i] + xy[i - 1]  # Rotating circle elements

    return xx, xy, phi_c


def proportion_longitudinal_acc(proportion: float,
                                velocity: float,
                                localgg: np.ndarray,
                                acc_mach_max: np.ndarray):

    """
    This function retriefs the  available acceleration in x- and y-direction based on friction circle ('`localgg`'),
    motor limits ('`acc_mach_max`') for current velocity and proportion
    (relation between long / lat --> interpreted as angle in friction circle).

    :param proportion:          ratio of a_y / a_x
    :param velocity:            actual velocity in [m/s]
    :param localgg:             maximum g-g-values at a certain position on the map (currently, only the max a used)
    :param acc_mach_max:        velocity dependent maximum acceleration (motor limits)
    :returns:
        * **acc_x** -           possible acceleration in x-direction in [m/s**2]
        * **acc_y** -           possible acceleration in y-direction in [m/s**2]

    :Authors:
        * Maximilian Bayerlein
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        20.01.2020

    """

    phi = proportion * np.pi * 0.5                              # angle in gg-diagram
    acc_x_max = max(localgg[:, 3])                              # max. acceleration (tires) in x-direction
    acc_y_max = max(localgg[:, 4])                              # max. acceleration (tires) in y-directions

    # calculate possible acceleration in x- and y-direction
    acc_x = np.sqrt((acc_y_max * np.sin(phi)) ** 2 + (acc_x_max * np.cos(phi)) ** 2) * np.sin(phi)
    acc_y = np.sqrt((acc_y_max * np.sin(phi)) ** 2 + (acc_x_max * np.cos(phi)) ** 2) * np.cos(phi)

    # if positive (>0) acceleration in x-direction, check if exceeded physical motor limitations
    if proportion > 0:
        possible_max_acc = np.interp(velocity, acc_mach_max[:, 0], acc_mach_max[:, 1])
        if acc_x > possible_max_acc:
            acc_x = possible_max_acc

    return acc_x, acc_y


if __name__ == "__main__":
    toppath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

    localgg_path = "/params/veh_dyn_info/localgg.csv"
    ax_max_path = "/params/veh_dyn_info/ax_max_machines.csv"

    # retrieve localgg and ax_max_machines
    ax_max_machines_f = tph.import_veh_dyn_info.import_veh_dyn_info(ax_max_machines_import_path=(toppath
                                                                                                 + ax_max_path))[1]
    localgg = tph.import_veh_dyn_info_2.import_veh_dyn_info_2(filepath2localgg=(toppath + localgg_path))

    # calculate guaranteed occupation maps in visual mode
    guar_occ_calc(t_max=2.0,
                  d_t=0.1,
                  v_max=50.0,
                  d_v=0.5,
                  localgg=localgg,
                  ax_max_mach=ax_max_machines_f,
                  export_path=None)
