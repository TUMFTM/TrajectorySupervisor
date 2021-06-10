import numpy as np
import scenario_testing_tools as stt


def export_traj(file_path: str,
                bound_l: np.ndarray,
                bound_r: np.ndarray,
                ego_traj_xy: np.ndarray,
                ego_psi: np.ndarray,
                ego_kappa: np.ndarray,
                ego_vx: np.ndarray,
                ego_ax: np.ndarray,
                sp_res: float = 2.0
                ) -> None:

    """
    This function is used to export a trajectory to a file for further usage in the graph-based local trajectory
    planner. Therefore, a reference-line and matching normal-vectors (required by the planner) are calculated and
    exported.


    :param file_path:       path (including file name) the generated trajectory file should be stored to
    :param bound_l:         coordinates of the left bound (numpy array with columns x and y)
    :param bound_r:         coordinates of the right bound (numpy array with columns x and y)
    :param ego_traj_xy:     coordinates of the ego trajectory (numpy array with columns x and y)
    :param ego_psi:         heading of the ego-trajectory at given coordinates
    :param ego_kappa:       curvature of the ego-trajectory at given coordinates
    :param ego_vx:          velocity of the ego-trajectory at given coordinates
    :param ego_ax:          acceleration of the ego-trajectory at given coordinates
    :param sp_res:          (optional) support point resolution, defines the step-width of the reference line

                            * a small step-width might result in curvature spikes, since the splines have less freedom
                            * a large step-width might result in hitting the bounds due to less accurate tracking

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        12.05.2020

    """

    # -- fit a reference line for given bounds -------------------------------------------------------------------------
    output_rl, v_normal, tw_left, tw_right = stt.generate_refline.generate_refline(bound_l=bound_l,
                                                                                   bound_r=bound_r,
                                                                                   resolution=sp_res)

    # update bounds with new resolution
    bound_l = output_rl - v_normal * tw_left
    bound_r = output_rl + v_normal * tw_right

    # -- match ego-trajectory to reference-line and normal-vectors -----------------------------------------------------
    n_el = v_normal.shape[0]
    alphas = np.zeros(n_el)
    psi = np.zeros(n_el)
    kappa = np.zeros(n_el)
    vx = np.zeros(n_el)
    ax = np.zeros(n_el)

    # for every normal vector (find intersection point)
    for i in range(n_el):
        # scan through all trajectory segments until intersection is found
        for j in range(ego_traj_xy.shape[0] - 1):
            # check if intersection in segment
            if intersect(pnt_a=bound_l[i, :],
                         pnt_b=bound_r[i, :],
                         pnt_c=ego_traj_xy[j, :],
                         pnt_d=ego_traj_xy[j + 1, :]):
                # if intersection found, calculate intersection point and exit inner loop
                pnt_temp = intersect_point(pnt_a=bound_l[i, :],
                                           pnt_b=bound_r[i, :],
                                           pnt_c=ego_traj_xy[j, :],
                                           pnt_d=ego_traj_xy[j + 1, :])

                # calculate offset to reference line and store in alpha array
                alphas[i] = -np.hypot(output_rl[i, 0] - pnt_temp[0], output_rl[i, 1] - pnt_temp[1])

                # get sign of alpha (left / right of reference-line) - calculated via sign of determinant
                alphas[i] *= np.sign((output_rl[i + 1, 0] - output_rl[i, 0]) * (pnt_temp[1] - output_rl[i, 1])
                                     - (output_rl[i + 1, 1] - output_rl[i, 1]) * (pnt_temp[0] - output_rl[i, 0]))

                # interpolate corresponding heading, curvature, velocity and acceleration
                psi[i] = stt.interp_heading.interp_heading(heading=ego_psi[j:j + 2],
                                                           t_series=ego_traj_xy[j:j + 2, 0],
                                                           t_in=pnt_temp[0])
                kappa[i] = np.interp(pnt_temp[0], ego_traj_xy[j:j + 2, 0], ego_kappa[j:j + 2])
                vx[i] = np.interp(pnt_temp[0], ego_traj_xy[j:j + 2, 0], ego_vx[j:j + 2])
                ax[i] = np.interp(pnt_temp[0], ego_traj_xy[j:j + 2, 0], ego_ax[j:j + 2])

                break

    # calculate new ego_trajectory with new step-width
    ego_traj_xy = output_rl + v_normal * np.expand_dims(alphas, 1)

    # calculate s-coordinate
    el_length = np.sqrt(np.sum(np.diff(ego_traj_xy, axis=0) ** 2, axis=1))
    s = np.insert(np.cumsum(el_length), 0, 0.0)

    # -- create data structure for local trajectory planner ------------------------------------------------------------
    traj_ltpl = np.column_stack((output_rl,
                                 tw_right,
                                 tw_left,
                                 v_normal,
                                 alphas,
                                 s,
                                 psi,
                                 kappa,
                                 vx,
                                 ax))

    # -- export trajectory data for local planner ----------------------------------------------------------------------
    header = "x_ref_m; y_ref_m; width_right_m; width_left_m; x_normvec_m; y_normvec_m; " \
             "alpha_m; s_racetraj_m; psi_racetraj_rad; kappa_racetraj_radpm; vx_racetraj_mps; ax_racetraj_mps2"
    fmt = "%.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f"
    with open(file_path, 'w') as fh:
        np.savetxt(fh, traj_ltpl, fmt=fmt, header=header)


def intersect(pnt_a: tuple,
              pnt_b: tuple,
              pnt_c: tuple,
              pnt_d: tuple) -> bool:
    """
    Determines if two line segments [pnt_a pnt_b] and [pnt_c pnt_d] intersect.

    :param pnt_a:           point defined by a tuple of x and y coordinate
    :param pnt_b:           point defined by a tuple of x and y coordinate
    :param pnt_c:           point defined by a tuple of x and y coordinate
    :param pnt_d:           point defined by a tuple of x and y coordinate
    :returns:
        * **intersect** -   'true' if the line-segments intersect, 'false' otherwise

    """

    def ccw_order(pnt_1: tuple,
                  pnt_2: tuple,
                  pnt_3: tuple):
        """
        Determines if three points pnt_1, pnt_2, pnt_3 are listed in a counterclockwise order.

        :param pnt_1:       point defined by a tuple of x and y coordinate
        :param pnt_2:       point defined by a tuple of x and y coordinate
        :param pnt_3:       point defined by a tuple of x and y coordinate
        :returns:
            * **ccw** -     true' if arranged in a counterclockwise order, 'false' otherwise
        """

        return (pnt_3[1] - pnt_1[1]) * (pnt_2[0] - pnt_1[0]) > (pnt_2[1] - pnt_1[1]) * (pnt_3[0] - pnt_1[0])

    return (ccw_order(pnt_a, pnt_c, pnt_d) != ccw_order(pnt_b, pnt_c, pnt_d)
            and ccw_order(pnt_a, pnt_b, pnt_c) != ccw_order(pnt_a, pnt_b, pnt_d))


def intersect_point(pnt_a: tuple,
                    pnt_b: tuple,
                    pnt_c: tuple,
                    pnt_d: tuple) -> tuple or False:
    """
    Determines the intersection point (x, y) of two lines, each going through the point pairs [pnt_a pnt_b] and
    [pnt_c pnt_d].

    :param pnt_a:           point defined by a tuple of x and y coordinate
    :param pnt_b:           point defined by a tuple of x and y coordinate
    :param pnt_c:           point defined by a tuple of x and y coordinate
    :param pnt_d:           point defined by a tuple of x and y coordinate
    :returns:
        * **intersect** -   intersection point defined by a tuple of x and y coordinate; 'false' if no intersection

    """

    # produces coeffs a, b, c of line equation by two points provided
    def line(pnt_1: tuple,
             pnt_2: tuple):
        a = (pnt_1[1] - pnt_2[1])
        b = (pnt_2[0] - pnt_1[0])
        c = (pnt_1[0] * pnt_2[1] - pnt_2[0] * pnt_1[1])
        return a, b, -c

    # get line coeffs for given points
    line1 = line(pnt_a, pnt_b)
    line2 = line(pnt_c, pnt_d)

    d = line1[0] * line2[1] - line1[1] * line2[0]
    dx = line1[2] * line2[1] - line1[1] * line2[2]
    dy = line1[0] * line2[2] - line1[2] * line2[0]
    if d != 0:
        x = dx / d
        y = dy / d
        return x, y
    else:
        return False


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
