import time
import numpy as np
import platform
import multiprocessing
import warnings
import scenario_testing_tools as stt


def sync_data(traj_perf: dict,
              traj_em: dict,
              objects: dict,
              allowed_t_offset: float = 0.5,
              use_mp: bool = False) -> tuple:
    """
    This function syncs a given pair of trajectories with the object-list. Therefore, it extrapolates the
    trajectory or objects to the newest time-stamp available (among them) in order to be able to set them in
    relation to each other. If the trajectory has a more recent timestamp, the (older) objects are predicted forward
    accordingly with a constant velocity model. If the objects have a more recent time stamp, the (older) trajectories
    are integrated forward until the time difference is equalized.

    In general, this function does not have to be used if it can be ensured that the objects used to plan the
    trajectories can be transmitted synchronously with each other.  If this is not the case and/or one of the two
    data is significantly more frequent and a comparison is desirable, this function can be used.

    Note: for the synchronization to work, a time stamp in seconds must be stored in the dict of the trajectories as
    well as in the dict of the objects. It is essential that both use the same time source. In addition, the time stamp
    of the trajectory must be the same as the time stamp of the objects that serve as the basis for planning.

    :param traj_perf:           performance trajectory in form of a dict with the following entries:
                                   - 'traj':  trajectory data as numpy array with columns: s, x, y, head, curv, vel, acc
                                   - 'id':    unique id of the trajectory (int)
                                   - 'time':  time stamp of the trajectory in seconds
    :param traj_em:             emergency trajectory in form of a dict (same entries / format as traj_perf)
    :param objects:             object-list dict with each key being a dedicated object id and each object hosting a
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
    :param allowed_t_offset:    (optional) maximum allowed temporal offset between trajectories and object-list [in s]
                                -> if exceeded, a warning is raised and the sync is not performed
    :param use_mp:              (optional) if set to true and executed on a Linux machine, multiprocessing is used

    :returns:
        * **traj_perf_sync** -  synced performance trajectory dict
        * **traj_em_sync** -    synced emergency trajectory dict
        * **obj_sync** -        synced object list

    """

    # init variables
    traj_sync = {'perf': None,
                 'emerg': None}

    # if no objects in object-list (nothing to sync --> return plain trajectory)
    if not objects:
        return traj_perf, traj_em, objects

    # retrieve time-stamps (assume all trajectories and objects hold same time-stamps in their respective sets)
    traj_stamp = traj_perf['time']
    obj_stamp = list(objects.values())[0]['time']

    temp_offset = abs(traj_stamp - obj_stamp)

    if temp_offset > allowed_t_offset:
        # if time-stamp of object-list and trajectory diverges to much
        warnings.warn("Skipped sync (rating of trajectory only)! Time-stamp of last trajectory and object list"
                      " diverged more than %.2fs (actual offset: %.2fs)!" % (allowed_t_offset, temp_offset))

        return traj_perf, traj_em, objects

    elif temp_offset < 0.01:
        # do not adjust, when almost same time-stamp

        return traj_perf, traj_em, objects

    else:
        if traj_stamp > obj_stamp:
            # if trajectory is more recent than objects -> predict obj into future (for period of temp_offset)
            # NOTE: Everything is assumed to be constant, except position is translated according to velocity

            # copy current entries
            traj_sync['perf'] = traj_perf['traj']
            traj_sync['emerg'] = traj_em['traj']
            obj_sync = objects

            # for every object
            for obj_id in obj_sync.keys():
                # Translate objects based on velocity and temporal offset
                obj_sync[obj_id]['X'] += (obj_sync[obj_id]['v_x'] * np.sin(-obj_sync[obj_id]['theta'])
                                          * temp_offset)
                obj_sync[obj_id]['Y'] += (obj_sync[obj_id]['v_x'] * np.cos(obj_sync[obj_id]['theta'])
                                          * temp_offset)

        else:
            # if objects are more recent than trajectory --> travel along trajectory for given temp_offset

            # copy current entries
            obj_sync = objects

            # clip performance and emergency trajectory (parallel processing if enabled)
            mp_queue = multiprocessing.Queue()

            jobs = []
            pending_results = 0
            for traj_type, traj_orig in zip(['perf', 'emerg'], [traj_perf, traj_em]):

                if platform.system() == 'Linux' and use_mp:
                    # call trajectory clipping in parallel manner
                    p = multiprocessing.Process(target=clip_trajectory,
                                                args=(traj_orig['traj'], temp_offset, traj_type, mp_queue))
                    jobs.append(p)
                    p.start()
                else:
                    # call trajectory clipping
                    clip_trajectory(traj_in=traj_orig['traj'],
                                    temp_offset=temp_offset,
                                    traj_type=traj_type,
                                    mp_queue=mp_queue)

                pending_results += 1

            # sync jobs
            for p in jobs:
                p.join()

            # extract results from queue
            while pending_results != 0 or not mp_queue.empty():
                msg = mp_queue.get(timeout=0.1)

                for traj_type in msg.keys():
                    traj_sync[traj_type] = msg[traj_type]
                    pending_results -= 1

            mp_queue.close()

    # write sync entries to trajectories
    traj_perf['traj'] = traj_sync['perf']
    traj_em['traj'] = traj_sync['emerg']

    return traj_perf, traj_em, obj_sync


def clip_trajectory(traj_in: np.ndarray,
                    temp_offset: float,
                    traj_type: str,
                    mp_queue: multiprocessing.Queue) -> None:
    """
    Clips the start of a given trajectory based on the temporal offset ('`temp_offset`') provided.

    :param traj_in:         trajectory to be handled with following columns [s, x, y, heading, curv, vel, acc]
    :param temp_offset:     temporal offset to be clipped at beginning of trajectory in seconds
    :param traj_type:       string describing the trajectory type ('perf' or 'emerg')
    :param mp_queue:        queue that will receive all results

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        30.03.2020

    """

    # iterate in trajectory until temporal offset is exceeded
    t = 0.0
    dt = 0.0
    i = 0

    while (t < temp_offset) and (i < (traj_in.shape[0] - 2)):
        # calculate passed time based on velocity and travelled distance in trajectory
        dt = (((((traj_in[i, 1] - traj_in[i + 1, 1]) ** 2)
                + ((traj_in[i, 2] - traj_in[i + 1, 2]) ** 2)) ** 0.5)
              / max(0.5 * (traj_in[i, 5] + traj_in[i + 1, 5]), 0.0001))

        t += dt
        i += 1

    # interpolate new start point between relevant points (caution: heading jump between -pi and pi)
    # Spacing: t - dt (point before current position) | temp_offset (current position) | t (point after)
    s = np.interp(temp_offset, [t - dt, t], [traj_in[max(i - 1, 0), 0], traj_in[i, 0]])
    pos_x = np.interp(temp_offset, [t - dt, t], [traj_in[max(i - 1, 0), 1], traj_in[i, 1]])
    pos_y = np.interp(temp_offset, [t - dt, t], [traj_in[max(i - 1, 0), 2], traj_in[i, 2]])

    # convert to positive values and back in order to avoid linear interpolation issues
    heading = stt.interp_heading.\
        interp_heading(heading=np.array([traj_in[max(i - 1, 0), 3], traj_in[i, 3]]),
                       t_series=np.array([t - dt, t]),
                       t_in=temp_offset)
    # print([heading, np.array([traj_orig[max(i - 1, 0), 2], traj_orig[i, 2]])])

    curv = np.interp(temp_offset, [t - dt, t], [traj_in[max(i - 1, 0), 4], traj_in[i, 4]])
    vel = np.interp(temp_offset, [t - dt, t], [traj_in[max(i - 1, 0), 5], traj_in[i, 5]])
    acc = np.interp(temp_offset, [t - dt, t], [traj_in[max(i - 1, 0), 6], traj_in[i, 6]])

    # insert new point at beginning and join with trajectory following that point
    traj_candidate = np.vstack((np.array((s, pos_x, pos_y, heading, curv, vel, acc)), traj_in[i:, :]))

    # realign new s-coorinate (start at 0.0 again
    traj_candidate[:, 0] -= traj_candidate[0, 0]

    # push result to queue
    mp_queue.put({traj_type: traj_candidate})

    # wait to ensure proper handling of queue
    time.sleep(0.00001)
