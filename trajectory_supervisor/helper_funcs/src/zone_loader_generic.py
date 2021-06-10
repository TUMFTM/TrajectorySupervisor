import json
import numpy as np


def zone_loader_generic(path_map: str,
                        raceline: np.ndarray) -> tuple:
    """
    This script reads in the roborace json map including the zone definitions. It outputs the zones with their
    respective s coordinates for start and end points, the pit-lane and the map reference point.

    :param path_map:            path pointing to the map file to be loaded
    :param raceline:            raceline with (at least) following first three columns: s, x, y
    :returns:
        * **zones** -           list of dicts, where each dict describes a zone with the following keys:

                                * 'type':       integer describing zone type:

                                                * 0: start/finish
                                                * 1: trigger
                                                * 2: overtake left
                                                * 3: overtake right
                                                * 4: stop zone
                                                * 5: pit

                                * 'id':         unique id for each zone
                                * 's_start':    s-coordinate along the race-line, where zone starts
                                * 's_end':      s-coordinate along the race-line, where zone ends
                                * 'pos_start':  position where zone starts
                                * 'pos_end':    position where zone ends

        * **pitline** -         pitline s-coord and coords (columns: s, x, y)
        * **map_ref_point** -   reference point for the map

    :Authors:
        * Alexander Heilmeier <alexander.heilmeier@tum.de>
        * Tim Stahl <tim.stahl@tum.de>
        * Yves Huberty

    :Created on:
        02.07.2020

    """

    # ------------------------------------------------------------------------------------------------------------------
    # LOAD INPUT MAP ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # load map data
    with open(path_map, 'r') as fh:
        data_map = json.load(fh)

    # ------------------------------------------------------------------------------------------------------------------
    # PROCESS MAP DATA -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # try to get pitline (only in v2 regulations)
    if 'CentrePitlane' in data_map:
        # calculate distances for pitline to get array as [s, x, y]
        pitline = np.array(data_map['CentrePitlane'])
        el_lenghts = np.sqrt(np.sum(np.power(np.diff(pitline, axis=0), 2), axis=1))
        dists_pitline = np.cumsum(el_lenghts)
        dists_pitline = np.insert(dists_pitline, 0, 0.0)
        pitline = np.column_stack((dists_pitline, pitline))
    else:
        pitline = None

    zones = []

    # process map data
    for zone in data_map["Zone"]:
        # 0, 4: start finish line, stop zone
        if zone["Type"] in [0, 4]:
            zone_xy = {'type': zone['Type'],
                       'id': None,
                       'pos_start': data_map["Centre"][zone["Start"]],  # [x, y]
                       'pos_end': data_map["Centre"][zone["End"]]}      # [x, y]

            zones.append(zone_xy)

        # 1: trigger zones
        elif zone["Type"] == 1:
            zone_xy = {'type': zone['Type'],
                       'id': -zone["ID"],  # negative IDs because objects have positive IDs
                       'pos_start': data_map["Centre"][zone["Start"]],  # [x, y]
                       'pos_end': data_map["Centre"][zone["End"]]}  # [x, y]

            zones.append(zone_xy)

        # 2, 3: overtaking zones (overtake on left, overtake on right)
        elif zone["Type"] in [2, 3]:
            zone_xy = {'type': zone['Type'],
                       'id': -zone["ID"],                               # negative IDs because objects have positive IDs
                       'pos_start': data_map["Centre"][zone["Start"]],  # [x, y]
                       'pos_end': data_map["Centre"][zone["End"]]}      # [x, y]

            zones.append(zone_xy)

        # 5, 6: pit lane zone and pit stop zone (related to pitline)
        elif zone["Type"] in [5, 6]:
            zone_xy = {'type': zone['Type'],
                       'id': None,
                       'pos_start': data_map["CentrePitlane"][zone["Start"]],  # [x, y]
                       'pos_end': data_map["CentrePitlane"][zone["End"]]}  # [x, y]

            zones.append(zone_xy)

    # extract map reference point for VDC
    if "ReferencePoint" in data_map:
        map_ref_point = data_map["ReferencePoint"]
    else:
        map_ref_point = None

    # ------------------------------------------------------------------------------------------------------------------
    # PROCESS ZONES ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate s_start and s_end for every zone

    for zone in zones:
        # RELATED TO RACELINE
        # start finish line, trigger zone and overtaking zones
        if zone['type'] in [0, 1, 2, 3, 4]:
            dists_start = np.sqrt(np.power(raceline[:, 1] - zone['pos_start'][0], 2)
                                  + np.power(raceline[:, 2] - zone['pos_start'][1], 2))
            dists_end = np.sqrt(np.power(raceline[:, 1] - zone['pos_end'][0], 2)
                                + np.power(raceline[:, 2] - zone['pos_end'][1], 2))

            zone["s_start"] = raceline[np.argmin(dists_start), 0]
            zone["s_end"] = raceline[np.argmin(dists_end), 0]

            if zone['type'] == 0 and not (zone['s_start'] <= 5.0 or zone['s_start'] >= raceline[-1, 0] - 5.0):
                print('WARNING: Start finish line is not close the start of the race line!')

        # RELATED TO PITLINE
        # pit lane zone and pit stop zone
        elif zone['type'] in [5, 6]:
            dists_start = np.sqrt(np.power(pitline[:, 1] - zone['pos_start'][0], 2)
                                  + np.power(pitline[:, 2] - zone['pos_start'][1], 2))
            dists_end = np.sqrt(np.power(pitline[:, 1] - zone['pos_end'][0], 2)
                                + np.power(pitline[:, 2] - zone['pos_end'][1], 2))

            zone["s_start"] = pitline[np.argmin(dists_start), 0]
            zone["s_end"] = pitline[np.argmin(dists_end), 0]

    return zones, pitline, map_ref_point


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
