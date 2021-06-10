import numpy as np
import configparser
import trajectory_supervisor


class SupModStaticCollision(object):
    """
    Class handling safety checks regarding collision with map-defined boundaries / static objects.

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
                 veh_params: dict):
        """
        Init the SupMod.

        :param supmod_config_path:  path to Supervisor config file
        :param veh_params:          dict of vehicle parameters; must hold the following keys:
                                      veh_width -     width of the ego-vehicle [in m]
                                      veh_length -    length of the ego-vehicle [in m]

        """

        # read configuration file
        safety_param = configparser.ConfigParser()
        if not safety_param.read(supmod_config_path):
            raise ValueError('Specified config file does not exist or is empty!')

        # -- set internal variables ------------------------------------------------------------------------------------
        self.__occ_map = None
        self.__bound_l = None
        self.__bound_r = None

        # -- get parameters for the static safety assessment from configuration file -----------------------------------
        self.__plot_occ = safety_param.getboolean('STATIC_COLLISION', 'plot_occupancy')
        safety_factor = safety_param.getfloat('STATIC_COLLISION', 'safety_factor')

        self.__veh_width = veh_params['veh_width']
        self.__veh_length = veh_params['veh_length']

        # calculate collision check width based on safety factor
        if 0.0 <= safety_factor <= 1.0:
            veh_diagonal = np.hypot(self.__veh_width, self.__veh_length)

            self.__col_width = np.interp(safety_factor, [0.0, 1.0], [self.__veh_width, veh_diagonal])
        else:
            raise ValueError("'safety_factor' should be in range [0, 1] and is '" + str(safety_factor) + "'!")

    # ------------------------------------------------------------------------------------------------------------------
    # DESTRUCTOR -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __del__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # UPDATE MAP -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def update_map(self,
                   occ_map: dict = None,
                   bound_l: np.ndarray = None,
                   bound_r: np.ndarray = None) -> None:
        """
        Update the internal map representation (either occupancy map OR left and right bound).

        :param occ_map:     dict with the following keys
                                * grid:           occupancy grid (np-array) holding 1 for occ. and 0 for unocc. cells
                                * origin:         x, y coordinates of the origin (0, 0) in the occupancy grid
                                * resolution:     grid resolution in meters
        :param bound_l:     coordinates of the left bound (numpy array with columns x, y)
        :param bound_r:     coordinates of the right bound (numpy array with columns x, y)
        """

        # check if valid set of parameters was provided, else raise error
        if occ_map is None and (bound_l is None or bound_r is None):
            raise ValueError("Map-udpate error! At least an occupancy map OR both bounds must be provided!")

        if occ_map is not None and (bound_l is not None or bound_r is not None):
            raise ValueError("Map-udpate error! Please provide only one map format, either occupancy map OR left and "
                             "right bound coordinates!")

        # store values in class variables
        self.__occ_map = occ_map
        self.__bound_l = bound_l
        self.__bound_r = bound_r

        return

    # ------------------------------------------------------------------------------------------------------------------
    # CALC SCORE -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def calc_score(self,
                   ego_data: np.ndarray) -> tuple:
        """
        Asses the trajectory safety for a given time instance regarding collision with the wall or any static obstacle
        on the race track

        :param ego_data:        data of the ego vehicle(s, pos_x, pos_y, heading, curvature, velocity, acceleration)
                                for a given time instance
        :returns:
            * **safety** -            binary value indicating the safety state. 'False' = unsafe and 'True' = safe
            * **safety_parameters** - parameter dict

        """

        # -- check safety regarding obstacle collision with wall bounds or static obstacles ----------------------------
        safety_parameters_bound_intersect = dict()

        if self.__occ_map is None and (self.__bound_l is None or self.__bound_r is None):
            raise ValueError("Could not calculate static safety score since no map information was provided!")

        if self.__occ_map is not None:
            safety = trajectory_supervisor.supervisor_modules.supmod_static_collision.src.check_collision_occ_map.\
                check_collision_occ_map(ego_path=ego_data[:, 1:4],
                                        occ_map=self.__occ_map,
                                        col_width=self.__col_width,
                                        plot_occ=self.__plot_occ)
        else:
            safety, safety_parameters_bound_intersect = trajectory_supervisor.supervisor_modules.\
                supmod_static_collision.src.check_collision_lanelet.check_collision_lanelet(
                    ego_path=ego_data[:, 1:4],
                    bound_l=self.__bound_l,
                    bound_r=self.__bound_r,
                    col_width=self.__col_width,
                    veh_length=self.__veh_length)
        safety_parameters = {"stat_col_cur": safety,
                             **safety_parameters_bound_intersect}

        return safety, safety_parameters
