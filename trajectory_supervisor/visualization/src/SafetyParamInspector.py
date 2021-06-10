import numpy as np
import time

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Show plot in serif font (e.g. for publications)
# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

# ----------------------------------------------------------------------------------------------------------------------
# - USER SPECS ---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Keys containing one of these string-snippets will be ignored
# '*_bound':    assumes keys with this optional entry, specify the limit for the corresponding value (checked sep.)
# 'gocc_*' :    2D polygon of guaranteed occupation area - handled separately
IGNORE_LIST = ['_bound', 'gocc_', 'gocccol', 'rrese_', 'rresecol', 'stat_intersect', 'stat_idx',
               'bound_reach_set_outline', 'a_lon_used', 'a_lat_used', 'a_comb_used', 'calctime']

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

# Configure visual experience
BTN_COLOR = 'lightgrey'
VEH_COLORS = [TUM_colors['TUM_orange'], TUM_colors['TUM_blue'], TUM_colors['TUM_green']]


class SafetyParamInspector(object):
    """
    This class handles the visualization of the course of safety scores and relevant parameters.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        17.04.2019
    """

    def __init__(self,
                 time_stamps: list,
                 safety_combined: list,
                 safety_base: dict) -> None:
        # --------------------------------------------------------------------------------------------------------------
        # - INIT VARIABLES ---------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        self.__time = time_stamps
        self.__safety_combined = safety_combined
        self.__safety_base = safety_base

        # --------------------------------------------------------------------------------------------------------------
        # - INIT PLOTS -------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # determine number of fields plotted
        field_cnt = 0
        field_names = []
        for key in sorted(self.__safety_base.keys()):

            # if key is not on ignore list
            if not any(x in key for x in IGNORE_LIST):
                field_names.append(key)
                field_cnt += 1

        # -- configure plot --
        self.__fig, axes = plt.subplots(field_cnt + 1, 1, sharex=True)
        self.__fig.canvas.set_window_title("Safety Parameter Inspector")

        # axis handle for all plots
        self.__ax_handle = dict()

        # data handles
        self.__safety_score_handle = None
        self.__safety_base_handle = dict()
        self.__cursor_handles = dict()

        # main axis (safety evaluation)
        if field_cnt > 0:
            self.__ax_handle[0] = axes[0]
        else:
            self.__ax_handle[0] = axes
        # self.__ax_handle[0].grid()
        self.__ax_handle[0].set_ylabel("Safety")
        self.__ax_handle[0].set_xlim([self.__time[0], self.__time[-1]])
        self.__ax_handle[0].set_ylim([-0.1, 1.1])

        # force cursor coordinates to not use scientific mode (exponential notation)
        self.__ax_handle[0].format_coord = lambda x, y: f"x={x:.2f}, y={y:.2f}"

        self.__cursor_handles[0], = self.__ax_handle[0].plot([0.0], [0.0], lw=1, color='r', zorder=999)

        self.__safety_score_handle, = self.__ax_handle[0].plot([], [], lw=1, color=TUM_colors['TUM_orange'])

        # setup axis for all metrics
        for i in range(field_cnt):
            self.__ax_handle[i + 1] = axes[i + 1]
            # self.__ax_handle[i + 1].grid()
            self.__ax_handle[i + 1].set_ylabel(field_names[i].replace("_cur", ""))

            # force cursor coordinates to not use scientific mode (exponential notation)
            self.__ax_handle[i + 1].format_coord = lambda x, y: f"x={x:.2f}, y={y:.2f}"

            self.__safety_base_handle[field_names[i]], = \
                self.__ax_handle[i + 1].plot([], [], lw=1, color=TUM_colors['TUM_blue'], label=field_names[i])

            if field_names[i].replace("_cur", "_bound") in self.__safety_base.keys():
                self.__safety_base_handle[field_names[i].replace("_cur", "_bound")], = \
                    self.__ax_handle[i + 1].plot([], [], ':', lw=1, color=TUM_colors['TUM_blue'], label="Bound")

                min_y = min(min(min(self.__safety_base[field_names[i].replace("_cur", "_bound")]),
                                min(self.__safety_base[field_names[i]])), 0.0)
                max_y = max(max(self.__safety_base[field_names[i].replace("_cur", "_bound")]),
                            max(self.__safety_base[field_names[i]]))

            else:
                min_y = min(min(self.__safety_base[field_names[i]]), 0.0)
                max_y = max(self.__safety_base[field_names[i]])

            self.__ax_handle[i + 1].set_xlim([self.__time[0], self.__time[-1]])
            self.__ax_handle[i + 1].set_ylim([min_y - 0.1 * abs(max_y - min_y), max_y + 0.1 * abs(max_y - min_y)])
            self.__cursor_handles[i + 1], = self.__ax_handle[i + 1].plot([0.0], [0.0], lw=1, color='r', zorder=999)

            # self.__ax_handle[i+1].legend(loc='upper right')

        # x-axis label only for last plot
        self.__ax_handle[field_cnt].set_xlabel("t in s")

        # --------------------------------------------------------------------------------------------------------------
        # - DEFINE GUI ELEMENTS ----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Playback button
        plybck_ax = plt.axes([0.8, 0.95, 0.15, 0.04])
        self.__button_plybck = Button(plybck_ax, 'Playback', color=BTN_COLOR, hovercolor='0.975')
        self.__button_plybck.on_clicked(self.on_playback_click)

        # Set time_stamps window as active figure
        plt.figure(self.__fig.number)

        self.update_plot_until()

        plt.draw()
        plt.pause(0.001)

    def update_plot_until(self,
                          end_time: float = np.Inf) -> None:
        """
        Update the plot until the specified time.

        :param end_time:    (optional) final timestamp in series to be visualized

        """

        # find closest time_stamp
        if any(np.array(self.__time) - end_time > 0):
            # get index
            num_plt_elements = np.argmax(np.array(self.__time) - end_time > 0)

        else:
            # plot all
            num_plt_elements = len(self.__time)

        # plot main safety score
        self.__safety_score_handle.set_data([self.__time[:num_plt_elements],
                                             self.__safety_combined[:num_plt_elements]])

        # plot all entries
        for key in self.__safety_base_handle.keys():
            self.__safety_base_handle[key].set_data([self.__time[:len(self.__safety_base[key][:num_plt_elements])],
                                                     self.__safety_base[key][:num_plt_elements]])
        self.__fig.canvas.draw_idle()

    def move_cursor(self,
                    t: float = None) -> None:
        """
        Moves all cursors (in all subplots) to new time value, if set to 'None' all cursors are hidden.

        :param t:   timestamp all cursors are moved to; 'None' hides all cursors

        """

        for key in self.__cursor_handles.keys():
            if t is None:
                self.__cursor_handles[key].set_data([0, 0], [0, 0])
            else:
                self.__cursor_handles[key].set_data([t, t], self.__ax_handle[key].get_ylim())

    def on_playback_click(self, _):
        """
        Whenever the playback button is clicked.
        """

        # disable cursors
        self.move_cursor(t=None)

        # playback the scenario in real time (with updates as fast as possible)
        # get range of x axis
        t_range = [self.__time[0], self.__time[-1]]
        t = t_range[0]

        while t < t_range[1]:
            tic = time.time()

            self.update_plot_until(end_time=t)

            plt.pause(0.001)
            t += time.time() - tic
