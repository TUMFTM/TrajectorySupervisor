import io
import dill
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffcts
from matplotlib.widgets import RadioButtons, Button
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import numpy.matlib as npm

# Show plot in serif font (e.g. for publications)
# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

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


class PlotHandler(object):
    """
    This class provides several functions to plot the debug visualization for safety assessment logs.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        24.01.2020
    """

    def __init__(self,
                 plot_title="Log Visualization Tool") -> None:
        """
        :param plot_title:      string specifying the figure title
        """

        # define canvas
        self.__fig = plt.figure(plot_title)

        # define axes
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 4])

        # -- TIME EVENT PLOT (RUN OVERVIEW) ----------------------------------------------------------------------------
        plt.subplot(gs[0])
        self.__time_ax = self.__fig.gca()
        self.__time_ax.set_ylim([-0.5, 3.5])
        self.__time_ax.set_yticks([0, 1, 2, 3])
        self.__time_ax.set_yticklabels(['DATA', 'INFO', 'WARNING', 'CRITICAL'])

        self.__time_ax2 = self.__time_ax.twinx()
        self.__time_ax2.set_title("Run analysis")
        self.__time_ax2.set_xlabel('$t$ in s')
        self.__time_ax2.set_ylabel('Safety Scores')
        self.__time_ax2.set_ylim([-0.5, 4.5])
        self.__time_ax2.set_yticks([0.5, 2.0, 3.5])
        self.__time_ax2.set_yticklabels(['Static', 'Dynamic', 'Overall'])
        self.__time_ax2.grid()

        # in order to still enable onhover event with twinx
        self.__time_event_ax = self.__time_ax.figure.add_axes(self.__time_ax.get_position(True),
                                                              sharex=self.__time_ax, sharey=self.__time_ax,
                                                              frameon=False)
        self.__time_event_ax.xaxis.set_visible(False)
        self.__time_event_ax.yaxis.set_visible(False)

        # force cursor coordinates to not use scientific mode (exponential notation)
        self.__time_event_ax.format_coord = lambda x, y: f"x={x:.2f}, y={y:.2f}"

        # -- TIME EVENT PLOT (AT CURRENT TIME STEP) --------------------------------------------------------------------
        plt.subplot(gs[1])
        self.__time_rel_ax = self.__fig.gca()
        self.__time_rel_ax.set_title("Time step analysis")
        self.__time_rel_ax.set_xlabel('$t$ in s')
        self.__time_rel_ax.set_ylabel(r'$v_x$ in m/s'
                                      '\n'
                                      r'$a_x+30$ in m/s$^2$')
        self.__time_rel_ax.set_xlim([0.0, 5.0])
        self.__time_rel_ax.set_ylim([0.0, 60.0])
        self.__time_rel_ax.grid()
        self.__time_rel_ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        self.__time_rel_ax.grid(which='major', linestyle='-', linewidth='0.6', color='gray')
        self.__time_rel_ax.minorticks_on()

        self.__time_rel_ax2 = self.__time_rel_ax.twinx()
        self.__time_rel_ax2.set_ylabel(r'$\kappa$ in 1/m'
                                       '\n'
                                       r'$\psi$ in rad/(10$\pi$)')
        self.__time_rel_ax2.set_ylim([-0.1, 0.1])

        red_patch = ptch.Patch(color=TUM_colors['TUM_orange'], label=r'$v_x$')
        grey_patch = ptch.Patch(color=TUM_colors['TUM_grey_dark'], label=r'$a_x$')
        blue_patch = ptch.Patch(color=TUM_colors['TUM_blue'], label=r'$\kappa$')
        green_patch = ptch.Patch(color=TUM_colors['TUM_green'], label=r'$\psi$')
        self.__time_rel_ax2.legend(handles=[red_patch, grey_patch, blue_patch, green_patch])

        # -- MAIN PLOT (MAP OVERVIEW) ----------------------------------------------------------------------------------
        plt.subplot(gs[2])
        self.__main_ax = self.__fig.gca()

        # configure main axis
        self.__main_ax.grid()
        self.__main_ax.set_aspect("equal", "datalim")
        self.__main_ax.set_xlabel("East in m")
        self.__main_ax.set_ylabel("North in m")

        # containers
        self.__highlight_paths = dict()
        self.__highlight_pos = dict()
        self.__highlight_time = dict()
        self.__veh_ptch_hdl = dict()
        self.__polygon_ptch_hdl = dict()
        self.__reach_sets_ptch_hdl = dict()
        self.__track_hdl = dict()
        self.__veh_text_handle = dict()
        self.__text_display = None
        self.__text_display2 = None
        self.__time_annotation = None
        self.__time_rel_line_handle = None

        self.__fig.canvas.mpl_connect('motion_notify_event',
                                      lambda event: self.onhover(event=event))

        self.__text_display = self.__main_ax.text(0.02, 0.95, "", transform=plt.gcf().transFigure)
        self.__text_display2 = self.__main_ax.text(0.8, 0.9, "", transform=plt.gcf().transFigure)

        self.__time_annotation = self.__time_ax.annotate("", xy=(0, 0), xytext=(0.14, 0.95),
                                                         textcoords='figure fraction',
                                                         bbox=dict(boxstyle="round", fc="w"),
                                                         arrowprops=dict(arrowstyle="->"))

        # Radio buttons for traj.-type selection
        rax = self.__fig.add_axes([0.01, 0.89, 0.12, 0.1], facecolor='lightgrey')
        self.__radio = RadioButtons(rax, ('perf', 'emerg'), active=0)
        self.__radio.on_clicked(self.toggled_radio)

        # Open plot window button
        self.__button_plt_wndw = Button(self.__fig.add_axes([0.84, 0.01, 0.15, 0.04]), 'Open Plot',
                                        color='lightgray', hovercolor='0.975')
        self.__button_plt_wndw.on_clicked(self.open_plot_window)

        # event handler variables
        self.__node_plot_marker = None
        self.__edge_plot_marker = None
        self.__annotation = None
        self.__time_marker = None
        self.__lambda_fct = None

        self.__last_t_stamp = 0.0

        # friction plot
        self.__fig_acc = None
        self.__data_acc_valid = None
        self.__data_acc_invalid = None

    def toggled_radio(self, _) -> None:
        """
        Function called when the radio buttons are toggled.

        :param _:       parameter that is handed by event, but unused

        """

        self.force_update()

    def plot_map(self,
                 bound_l: np.ndarray,
                 bound_r: np.ndarray,
                 name: str = 'default') -> None:
        """
        Visualization of the map.

        :param bound_l:     coordinates of the left bound
        :param bound_r:     coordinates of the right bound
        :param name:        string specifier for type to be plotted (if "default": all others will be deleted)

        """

        # update selected track handle (delete first), if "default", delete all
        for key in self.__track_hdl.keys():
            if name == "default" or name in key:
                self.__track_hdl[key].remove()
                del self.__track_hdl[key]

        if bound_l is not None and bound_r is not None:
            # close path, if relevant
            if (np.hypot(bound_l[0, 0] - bound_l[-1, 0], bound_l[0, 1] - bound_l[-1, 1]) < 31.0
                    and np.hypot(bound_r[0, 0] - bound_r[-1, 0], bound_r[0, 1] - bound_r[-1, 1]) < 31.0):
                bound_l = np.vstack((bound_l, bound_l[0, :]))
                bound_r = np.vstack((bound_r, bound_r[0, :]))

            # track patch
            patch_xy = np.vstack((bound_l, np.flipud(bound_r)))
            track_ptch = ptch.Polygon(patch_xy, facecolor="black", alpha=0.2, zorder=1)
            self.__track_hdl['patch_' + name] = self.__main_ax.add_artist(track_ptch)

            # connecting lines
            n_shared_el = int(min(bound_l.size / 2, bound_r.size / 2))
            data_bound_lines = [[], []]
            if n_shared_el > 1:
                for i in range(n_shared_el):
                    data_bound_lines[0].extend([bound_l[i, 0], bound_r[i, 0], None])
                    data_bound_lines[1].extend([bound_l[i, 1], bound_r[i, 1], None])
            self.__track_hdl['normals_' + name], = self.__main_ax.plot(data_bound_lines[0], data_bound_lines[1], '--',
                                                                       color=TUM_colors['TUM_grey_dark'], zorder=5)

            # bound lines
            x = list(bound_l[:, 0])
            y = list(bound_l[:, 1])
            x.append(None)
            y.append(None)
            x.extend(list(bound_r[:, 0]))
            y.extend(list(bound_r[:, 1]))
            self.__track_hdl['bounds_' + name], = self.__main_ax.plot(x, y, "k-", linewidth=1.4, label="Bounds",
                                                                      zorder=6)

    def plot_vehicle(self,
                     pos: np.ndarray,
                     heading: list,
                     width: float,
                     length: float,
                     zorder: int = 10,
                     color_str: str = 'blue',
                     id_in: str = 'default',
                     alpha: float = 1.0) -> None:
        """
        Plot the pose and shape of one or multiple vehicles.

        :param pos:        numpy array holding one or multiple positions
        :param heading:    list of floats holding the heading for each position
        :param width:      width of the vehicle in meters
        :param length:     length of the vehicle in meters
        :param zorder:     (optional) z-order of the vehicle
        :param color_str:  (optional) string specifying the color of the plotted vehicle
        :param id_in:      (optional) string specifying the id (used to delete the same type of vehicle plotted prev.)
        :param alpha:      (optional) transparency setting [0, 1]

        """

        # check if color string defined in TUM colors
        if color_str in TUM_colors.keys():
            color_str = TUM_colors[color_str]

        # force position to be 2-dimensional
        pos = np.atleast_2d(pos)

        # delete highlighted positions with handle
        if id_in in self.__veh_ptch_hdl.keys():
            self.__veh_ptch_hdl[id_in].remove()
            del self.__veh_ptch_hdl[id_in]

            # check for further time instances from the last time-step
            counter = 1
            while id_in + "_i" + str(counter) in self.__veh_ptch_hdl.keys():
                self.__veh_ptch_hdl[id_in + "_i" + str(counter)].remove()
                del self.__veh_ptch_hdl[id_in + "_i" + str(counter)]
                counter += 1

        counter = 0
        # for every position to be plotted
        for head in heading:
            theta = head - np.pi / 2
            pos_i = pos[counter, :]

            bbox = (npm.repmat([[pos_i[0]], [pos_i[1]]], 1, 4)
                    + np.matmul([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]],
                                [[-length / 2, length / 2, length / 2, -length / 2],
                                 [-width / 2, -width / 2, width / 2, width / 2]]))

            patch = np.array(bbox).transpose()
            patch = np.vstack((patch, patch[0, :]))

            # for counter >0 generate further ids:
            handle_id = id_in
            if counter > 0:
                handle_id += "_i" + str(counter)

            plt_patch = ptch.Polygon(patch, facecolor=color_str, linewidth=1, edgecolor='k', zorder=zorder, alpha=alpha)
            self.__veh_ptch_hdl[handle_id] = self.__main_ax.add_artist(plt_patch)

            counter += 1

    def plot_text(self,
                  pos: np.ndarray,
                  heading: list,
                  text_list: list,
                  text_dist: float = 10.0,
                  plot_ith: int = 2,
                  avoid_pos: np.ndarray = None,
                  zorder: int = 100,
                  id_in: str = 'default') -> None:
        """
        Plot text (perpendicular) next to poses. The heading is used to determine in a perpendicular direction the
        offset of the text to the pose (allows to plot text next to certain poses on a spline). If empty arrays are
        provided, all previous instances with the specified 'id_in' will be removed.

        :param pos:        numpy array holding one or multiple positions  (columns x, y; each row a pose)
        :param heading:    list of floats holding the heading for each position (if only one pose, list with one float)
        :param text_list:  list of texts to be printed next to each pose (if only one pose, list with one text)
        :param text_dist:  (optional) distance the text should be away from 'pos' in m
        :param plot_ith:   (optional) plot only the text for every i-th pose (e.g. if '1' every pose, if '2' every 2nd)
        :param avoid_pos:  (optional) numpy array of positions to be avoided by text
                                      -> Select favored side (left/right) of heading
        :param zorder:     (optional) z-order of the vehicle
        :param id_in:      (optional) string specifying the id (used to delete same type of vehicle plotted previously)
        :return:
        """

        # force position to be 2-dimensional
        pos = np.atleast_2d(pos)

        # delete highlighted positions with handle
        if id_in in self.__veh_text_handle.keys():
            self.__veh_text_handle[id_in].remove()
            del self.__veh_text_handle[id_in]

            # check for further time instances from the last time-step
            counter = 0 + plot_ith
            while id_in + "_i" + str(counter) in self.__veh_text_handle.keys():
                self.__veh_text_handle[id_in + "_i" + str(counter)].remove()
                del self.__veh_text_handle[id_in + "_i" + str(counter)]
                counter += plot_ith

        # return, if no data provided
        if not heading:
            return

        # check which side of heading occupies less instances of "avoid_pos"
        theta_l = np.array(heading) + np.pi / 2
        theta_r = np.array(heading) - np.pi / 2

        pos_l = np.column_stack((pos[:, 0] - np.sin(theta_l) * text_dist, pos[:, 1] + np.cos(theta_l) * text_dist))
        pos_r = np.column_stack((pos[:, 0] - np.sin(theta_r) * text_dist, pos[:, 1] + np.cos(theta_r) * text_dist))

        if avoid_pos is not None:
            # force position to be 2-dimensional
            avoid_pos = np.atleast_2d(avoid_pos)

            # calculate distances between all combinations of pos and avoid pos
            dist_l = euclidean_distances(pos_l, avoid_pos)
            dist_r = euclidean_distances(pos_r, avoid_pos)

            # get number of points that are smaller than the plot text distance
            num_dist_l = sum(i < text_dist for i in dist_l.min(axis=1))
            num_dist_r = sum(i < text_dist for i in dist_r.min(axis=1))

            if num_dist_l < num_dist_r:
                pos_text = pos_l
            else:
                pos_text = pos_r
        else:
            pos_text = pos_l

        counter = 0
        # for every position to be plotted
        for head, text in zip(heading, text_list):
            if counter % plot_ith != 0:
                counter += 1
                continue

            pos_i = pos_text[counter, :]

            # for counter >0 generate further ids:
            handle_id = id_in
            if counter > 0:
                handle_id += "_i" + str(counter)

            self.__veh_text_handle[handle_id] = self.__main_ax.text(pos_i[0],
                                                                    pos_i[1],
                                                                    text,
                                                                    rotation=0.0,
                                                                    verticalalignment="center",
                                                                    horizontalalignment="center",
                                                                    clip_on=True,
                                                                    zorder=zorder)
            self.__veh_text_handle[handle_id].set_path_effects([patheffcts.withStroke(linewidth=2, foreground='w')])

            counter += 1

    def plot_polygon(self,
                     polygon_list: list,
                     zorder: int = 10,
                     color_str: str = 'blue',
                     color_e_str: str = 'k',
                     id_in: str = 'default',
                     alpha: float = 0.5) -> None:
        """
        Plot one / multiple polygon patches.

        :param polygon_list: list of polygon coordinates (each given as an np.ndarray with columns [x, y])
        :param zorder:       (optional) z-order of the vehicle
        :param color_str:    (optional) string specifying the color of the plotted vehicle
        :param color_e_str:  (optional) string specifying the color of the outline for the plotted vehicle
        :param id_in:        (optional) string specifying the id (used to delete the same type of vehicle plotted prev.)
        :param alpha:        (optional) transparency setting [0, 1]

        """
        # check if color string defined in TUM colors
        if color_str in TUM_colors.keys():
            color_str = TUM_colors[color_str]

        # Delete highlighted reachable sets with handle
        if id_in in self.__polygon_ptch_hdl.keys():
            self.__polygon_ptch_hdl[id_in].remove()
            del self.__polygon_ptch_hdl[id_in]

            # Check for further time instances from the last time-step
            counter = 1
            while id_in + '_i' + str(counter) in self.__polygon_ptch_hdl.keys():
                self.__polygon_ptch_hdl[id_in + "_i" + str(counter)].remove()
                del self.__polygon_ptch_hdl[id_in + "_i" + str(counter)]
                counter += 1

        counter = 0
        # For every polygon in list
        for polygon_coords in polygon_list:
            # for counter >0 generate further ids:
            handle_id = id_in
            if counter > 0:
                handle_id += "_i" + str(counter)

            plt_patch = ptch.Polygon(polygon_coords, closed=True, facecolor=color_str, alpha=alpha, zorder=zorder,
                                     linewidth=1, edgecolor=color_e_str, fill=(color_str is not None))

            self.__polygon_ptch_hdl[handle_id] = self.__main_ax.add_artist(plt_patch)

            counter += 1

    def update_text_field(self,
                          text_str: str,
                          color_str: str = 'k',
                          text_field_id: int = 1) -> None:
        """
        Update the text field in the plot window

        :param text_str:        text string to be displayed
        :param color_str:       (optional) string specifying the color of the plotted text
        :param text_field_id:   (optional) id specifying the text field to be updated (default: 1)

        """

        if text_field_id == 1:
            self.__text_display.set_text(text_str)
            self.__text_display.set_color(color_str)
        elif text_field_id == 2:
            self.__text_display2.set_text(text_str)
            self.__text_display2.set_color(color_str)
        else:
            print("No text_field_id '%i' defined!" % text_field_id)

    def highlight_pos(self,
                      pos_coords: list,
                      color_str: str = 'y',
                      zorder: int = 10,
                      radius=None,
                      marker: str = 'o',
                      id_in: str = 'default') -> None:
        """
        Highlight a position with an plot marker ('´marker´') or a circle with a parametrizable radius ('`radius`').
        If '`radius`' is not provided or 'None', the marker option is used.

        :param pos_coords:  coordinates of the position ot be highlighted
        :param color_str:   (optional) string specifying the color of the highlight
        :param zorder:      (optional) z-order of the highlight marker (default: 10)
        :param radius:      (optional) radius of the marker to be displayed (if not provided, a standard marker is used)
        :param marker:      (optional) string specifying the marker style
        :param id_in:       (optional) string specifying the id (used to delete the same type of vehicle plotted prev.)

        """

        # check if color string defined in TUM colors
        if color_str in TUM_colors.keys():
            color_str = TUM_colors[color_str]

        # delete highlighted positions with handle
        if id_in in self.__highlight_pos.keys():
            self.__highlight_pos[id_in].remove()
            del self.__highlight_pos[id_in]

        # plot pos
        if pos_coords:
            if radius is None:
                self.__highlight_pos[id_in], = \
                    self.__main_ax.plot(pos_coords[0], pos_coords[1], 's', color=color_str, marker=marker,
                                        zorder=zorder)
            else:
                plt_circle = plt.Circle(tuple(pos_coords), radius, color=color_str, fill=True, zorder=zorder)
                self.__highlight_pos[id_in] = self.__main_ax.add_artist(plt_circle)

    def highlight_time(self,
                       time_coords: list,
                       color_str: str = 'y',
                       zorder: int = 10,
                       marker: str = 'o',
                       id_in: str = 'default') -> None:
        """
        Highlight a position with an plot marker ('´marker´') in the temporal plot.

        :param time_coords: coordinates of the time entry to be highlighted
        :param color_str:   (optional) string specifying the color of the highlight
        :param zorder:      (optional) z-order of the highlight marker (default: 10)
        :param marker:      (optional) string specifying the marker style
        :param id_in:       (optional) string specifying the id (used to delete the same type of vehicle plotted prev.)

        """

        # check if color string defined in TUM colors
        if color_str in TUM_colors.keys():
            color_str = TUM_colors[color_str]

        # delete highlighted positions with handle
        if id_in in self.__highlight_time.keys():
            self.__highlight_time[id_in].remove()
            del self.__highlight_time[id_in]

        # plot pos
        if time_coords:
            self.__highlight_time[id_in], = \
                self.__time_rel_ax.plot(time_coords[0], time_coords[1], 's', color=color_str, marker=marker,
                                        zorder=zorder)

    def highlight_path(self,
                       path_coords: np.ndarray,
                       id_in: str = 'default',
                       color_str: str = 'red',
                       linewidth: float = 1.4,
                       zorder: int = 99) -> None:
        """
        Highlight a given coordinate sequence.

        :param path_coords: coordinates of path to be plotted (each coordinate in separate column)
        :param id_in:       (optional) id used for the handle (plots with same id will be removed before plot of next)
        :param color_str:   (optional) color to be used for path highlight
        :param linewidth:   (optional) linewidth of the highlighted path (default: 1.4)
        :param zorder:      (optional) z-order of the highlighted path (default: 99)

        """

        # check if color string defined in TUM colors
        if color_str in TUM_colors.keys():
            color_str = TUM_colors[color_str]

        # delete highlighted paths with handle
        if id_in in self.__highlight_paths.keys():
            self.__highlight_paths[id_in].remove()
            del self.__highlight_paths[id_in]

        # plot the spline
        if path_coords.size > 0:
            self.__highlight_paths[id_in], = self.__main_ax.plot(path_coords[:, 0], path_coords[:, 1], color=color_str,
                                                                 linewidth=linewidth, label="Local Path", zorder=zorder)

    def plot_timeline_stamps(self,
                             time_stamps: list,
                             types: list,
                             lambda_fct) -> None:
        """
        Plot a timeline information for a log file

        :param time_stamps:     list of float values holding time in seconds
        :param types:           list of strings indicating the type of the provided timestamp
        :param lambda_fct:      lambda function to be handed (to be called when mouse hovers over time-line)

        """

        type_names = ["DATA", "INFO", "WARNING", "CRITICAL", "DEBUG"]
        type_marker = {"DATA": "gx", "INFO": "bx", "WARNING": "yx", "CRITICAL": "rx", "DEBUG": "cx"}

        for i, type_name in enumerate(type_names):
            if type_name in types:
                rel_time_stamps = [x for x, t in zip(time_stamps, types) if t == type_name]
                self.__time_ax.plot(rel_time_stamps, np.zeros((len(rel_time_stamps), 1)) + i, type_marker[type_name])

        time_line_marker, = self.__time_ax.plot([], [], 'r-')
        self.set_time_markers(time_line_marker=time_line_marker,
                              lambda_fct=lambda_fct)

    def plot_timeline_course(self,
                             line_coords_list: list) -> None:
        """
        Plot three curves in the temporal information subplot.

        :param line_coords_list:    list of cuvre courses

        """

        # color masks
        color = [TUM_colors['TUM_green'], TUM_colors['TUM_blue'], TUM_colors['TUM_orange']]
        group_axes = [self.__time_ax2, self.__time_ax2, self.__time_ax2]

        # plot lines in array (plot reversed, such that the last option is on the lowest layer in the plot)
        for idx, line_coords in enumerate(line_coords_list):
            line_coords = np.array(line_coords)
            group_axes[idx].plot(line_coords[:, 0], line_coords[:, 1], color[idx], linewidth=1.4, label="Local Path",
                                 zorder=0)

    def highlight_timeline(self,
                           time_stamp: float,
                           type_in: str,
                           message: str) -> None:
        """
        Display a text message from the log file, with an arrow pointing to the relevant timestamp.

        :param time_stamp:      timestamp the arrow should point to
        :param type_in:         string specifying the type of error (relevant for color)
        :param message:         string of message to be displayed

        """

        type_names = ["DATA", "INFO", "WARNING", "CRITICAL", "DEBUG"]
        type_marker = {"DATA": "g", "INFO": "b", "WARNING": "y", "CRITICAL": "r", "DEBUG": "c"}

        pos = [time_stamp, type_names.index(type_in)]
        self.__time_annotation.xy = pos

        self.__time_annotation.set_text(message)
        self.__time_annotation.get_bbox_patch().set_facecolor(type_marker[type_in])
        self.__time_annotation.get_bbox_patch().set_alpha(0.4)

        self.__time_annotation.get_visible()

    def plot_time_rel_line(self,
                           line_coords_list: list) -> None:
        """
        Highlight a list of given coordinate sequences (each with a different color)

        :param line_coords_list:  list of lists (grouped sets) of lists holding the coordinates of paths each

        """

        # color masks
        color_masks = [TUM_colors['TUM_orange'], TUM_colors['TUM_grey_dark'],
                       TUM_colors['TUM_blue'], TUM_colors['TUM_green']]
        group_axes = [self.__time_rel_ax, self.__time_rel_ax, self.__time_rel_ax2, self.__time_rel_ax2]

        # delete existing time courses with stored handle
        if self.__time_rel_line_handle is not None:
            for handle in self.__time_rel_line_handle:
                handle.remove()

        #
        self.__time_rel_line_handle = []

        # plot lines in array (plot reversed, such that the last option is on the lowest layer in the plot)
        for group_idx, group_data in enumerate(line_coords_list):
            for idx, line_coords in enumerate(reversed(group_data)):
                # generate a color for the line
                fade_in_clr = (idx + 1) / len(group_data)

                # plot the spline
                line_coords = np.array(line_coords)
                temp_handle, = group_axes[group_idx].plot(line_coords[:, 0], line_coords[:, 1],
                                                          color=color_masks[group_idx],
                                                          linewidth=1.4, label="Local Path", zorder=99 + idx,
                                                          alpha=fade_in_clr)
                # append handle to array
                self.__time_rel_line_handle.append(temp_handle)

    def init_acc_plot(self,
                      a_lat_max_tires: float,
                      a_lon_max_tires: float,
                      dyn_model_exp: float) -> None:
        """
        Initialize acceleration data plot (friction circle)

        :param a_lat_max_tires:     maximum allowed pure lateral acceleration
        :param a_lon_max_tires:     maximum allowed pure longitudinal acceleration
        :param dyn_model_exp:       dynamic model exponent

        """

        if self.__fig_acc is None:
            self.__fig_acc = plt.figure("Acceleration Analysis")

        else:
            plt.figure(self.__fig_acc.number)

        # plot bounds
        a_lat_tmp = np.linspace(0, a_lat_max_tires, 100)
        a_lon_tmp = a_lon_max_tires * np.power(1.0 - np.power(a_lat_tmp / a_lat_max_tires, dyn_model_exp),
                                               1.0 / dyn_model_exp)
        a_lat_plt = np.concatenate((a_lat_tmp, [None], -a_lat_tmp, [None], -a_lat_tmp, [None], a_lat_tmp))
        a_lon_plt = np.concatenate((a_lon_tmp, [None], a_lon_tmp, [None], -a_lon_tmp, [None], -a_lon_tmp))

        plt.plot(a_lat_plt, a_lon_plt, color=TUM_colors['TUM_blue'], label='Acc. limits')

        # generate data plot handles
        self.__data_acc_valid, = plt.plot([], [], 'x', color=TUM_colors['TUM_green'], label='Acc. limits respected')
        self.__data_acc_invalid, = plt.plot([], [], 'x', color=TUM_colors['TUM_orange'], label='Acc. limits violated')

        # plot setup
        plt.grid()
        plt.axis('equal')
        plt.xlabel('Lateral acceleration in m/s²')
        plt.ylabel('Longitudinal acceleration m/s²')
        plt.legend(loc='upper left')
        plt.draw()

        # set main figure as active figure
        plt.figure(self.__fig.number)

    def update_acc_plot(self,
                        acc_limit_valid: list,
                        acc_limit_invalid: list) -> None:
        """
        Update acceleration data in friction circle plot.

        :param acc_limit_valid:     acceleration coordinates (lat / long) holding an valid entry, plotted in green
        :param acc_limit_invalid:   acceleration coordinates (lat / long) holding an invalid entry, plotted in red
        """

        # if acceleration plot is not initialized yet
        if self.__fig_acc is None:
            print("Could not update plot, since acceleration plot was not initialized")
            return

        # set acceleration figure as active figure
        plt.figure(self.__fig_acc.number)

        # update data
        self.__data_acc_valid.set_data(acc_limit_valid[0], acc_limit_valid[1])
        self.__data_acc_invalid.set_data(acc_limit_invalid[0], acc_limit_invalid[1])

        # set main figure as active figure
        plt.figure(self.__fig.number)

    # ------------------------------------------------------------------------------------------------------------------
    # - EVENTS ---------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def open_plot_window(self,
                         _) -> None:
        """
        Called when the 'Open Plot'-button is clicked.
        """

        # hide text displays
        self.__text_display.set_visible(False)
        self.__text_display2.set_visible(False)

        # dump the whole plot in a pickle
        inx = list(self.__fig.axes).index(self.__main_ax)
        buf = io.BytesIO()
        dill.dump(self.__fig, buf)
        buf.seek(0)

        # load pickle in new plot figure (without buttons)
        fig_plot = dill.load(buf)
        fig = plt.gcf()
        fig.set_size_inches(3.8, 5.0, forward=True)  # 3.5in is IEEE double-column width
        fig_plot.set_tight_layout(True)

        # delete everything except main axes
        for i, ax in enumerate(fig_plot.axes):
            if i != inx:
                fig_plot.delaxes(ax)

        fig_plot.show()

        # un-hide text displays
        self.__text_display.set_visible(True)
        self.__text_display2.set_visible(True)

    def set_time_markers(self,
                         time_line_marker,
                         lambda_fct) -> None:
        """
        Store the timeline-marker and event lambda function.

        :param time_line_marker:    timeline-marker to be stored
        :param lambda_fct:          lambda function ot be stored

        """

        self.__time_marker = time_line_marker
        self.__lambda_fct = lambda_fct

    def onhover(self, event):
        """
        Function called, when the mouse is moved over the plot window.

        :param event:   mouse event (also holding the position of the pointer)

        """
        if self.__time_ax.get_window_extent().contains(event.x, event.y) and event.xdata is not None:
            self.__time_marker.set_data([event.xdata, event.xdata], self.__time_ax.get_ylim())
            self.__fig.canvas.draw()

            self.__lambda_fct(event.xdata, self.__radio.value_selected)
            self.__last_t_stamp = event.xdata

        else:
            self.__lambda_fct(self.__last_t_stamp, self.__radio.value_selected)

    def force_update(self) -> None:
        """
        This function triggers an forced update.

        """

        self.__lambda_fct(self.__last_t_stamp, self.__radio.value_selected)

    @staticmethod
    def show_plot(non_blocking=False):
        if non_blocking:
            plt.draw()
            plt.pause(0.0001)
        else:
            plt.show()
