import matplotlib.pyplot as plt
import numpy as np
import math


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


def timing_histogram(calc_times: dict):
    """
    Plots a histogram for the provided timing information.

    :param calc_times:  dict with keys holding a specific entity / module and values being the associated calculation
                        times per iteration

    """

    # define colors and extract keys
    colors = [TUM_colors['TUM_blue']] * len(calc_times.keys())
    keys = list(calc_times.keys())

    # move "overall" to first place
    if "overall" in keys:
        keys.insert(0, keys.pop(keys.index("overall")))
        colors[0] = TUM_colors['TUM_orange']

    # find maximum value
    print("\n\n#### MEAN CALCULATION TIMES ###")
    upper_limit = 0
    for key in keys:
        upper_limit = max(upper_limit, max(calc_times[key]))

        print("- " + str(key) + ": " + str(round(sum(calc_times[key]) / len(calc_times[key]) * 100000) / 100) + "ms")

    # add 10%, convert to ms and round to next even number
    upper_limit = math.ceil((upper_limit * 1.1) * 1000 / 2.) * 2

    # define bins
    bins = np.linspace(0, upper_limit, int(upper_limit / 2))

    # init plot
    fig, axs = plt.subplots(len(calc_times.keys()), 1, tight_layout=True, sharex=True)
    fig.canvas.set_window_title("Calculation Times")

    for i, key in enumerate(keys):
        axs[i].hist(np.array(calc_times[key]) * 1000, bins, label=key, color=colors[i], edgecolor='black',
                    linewidth=1.2)

        axs[i].legend(loc="upper right")
        axs[i].set_ylabel("Frequency")

    # add x-label and set proper limits
    plt.xlabel("Calculation time in ms")
    plt.xlim(0, upper_limit)


# ----------------------------------------------------------------------------------------------------------------------
# MAIN SCRIPT ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
