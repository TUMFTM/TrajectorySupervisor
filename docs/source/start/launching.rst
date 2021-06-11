======================
Launching the Software
======================

In order to use the Supervisor, integrate the trajectory_supervisor package in your code. In order to get started,
we provide two example scripts. These scripts demonstrate the code integration and allow to test the Supervisor's
features.

Minimal Example
===============
A minimal example of the Supervisor is parametrized in the '`main_supervisor_example_min.py`' script in the root
directory. Within this example, the Supervisor is executed in the simplest fashion without any configured logging.
The results of the safety rating in every time step of the simulated scenario is printed in the console.
Launch the code with the following command:

.. code-block:: bash

    python3 main_supervisor_example_min.py

Standard Example
================
A more comprehensive example of the Supervisor is given in the '`main_supervisor_example_std.py`' script (also in the
root directory). Within this example, the Supervisor is executed with basic functions:

* Logging to file - the environment and evaluated trajectory with safety rating of every time-stamp are logged to a
  file, which can be visualized with the interactive log-viewer afterwards
* Integration of a trajectory and object sync script (to be used, if objects and trajectories are received
  asynchronously)

The results of the safety rating in every time step of the simulated scenario is printed in the console.
Launch the code with the following command:

.. code-block:: bash

    python3 main_supervisor_example_std.py

.. note:: Logs are configured to be stored in the root folder of this repository. By executing the log viewer without
    any further inputs, the latest log is opened and visualized.

    Launch the log viewer (located in trajectory_supervisor/visualization/src) with the following command:

    .. code-block:: bash

        python3 visualize_safety_log.py


Further Steps
=============
A description for usage of the planner class in your project is given in :doc:`../../software/content/basics`.
Furthermore, the parameterization, development tools and the log visualizer is tackled in that Chapter.
