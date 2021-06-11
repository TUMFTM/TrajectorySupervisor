=============
Configuration
=============

All relevant configuration files can be found in the '`params`'-folder, located in the root-directory. All the files are
explained in the following.

supmod_config.ini
=================
The '`supmod_config.ini`'-file holds all parameterizable information required for the calculation of a safety score in
the supervisor modules.

The file is split into sections (section headers defined by square brackets, e.g. '`[HEADER]`'). Every supervisor module
should be represented by at least one individual section in the file. Each entry is accompanied by an explanation in
form of a comment. Besides the module-specific sections, the file hosts a '`[GENERAL]`'-section, where generic
information - required by all modules - is listed.

.. hint:: In order to read from the config file, the configparser library is used.

    First, initialize the configparser with the specified conifg-file:

    .. code-block:: python

        import configparser

        param = configparser.ConfigParser()
        if not param.read(repo_path + "/params/supmod_config.ini"):
            raise ValueError('Specified config file does not exist or is empty!')

    Second, retrieve the value of interest (here: '`veh_length`' in the section '`[GENERAL]`'):

    .. code-block:: python

        x_float = param.getfloat('GENERAL', 't_warn')

    In order to load different parameter types, use the corresponding class-methods (e.g. '`.getint()`'). When loading
    complex types - like the '`module_enabled`'-variable, use the json library to decode the variable:

    .. code-block:: python

        import json

        y_dict = json.loads(param.get('GENERAL', 'module_enabled'))

.. hint:: Whether your module should be prepended by 'static' or 'dynamic' depends on the fact whether your module
    depends on the dynamic environment - other objects - or not. If it is dependent, prepend 'dynamic', else use
    'static'.


interface_config_xx.ini
========================
The interface config files hold target-hardware specific or use-case  specific parameterizations. This includes the
following information:

- live-visualization on/off (e.g. not desired on the actual vehicle)
- logging on/off and level (e.g. not desired on a CI-server, but relevant on the vehicle and for debugging purposes)
- en- / disabling supervisor modules
- multiprocessing on/off (only available on linux machines)
- vehicle parameters (e.g. vehicle width)

The files end with a specifier for the relevant target setup (e.g. '`_ci`'), which is handed as a flag to the
main-script. Currently the following targets are listed (to be extended at any time):

- '`ci`' - execution of a CI-job (no logs, no visualization)
- '`sim`' - simulation on own machine with other modules running (logs, no visualization)

.. hint:: By default (when no flag is handed to the main-scirpt), the '`sim`' configuration (execution on vehicle) is
    assumed.

The '`[GENERAL_SAFETY]`'-section hosts one crucial variable, namely '`module_enabled`'. This variable is defined as a
dict and should host activation information for every supervisor module.

With this central parameter it is possible to (de-)activate individual supervisor modules for the safety score.
Furthermore, it is possible to specify this behavior individually for the rating of the performance and emergency
trajectory. The basic structure of this dict is shown below. Further supervisor modules should be added in the same way.

.. code-block:: python

    module_enabled = {"static_dummy": {"perf": false,
                                       "emerg": false},
                      "static_dummy2": {"perf": true,
                                        "emerg": false}}


veh_dyn_info
============
The dynamical behavior of the vehicle for the initially generated velocity profile can be adjusted with the files in the
'`params/veh_dyn_info`'-folder. The '`ax_max_machines.csv`'-file describes the acceleration resources of the motor at
certain velocity thresholds (values in between are interpolated linearly). The '`localgg.csv`'-file describes the
available friction based longitudinal and lateral acceleration at certain velocity thresholds (values in between are
interpolated linearly).
