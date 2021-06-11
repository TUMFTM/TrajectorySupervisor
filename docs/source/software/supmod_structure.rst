===========================
Supervisor Module Structure
===========================

Supervisor modules (SupMod) are called in the Supervisor class and each calculate an individual safety score (binary)
for the current situation (trajectory set in context with the surrounding environment). Details about the embedding in
Supervisor class and how the data is retrieved can be found in :doc:`basics`. All paths in the following are given in
relation to the folder '`trajectory_supervisor`'.

Each SupMod is situated in its own folder (subdirectory of '`/supervisor_modules`') and must host a python class located
in a python file with the same name as the parent folder.
All (activated) module classes are initialized in the supervisor-class (method '`init_supmods()`') and called in the
'`safety_rating()`' method. The returned boolean is used in the supervisor class to determine an overall safety score.

How to setup a new module
=========================
Each module should be designed like the provided "supmod_dummy" ('`/supervisor_modules/supmod_dummy`'). Thereby, the
contained class must hold a '`calc_score()`'-function that will be called during runtime. All parameters required by the
module should be located in the '`params/supmod_config.ini`'-file (including an option to disable the implemented module
- in total or split into several sub-components). Details on the config file can be found in :doc:`config`.

In short, the following steps setup a basic new SupMod:
    #. Code integration
    #. Config adaption
    #. Documentation

The following subsections detail on each of these steps.

Code integration
----------------
Start by duplicating the folder "supmod_dummy" ('`/supervisor_modules/supmod_dummy`'). Change all occurrences of 'dummy'
to your module name. Your module name should be descriptive (avoid very generic descriptions like 'safety') and not too
long.

Now you can add your code to the class skeleton. The existing functions in the skeleton must remain, but you are free to
add further functions to the class and / or add further python scripts to the directory.

In order to integrate your module in the supervisor class, the following two hooks must be provided:

    #. **Initialization of the SupMod**

       Your module should be initialized within the '`init_supmods()`'-function of the Supervisor class
       ('`/supervisor.py`'). Therefore, add your module initialization in the following style:

       .. code-block:: python

           # dummy module
           if any(self.__module_enabled['static_dummy'].values()):
               self.__mod_dict['dummy'] = supervisor_modules.supmod_dummy.src.supmod_dummy.SupModDummy()

       As shown in this example, the module is only initialized, if activated in the config (for details see
       :doc:`config`). Furthermore, the class instance is added to the dict '`self.__mod_dict`' (provide a unique and
       meaningful key here).

    #. **Triggering the calculation**

       The '`calc_score()`'-function of your module should be called within the '`safety_rating()`'-function of the
       Supervisor class. To do this, stick to the following scheme:

       .. code-block:: python

           # dummy module (example module - executed when activated and initialized)
           if 'dummy' in mod_dict.keys() and mod_enabled['static_dummy'][traj_type]:
               valid_dummy = mod_dict['dummy'].calc_score()

           else:
               valid_dummy = True

       Note, that the score is only calculated when the module is present and enabled, else no safety issues with
       regard of your module are assumed ('True' returned).

       Furthermore, add the score of your module to the conjunction resulting in the corresponding overall safety
       score.

       .. code-block:: python

           # fuse score of all active dynamic env. assessment modules (add more via conjunction)
           valid_dyn_env = valid_dummy and ...


Config adaption
---------------
The created module should be switchable on and off via the config-file. Furthermore, it should retrieve
parameterizations from there. Further details about these details are given in :doc:`config`.


Documentation
-------------
In order to ease readability and further development, your code should be well documented. Therefore, the documentation
should be done in two places:

    #. In the code

       Each class or method should host a detailed explanation (function description, input parameters, output
       parameters, author, date). In order to allow the sphinx to auto-generate the documentation of the code, stick to
       the exact format given in the dummy module.

       The auto-generated documentation will appear here: :doc:`../trajectory_supervisor/modules`.

    #. In the documentation-files

       Since the documentation of classes and methods does not provide an straight-forward insight on the overall
       overall function of the module, a dedicated documentation page within this documentation is desired. In order to
       add a dedicated page, add a '<supmod_your_module>.rst'-file with your module name in the following folder:
       '/docs/source/software/content/supervisor_modules'. Add the documentation of your module to this file.

       The added documentation will appear here: :doc:`supmod_doc`.

       .. hint:: Check the sample documentation of the '`supmod_dummy.py`'-module to get a first impression and further
           tips (:doc:`supervisor_modules/supmod_dummy`).


Further tips for your implementation
====================================

Logging
-------
Logging messages should be published via Python's "logging"-library.

.. code-block:: python

    import logging
    logging.getLogger("supervisor_logger")


Thereby, the following logging levels are available:
    - debugging (``logging.getLogger("supervisor_logger").debug("<msg here>")``) - print debugging messages here not
      relevant for execution and usage of the software
    - info (``logging.getLogger("supervisor_logger").info("<msg here>")``) - print informative messages, relevant for
      execution and usage of the software (e.g. map loaded, parameterization xy, ...)
    - warning (``logging.getLogger("supervisor_logger").warning("<msg here>")``) - print warning information, especially
      used for specifics on detected safety violations (e.g. where and why the trajectory is unsafe) or coding-related
      warnings (e.g. an essential SW part is deactivated / unable to load)


For each message, the following format should be pursued:

.. code-block:: python

    "<SupMod Module Name> | <Traj. Type (if relevant)> | <Message>"

For example:

.. code-block:: python

    "supmod_RSS | Collision with vehicle detected at xy sec into the trajectory!"


helper_funcs
------------
The folder '`/helper_funcs`' holds methods relevant or useful for all modules. If you want to use any function within
this folder, simply import the helper_funcs in your module

.. code-block:: python

    import helper_funcs


If you write a method that may be useful to more than just your module, consider adding it to the helper_funcs folder
instead of your module.
