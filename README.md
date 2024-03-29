# Trajectory Supervisor

![Picture Overview Safety Assessment](docs/source/figures/Overview.png)

The safety assessment module is a python-based online verification framework consisting of multiple sub-modules. The
safety checks serve as an independent check for generated motion primitives, with the goal of reducing the risk of
not properly handling safety critical behaviours that did not occur during development. Furthermore, the framework can
be used to pinpoint critical scenarios or time intervals during development.

To serve these purposes, the safety assessment module rates incoming trajectories with regard to
their safety. The trajectories can either be provided directly by the planning module on the target device or for
testing purposes (e.g. CI jobs) via stored scenario files.

### Disclaimer
This software is provided *as-is* and has not been subject to a certified safety validation. Autonomous Driving is a
highly complex and dangerous task. In case you plan to use this software on a vehicle, it is by all means required that
you assess the overall safety of your project as a whole. By no means is this software a replacement for a valid 
safety-concept. See the license for more details.

### Documentation
The documentation of the project can be found [here](https://trajectorysupervisor.readthedocs.io).

### People Involved

##### Core Developer
* [Tim Stahl](mailto:tim.stahl@tum.de)

##### Acknowledgements
Several students contributed to the success of the project during their Bachelor's, Master's or Project Thesis.
* Yujie Lian (RSS)
* Maroua Ben Lakhal (Static Safety)
* Maximilian Bayerlein (Guaranteed Occupation)
* Yves Huberty (Rule-based Reachable Sets)
* Philipp Radecker (Guaranteed Occupation)
* Nils Rack (Guaranteed Occupation)


### Contributions
[1] T. Stahl and F. Diermeyer, 
“Online Verification Enabling Approval of Driving Functions—Implementation for a Planner of an Autonomous Race Vehicle,”
IEEE Open Journal of Intelligent Transportation Systems, vol. 2, pp. 97–110, 2021, doi: 10/gj3535.\
[(view online)](https://ieeexplore.ieee.org/document/9424710)


[2] T. Stahl, M. Eicher, J. Betz, and F. Diermeyer,
“Online Verification Concept for Autonomous Vehicles – Illustrative Study for a Trajectory Planning Module,”
in 2020 IEEE Intelligent Transportation Systems Conference (ITSC), 2020.\
[(view pre-print)](https://arxiv.org/pdf/2005.07740)

If you find our work useful in your research, please consider citing: 
```
@article{stahl2021,
  title = {Online Verification Enabling Approval of Driving Functions \textemdash Implementation for a Planner of an Autonomous Race Vehicle},
  author = {Stahl, Tim and Diermeyer, Frank},
  year = {2021},
  volume = {2},
  pages = {97--110},
  issn = {2687-7813},
  doi = {10/gj3535},
  journal = {IEEE Open Journal of Intelligent Transportation Systems},
 }

@inproceedings{stahl2020,
     title = {Online Verification Concept for Autonomous Vehicles - Illustrative Study for a Trajectory Planning Module},
     booktitle = {2020 IEEE Intelligent Transportation Systems Conference (ITSC)},
     author = {Stahl, Tim and Eicher, Matthis and Betz, Johannes and Diermeyer, Frank},
     year = {2020}
   }
```