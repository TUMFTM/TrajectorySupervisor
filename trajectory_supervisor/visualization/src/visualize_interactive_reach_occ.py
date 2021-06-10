import os
import trajectory_planning_helpers as tph
from trajectory_supervisor import supervisor_modules as goc

"""
Script used to open an interactive plot that allows to view the physically possible reachable set and the resulting
guaranteed occupied area. The initial velocity and time horizon can be adapted via the GUI.

:Authors:
    * Tim Stahl <tim.stahl@tum.de>

:Created on:
    10.04.2020
"""

if __name__ == "__main__":
    toppath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    localgg_path = "/params/veh_dyn_info/localgg.csv"
    ax_max_path = "/params/veh_dyn_info/ax_max_machines.csv"

    # retrieve localgg and ax_max_machines
    ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(ax_max_machines_import_path=(toppath
                                                                                               + ax_max_path))[1]
    localgg = tph.import_veh_dyn_info_2.import_veh_dyn_info_2(filepath2localgg=(toppath + localgg_path))

    # calculate guaranteed occupation maps in visual mode, with given below
    goc.guar_occ_calc(t_max=2.0,        # maximum temporal horizon of the reachable-set [in s]
                      d_t=0.1,          # temporal step-width of the reachable- and occupation-set [in s]
                      v_max=50.0,       # maximum velocity of the vehicle to be calculated [in m/s]
                      d_v=0.5,          # velocity step-with to be covered, starting from zero to v_max [in m/s]
                      localgg=localgg,
                      ax_max_mach=ax_max_machines,
                      export_path=None)
