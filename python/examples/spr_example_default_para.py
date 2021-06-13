import sys
import os
#sys.path.append("/d/01_Projekte/P_02_DataCon/Software/spr/python/sprpy")
sys.path.append(os.getcwd().replace("/examples", ""))

import spr_py

spr = spr_py.SPRPython()
spr.load_joint_location_from_csv("../../data/Xinit_python.csv")
# print(spr.get_joint_location())
spr.load_point_cloud_from_pcd("../../data/test_cloud.pcd", "xyzi")
spr.set_verbose(True)
spr.compute_registered()
spr.save_registered_to_csv("../../data/Xregistered_python.csv")
# print(spr.get_registered_joint_location().shape)