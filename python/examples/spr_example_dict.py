import sys
import os
sys.path.append(os.getcwd().replace("/examples", ""))
#sys.path.append("/home/siyuchen/Hiwi/spr/build/python/sprpy")

import spr_py

init_paras = {
    "beta" : 0.1,
    "lambda0" : 100,
    "tau0" : 500,
    "nMaxIterations" : 30,
    "tolerance" : 1e-5,
    "kNN" : 30,
    "wOutliers" : 0.1,
    "annealingFactor" : 0.97,
    "normalization" : True
}

spr = spr_py.SPRPython(init_paras)
spr.load_joint_location_from_csv("../../data/Xinit_python.csv")
# print(spr.get_joint_location())
spr.load_point_cloud_from_pcd("../../data/test_cloud.pcd", "xyzi")
spr.set_verbose(True)
spr.compute_registered()
spr.save_registered_to_csv("../../data/Xregistered_python.csv")
# print(spr.get_registered_joint_location().shape)