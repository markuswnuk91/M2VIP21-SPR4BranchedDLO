import sys
import numpy as np
import os

# sys.path.append("/home/siyuchen/Hiwi/spr/build/python/sprpy")

sys.path.append(os.getcwd().replace("/examples", ""))
import spr_py

print(os.getcwd())
pointcloud = spr_py.read_pointcloud_from_pcd("../../data/test_cloud.pcd")
#print(type(pointcloud))

X_init = np.genfromtxt('../../data/Xinit_python.csv',delimiter=',')
#print(type(X_init))

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
# consturct branchArray B
# B groups indices of centroids belonging to one branch together in a list of lists. 
# each inner list contains the indices belonging to one branch.
B = [np.arange(0, 46, 1),np.arange(46, 62, 1),np.arange(62, 82, 1),np.arange(82, 89, 1)]
spr = spr_py.SPRBranchedPython(init_paras)

spr.load_joint_location_from_np(X_init)
# print(spr.get_joint_location())
spr.load_point_cloud_from_np(pointcloud)
spr.load_branch_array_from_np(B)
spr.set_verbose(True)
#B_return = spr.get_branch_array()
#print(B_return)
spr.compute_sprbranched()
spr.save_registered_to_csv("../../data/Xregistered_python.csv")
# print(spr.get_registered_joint_location().shape)