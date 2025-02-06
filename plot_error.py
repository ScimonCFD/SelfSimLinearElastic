# License
#  This program is free software: you can redistribute it and/or modify 
#  it under the teMSE of the GNU General Public License as published 
#  by the Free Software Foundation, either version 3 of the License, 
#  or (at your option) any later version.

#  This program is distributed in the hope that it will be useful, 
#  but WITHOUT ANY WARRANTY; without even the implied warranty of 
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

#  See the GNU General Public License for more details. You should have 
#  received a copy of the GNU General Public License along with this 
#  program. If not, see <https://www.gnu.org/licenses/>. 

# Description
#  This is a postprocessing tool. It creates plots for visualising the results
#  from the selfSimulation algorithm. 

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import numpy as np
# from auxiliary_functions import deserialise, terminal
from distutils.dir_util import mkpath
# import input_file
# from input_file import *
from joblib import load
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

mkpath("./Comparison/Plots/")
# terminal("cp -r " + ROUTE_NN_MODEL + "loadpass* " + ROUTE_NN_MODEL + 
#          "Results/")
# master_folder_NN = ROUTE_NN_MODEL + "Results/"

# LOAD_INC_INTEREST = 15 #9
# TOTAL_LOAD_INCREMENTS = 15
# TOTAL_NUMBER_PASSES = 10
# DELTA_PASSES = 1

sigma_Expected_List = []
epsilon_Expected_List = []
D_Expected_List = []

sigma_SelfSim_LoadInc1_List = []
epsilon_SelfSim_LoadInc1_List = []
D_SelfSim_LoadInc1_List = []

sigma_SelfSim_LoadInc5_List = []
epsilon_SelfSim_LoadInc5_List = []
D_SelfSim_LoadInc5_List = []

sigma_SelfSim_LoadInc15_List = []
epsilon_SelfSim_LoadInc15_List = []
D_SelfSim_LoadInc15_List = []

sigma_OneStepSelfSim_LoadInc1_List = []
epsilon_OneStepSelfSim_LoadInc1_List = []
D_OneStepSelfSim_LoadInc1_List = []

sigma_OneStepSelfSim_LoadInc5_List = []
epsilon_OneStepSelfSim_LoadInc5_List = []
D_OneStepSelfSim_LoadInc5_List = []

sigma_OneStepSelfSim_LoadInc15_List = []
epsilon_OneStepSelfSim_LoadInc15_List = []
D_OneStepSelfSim_LoadInc15_List = []


MSE_epsilon_loadInc1_SelfSim_coord_List = []
MSE_epsilon_loadInc1_SelfSim_coord_List = []
MSE_epsilon_loadInc1_SelfSim_coord_List = []


# # Set the default text font size
# plt.rc('font', size=16)# Set the axes title font size
# plt.rc('axes', titlesize=20)# Set the axes labels font size
# plt.rc('axes', labelsize=20)# Set the font size for x tick labels
# plt.rc('xtick', labelsize=16)# Set the font size for y tick labels
# plt.rc('ytick', labelsize=16)# Set the legend font size
# plt.rc('legend', fontsize=18)# Set the font size of the figure title
# plt.rc('figure', titlesize=24)


TOTAL_LOAD_PASSES = 10
LOAD_INCREMENTS_LIST = ["1", "5", "15"]

def deserialise(route_to_file, name_file):
    #This function imports a npy file
    with open(route_to_file  +  name_file + '.npy', 'rb') as f:
        temp = np.load(f, allow_pickle=True)
    f.close()
    return temp

# Just to get the dimensionality
sigma_Expected = deserialise("./ExpectedResults/" + str(1) + "/",  "sigma")

# Create empty arrays
sigma_expected_total = np.zeros([len(LOAD_INCREMENTS_LIST),
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])

epsilon_expected_total = np.zeros([len(LOAD_INCREMENTS_LIST),
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])


sigma_SelfSim_total_LoadInc1 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])
epsilon_SelfSim_total_LoadInc1 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])

sigma_SelfSim_total_LoadInc5 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])
epsilon_SelfSim_total_LoadInc5 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])

sigma_SelfSim_total_LoadInc15 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])
epsilon_SelfSim_total_LoadInc15 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])





for i in range(len(LOAD_INCREMENTS_LIST)):
    sigma_Expected = deserialise("./ExpectedResults/" + LOAD_INCREMENTS_LIST[int(i)] + "/",  "sigma")
    sigma_expected_total[i, :, :] = sigma_Expected
    
    epsilon_Expected = deserialise("./ExpectedResults/" + LOAD_INCREMENTS_LIST[int(i)] + "/",  "epsilon")
    epsilon_expected_total[i, :, :] = epsilon_Expected


for i in range(TOTAL_LOAD_PASSES):
    sigma_SelfSim_total_LoadInc1[i, :, :] = deserialise("./SelfSim/loadpass" + str(i+1) + "_loadInc1/" + "1a/", "sigma")
    epsilon_SelfSim_total_LoadInc1[i, :, :] = deserialise("./SelfSim/loadpass" + str(i+1) + "_loadInc1/" + "1a/", "epsilon")

    sigma_SelfSim_total_LoadInc5[i, :, :] = deserialise("./SelfSim/loadpass" + str(i+1) + "_loadInc5/" + "5a/", "sigma")
    epsilon_SelfSim_total_LoadInc5[i, :, :] = deserialise("./SelfSim/loadpass" + str(i+1) + "_loadInc5/" + "5a/", "epsilon")


    sigma_SelfSim_total_LoadInc15[i, :, :] = deserialise("./SelfSim/loadpass" + str(i+1) + "_loadInc15/" + "15a/", "sigma")
    epsilon_SelfSim_total_LoadInc15[i, :, :] = deserialise("./SelfSim/loadpass" + str(i+1) + "_loadInc15/" + "15a/", "epsilon")


error_sigma_SelfSim_total_LoadInc1 = sigma_SelfSim_total_LoadInc1 - sigma_expected_total[0, :, :]
error_epsilon_SelfSim_total_LoadInc1 = epsilon_SelfSim_total_LoadInc1 - epsilon_expected_total[0, :, :]


error_sigma_SelfSim_total_LoadInc5 = sigma_SelfSim_total_LoadInc5 - sigma_expected_total[1, :, :]
error_epsilon_SelfSim_total_LoadInc5 = epsilon_SelfSim_total_LoadInc5 - epsilon_expected_total[1, :, :]

error_sigma_SelfSim_total_LoadInc15 = sigma_SelfSim_total_LoadInc15 - sigma_expected_total[2, :, :]
error_epsilon_SelfSim_total_LoadInc15 = epsilon_SelfSim_total_LoadInc15 - epsilon_expected_total[2, :, :]



mean_L2_error_sigma_SelfSim_total_LoadInc1 = (1/sigma_expected_total.shape[1]) * np.sum(((error_sigma_SelfSim_total_LoadInc1)**2), axis = 1)**0.5
mean_L2_error_sigma_SelfSim_total_LoadInc5 = (1/sigma_expected_total.shape[1]) * np.sum(((error_sigma_SelfSim_total_LoadInc5)**2), axis = 1)**0.5
mean_L2_error_sigma_SelfSim_total_LoadInc15 = (1/sigma_expected_total.shape[1]) * np.sum(((error_sigma_SelfSim_total_LoadInc15)**2), axis = 1)**0.5

mean_L2_error_epsilon_SelfSim_total_LoadInc1 = (1/sigma_expected_total.shape[1]) * np.sum(((error_epsilon_SelfSim_total_LoadInc1)**2), axis = 1)**0.5
mean_L2_error_epsilon_SelfSim_total_LoadInc5 = (1/sigma_expected_total.shape[1]) * np.sum(((error_epsilon_SelfSim_total_LoadInc5)**2), axis = 1)**0.5
mean_L2_error_epsilon_SelfSim_total_LoadInc15 = (1/sigma_expected_total.shape[1]) * np.sum(((error_epsilon_SelfSim_total_LoadInc15)**2), axis = 1)**0.5









sigma_OneStepSelfSim_total_LoadInc1 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])
epsilon_OneStepSelfSim_total_LoadInc1 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])

sigma_OneStepSelfSim_total_LoadInc5 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])
epsilon_OneStepSelfSim_total_LoadInc5 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])

sigma_OneStepSelfSim_total_LoadInc15 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])
epsilon_OneStepSelfSim_total_LoadInc15 = np.zeros([TOTAL_LOAD_PASSES,
                                 sigma_Expected.shape[0],
                                 sigma_Expected.shape[1]])



for i in range(TOTAL_LOAD_PASSES):
    sigma_OneStepSelfSim_total_LoadInc1[i, :, :] = deserialise("./OneStepSelfSim/loadpass" + str(i+1) + "_loadInc1/" + "1/", "sigma")
    epsilon_OneStepSelfSim_total_LoadInc1[i, :, :] = deserialise("./OneStepSelfSim/loadpass" + str(i+1) + "_loadInc1/" + "1/", "epsilon")

    sigma_OneStepSelfSim_total_LoadInc5[i, :, :] = deserialise("./OneStepSelfSim/loadpass" + str(i+1) + "_loadInc5/" + "5/", "sigma")
    epsilon_OneStepSelfSim_total_LoadInc5[i, :, :] = deserialise("./OneStepSelfSim/loadpass" + str(i+1) + "_loadInc5/" + "5/", "epsilon")


    sigma_OneStepSelfSim_total_LoadInc15[i, :, :] = deserialise("./OneStepSelfSim/loadpass" + str(i+1) + "_loadInc15/" + "15/", "sigma")
    epsilon_OneStepSelfSim_total_LoadInc15[i, :, :] = deserialise("./OneStepSelfSim/loadpass" + str(i+1) + "_loadInc15/" + "15/", "epsilon")


error_sigma_OneStepSelfSim_total_LoadInc1 = sigma_OneStepSelfSim_total_LoadInc1 - sigma_expected_total[0, :, :]
error_epsilon_OneStepSelfSim_total_LoadInc1 = epsilon_OneStepSelfSim_total_LoadInc1 - epsilon_expected_total[0, :, :]


error_sigma_OneStepSelfSim_total_LoadInc5 = sigma_OneStepSelfSim_total_LoadInc5 - sigma_expected_total[1, :, :]
error_epsilon_OneStepSelfSim_total_LoadInc5 = epsilon_OneStepSelfSim_total_LoadInc5 - epsilon_expected_total[1, :, :]

error_sigma_OneStepSelfSim_total_LoadInc15 = sigma_OneStepSelfSim_total_LoadInc15 - sigma_expected_total[2, :, :]
error_epsilon_OneStepSelfSim_total_LoadInc15 = epsilon_OneStepSelfSim_total_LoadInc15 - epsilon_expected_total[2, :, :]



mean_L2_error_sigma_OneStepSelfSim_total_LoadInc1 = (1/sigma_expected_total.shape[1]) * np.sum(((error_sigma_OneStepSelfSim_total_LoadInc1)**2), axis = 1)**0.5
mean_L2_error_sigma_OneStepSelfSim_total_LoadInc5 = (1/sigma_expected_total.shape[1]) * np.sum(((error_sigma_OneStepSelfSim_total_LoadInc5)**2), axis = 1)**0.5
mean_L2_error_sigma_OneStepSelfSim_total_LoadInc15 = (1/sigma_expected_total.shape[1]) * np.sum(((error_sigma_OneStepSelfSim_total_LoadInc15)**2), axis = 1)**0.5

mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc1 = (1/sigma_expected_total.shape[1]) * np.sum(((error_epsilon_OneStepSelfSim_total_LoadInc1)**2), axis = 1)**0.5
mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc5 = (1/sigma_expected_total.shape[1])* np.sum(((error_epsilon_OneStepSelfSim_total_LoadInc5)**2), axis = 1)**0.5
mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc15 = (1/sigma_expected_total.shape[1]) * np.sum(((error_epsilon_OneStepSelfSim_total_LoadInc15)**2), axis = 1)**0.5



# Set the default text font size
plt.rc('font', size=35)# Set the axes title font size
plt.rc('axes', titlesize=35)# Set the axes labels font size
plt.rc('axes', labelsize=35)# Set the font size for x tick labels
plt.rc('xtick', labelsize=35)# Set the font size for y tick labels
plt.rc('ytick', labelsize=35)# Set the legend font size
plt.rc('legend', fontsize=35)# Set the font size of the figure title
plt.rc('figure', titlesize=35)

####################
fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_SelfSim_total_LoadInc1[:, 0]/1e3, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_OneStepSelfSim_total_LoadInc1[:, 0]/1e3, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
# plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_SelfSim_total_LoadInc5[:, 0]/1e3, color = "blue", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
# plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_OneStepSelfSim_total_LoadInc5z[:, 0]/1e3, color = "yellow", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)

plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Stress $(KPa)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_Sigma_xx_LoadInc1.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_SelfSim_total_LoadInc1[:, 1]/1e3, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_OneStepSelfSim_total_LoadInc1[:, 1]/1e3, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Stress $(KPa)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_Sigma_xy_LoadInc1.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_SelfSim_total_LoadInc1[:, 3]/1e3, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_OneStepSelfSim_total_LoadInc1[:, 3]/1e3, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Stress $(KPa)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_Sigma_yy_LoadInc1.png", bbox_inches='tight')
plt.close(fig)


fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_SelfSim_total_LoadInc1[:, 0]/1e-6, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc1[:, 0]/1e-6, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Strain $(\mu\ mm/mm)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_epsilon_xx_LoadInc1.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_SelfSim_total_LoadInc1[:, 1]/1e-6, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc1[:, 1]/1e-6, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Strain $(  \mu\ mm/mm)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_epsilon_xy_LoadInc1.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_SelfSim_total_LoadInc1[:, 3]/1e-6, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc1[:, 3]/1e-6, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Strain $(  \mu\ mm/mm)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_epsilon_yy_LoadInc1.png", bbox_inches='tight')
plt.close(fig)
############################





#####################################
fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_SelfSim_total_LoadInc5[:, 0]/1e3, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_OneStepSelfSim_total_LoadInc5[:, 0]/1e3, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Stress $(KPa)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_Sigma_xx_LoadInc5.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_SelfSim_total_LoadInc5[:, 1]/1e3, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_OneStepSelfSim_total_LoadInc5[:, 1]/1e3, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Stress $(KPa)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_Sigma_xy_LoadInc5.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_SelfSim_total_LoadInc5[:, 3]/1e3, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_OneStepSelfSim_total_LoadInc5[:, 3]/1e3, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Stress $(KPa)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_Sigma_yy_LoadInc5.png", bbox_inches='tight')
plt.close(fig)



fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_SelfSim_total_LoadInc5[:, 0]/1e-6, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc5[:, 0]/1e-6, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Strain $(\mu\ mm/mm)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_epsilon_xx_LoadInc5.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_SelfSim_total_LoadInc5[:, 1]/1e-6, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc5[:, 1]/1e-6, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Strain $(  \mu\ mm/mm)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_epsilon_xy_LoadInc5.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_SelfSim_total_LoadInc5[:, 3]/1e-6, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc5[:, 3]/1e-6, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Strain $(  \mu\ mm/mm)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_epsilon_yy_LoadInc5.png", bbox_inches='tight')
plt.close(fig)
##################################################



#####################################
fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_SelfSim_total_LoadInc15[:, 0]/1e3, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_OneStepSelfSim_total_LoadInc15[:, 0]/1e3, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Stress $(KPa)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_Sigma_xx_LoadInc15.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_SelfSim_total_LoadInc15[:, 1]/1e3, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_OneStepSelfSim_total_LoadInc15[:, 1]/1e3, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Stress $(KPa)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_Sigma_xy_LoadInc15.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_SelfSim_total_LoadInc15[:, 3]/1e3, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_sigma_OneStepSelfSim_total_LoadInc15[:, 3]/1e3, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Stress $(KPa)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_Sigma_yy_LoadInc15.png", bbox_inches='tight')
plt.close(fig)



fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_SelfSim_total_LoadInc15[:, 0]/1e-6, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc15[:, 0]/1e-6, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Strain $(\mu\ mm/mm)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_epsilon_xx_LoadInc15.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_SelfSim_total_LoadInc15[:, 1]/1e-6, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc15[:, 1]/1e-6, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Strain $(  \mu\ mm/mm)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_epsilon_xy_LoadInc15.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(15, 10))
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_SelfSim_total_LoadInc15[:, 3]/1e-6, color = "red", label = "SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.plot(range(1, TOTAL_LOAD_PASSES+1), mean_L2_error_epsilon_OneStepSelfSim_total_LoadInc15[:, 3]/1e-6, color = "green", label = "One-Step SelfSim", alpha=0.99, marker = "x", linewidth=7)
plt.xlabel(r'Load pass number', fontsize = 30)
plt.ylabel(r'Mean L2 error norm, Strain $(  \mu\ mm/mm)$', fontsize = 30) 
plt.legend()
# plt.show()
fig.savefig("./Comparison/Plots/MSE_epsilon_yy_LoadInc15.png", bbox_inches='tight')
plt.close(fig)
##################################################














# for i in LOAD_INCREMENTS_LIST:

#     sigma_Expected = deserialise("./ExpectedResults/" + str(i) + "/",  "sigma")
#     sigma_Expected_List.append(sigma_Expected)   
    
#     epsilon_Expected = deserialise("./ExpectedResults/" + str(i) + "/",  "epsilon")
#     epsilon_Expected_List.append(epsilon_Expected)   
    


# for i in range(1, TOTAL_LOAD_PASSES + 1):
#     sigma_SelfSim_LoadInc1 = deserialise("./SelfSim/loadpass" + str(i) + "_loadInc1/1a/",  "sigma")
#     sigma_SelfSim_LoadInc1_List.append(sigma_SelfSim_LoadInc1)

#     epsilon_SelfSim_LoadInc1 = deserialise("./SelfSim/loadpass" + str(i) + "_loadInc1/1a/",  "epsilon")
#     epsilon_SelfSim_LoadInc1_List.append(epsilon_SelfSim_LoadInc1)


# for i in range(1, TOTAL_LOAD_PASSES + 1):
#     sigma_SelfSim_LoadInc5 = deserialise("./SelfSim/loadpass" + str(i) + "_loadInc5/5a/",  "sigma")
#     sigma_SelfSim_LoadInc5_List.append(sigma_SelfSim_LoadInc5)

#     epsilon_SelfSim_LoadInc5 = deserialise("./SelfSim/loadpass" + str(i) + "_loadInc5/5a/",  "epsilon")
#     epsilon_SelfSim_LoadInc5_List.append(epsilon_SelfSim_LoadInc5)

# for i in range(1, TOTAL_LOAD_PASSES + 1):
#     sigma_SelfSim_LoadInc15 = deserialise("./SelfSim/loadpass" + str(i) + "_loadInc15/15a/",  "sigma")
#     sigma_SelfSim_LoadInc15_List.append(sigma_SelfSim_LoadInc15)

#     epsilon_SelfSim_LoadInc15 = deserialise("./SelfSim/loadpass" + str(i) + "_loadInc15/15a/",  "epsilon")
#     epsilon_SelfSim_LoadInc15_List.append(epsilon_SelfSim_LoadInc15)




# # Calculate the errors
# error_epsilon_SelfSim_LoadInc1_List = []
# error_sigma_SelfSim_LoadInc1_List = []

# error_epsilon_SelfSim_LoadInc5_List = []
# error_sigma_SelfSim_LoadInc5_List = []

# error_epsilon_SelfSim_LoadInc15_List = []
# error_sigma_SelfSim_LoadInc15_List = []


# error_epsilon_OneStepSelfSim_LoadInc1_List = []
# error_sigma_OneStepSelfSim_LoadInc1_List = []

# for i in range(TOTAL_LOAD_PASSES):
#     #SelfSim
#     error_epsilon_SelfSim_LoadInc1 = epsilon_SelfSim_LoadInc1_List[i] - epsilon_Expected_List[0]
#     error_epsilon_SelfSim_LoadInc1_List.append(error_epsilon_SelfSim_LoadInc1)
    
#     error_sigma_SelfSim_LoadInc1 = sigma_SelfSim_LoadInc1_List[i] - sigma_Expected_List[0]
#     error_sigma_SelfSim_LoadInc1_List.append(error_sigma_SelfSim_LoadInc1)
    
    
#     error_epsilon_SelfSim_LoadInc5 = epsilon_SelfSim_LoadInc5_List[i] - epsilon_Expected_List[1]
#     error_epsilon_SelfSim_LoadInc5_List.append(error_epsilon_SelfSim_LoadInc5)
    
#     error_sigma_SelfSim_LoadInc5 = sigma_SelfSim_LoadInc5_List[i] - sigma_Expected_List[1]
#     error_sigma_SelfSim_LoadInc5_List.append(error_sigma_SelfSim_LoadInc5)
    
    
#     error_epsilon_SelfSim_LoadInc15 = epsilon_SelfSim_LoadInc15_List[i] - epsilon_Expected_List[2]
#     error_epsilon_SelfSim_LoadInc15_List.append(error_epsilon_SelfSim_LoadInc15)
    
#     error_sigma_SelfSim_LoadInc15 = sigma_SelfSim_LoadInc5_List[i] - sigma_Expected_List[2]
#     error_sigma_SelfSim_LoadInc15_List.append(error_sigma_SelfSim_LoadInc15)


# fig = plt.figure(figsize=(15, 10))
# plt.plot(sigma_Expected_List[i][:,j], sigma_Expected_List[0][:,j], 
#           color = "blue", label = "Ideal", alpha=0.99)
# plt.xlabel(r'Expected stress $(Pa)$')
# plt.ylabel(r'Calculated stress $(Pa)$')
# fig.savefig(master_folder_NN + "Plots/" + "Sigma_" + component[j] 
#             + "_Iter" + str(i+1) + ".png", bbox_inches='tight')
# plt.close(fig)


#     #One Step SelfSim

# # for i in range(TOTAL_LOAD_PASSES):








# # for i in range(1, TOTAL_NUMBER_PASSES+1, DELTA_PASSES):
# #     sigma_Expected = deserialise(master_folder_NN + "loadpass" + str(i) + 
# #                                   "_loadInc" + str(LOAD_INC_INTEREST) + "/" +  
# #                                   str(LOAD_INC_INTEREST) + "a/" , 
# #                                   "sigmaExpected")
# #     sigma_Expected_List.append(sigma_Expected)
    
# #     epsilon_Expected = deserialise(master_folder_NN + "loadpass" + str(i) + 
# #                                     "_loadInc" + str(LOAD_INC_INTEREST) + "/" + 
# #                                     str(LOAD_INC_INTEREST) + "a/" , 
# #                                     "epsilonExpected")
# #     epsilon_Expected_List.append(epsilon_Expected)
    
# #     D_Expected = deserialise(master_folder_NN + "loadpass" + str(i) + 
# #                               "_loadInc" + str(LOAD_INC_INTEREST) + "/" +  
# #                               str(LOAD_INC_INTEREST) + "a/" , "DExpected")
# #     D_Expected_List.append(D_Expected)
    
# #     sigma_A = deserialise(master_folder_NN + "loadpass" + str(i) + "_loadInc" +
# #                         str(LOAD_INC_INTEREST) + "/" + str(LOAD_INC_INTEREST) +
# #                         "a/" , "sigma")
# #     sigma_A_List.append(sigma_A)
    
# #     sigma_B = deserialise(master_folder_NN + "loadpass" + str(i) + "_loadInc" +
# #                         str(LOAD_INC_INTEREST) + "/" + str(LOAD_INC_INTEREST) +
# #                         "B/" , "sigma")  
# #     sigma_B_List.append(sigma_B)
    
# #     epsilon_A = deserialise(master_folder_NN + "loadpass" + str(i) + "_loadInc" 
# #                             + str(LOAD_INC_INTEREST) + "/" + 
# #                             str(LOAD_INC_INTEREST) + "a/" , "epsilon")
# #     epsilon_A_List.append(epsilon_A)
    
    
# #     epsilon_B = deserialise(master_folder_NN + "loadpass" + str(i) + "_loadInc" 
# #                             + str(LOAD_INC_INTEREST) + "/" + 
# #                             str(LOAD_INC_INTEREST) + "B/" , "epsilon")   
# #     epsilon_B_List.append(epsilon_B)
    
# #     D_A = deserialise(master_folder_NN + "loadpass" + str(i) + "_loadInc" +
# #                         str(LOAD_INC_INTEREST) + "/" + str(LOAD_INC_INTEREST) +
# #                         "a/" , "D")
# #     D_A_List.append(D_A)

# #     D_B = deserialise(master_folder_NN + "loadpass" + str(i) + "_loadInc" +
# #                         str(LOAD_INC_INTEREST) + "/" + str(LOAD_INC_INTEREST) +
# #                         "B/" , "D")  
# #     D_B_List.append(D_B)
    
# # epsilon_simul_A_OriginalModel = deserialise(master_folder_NN, 
# #                                     "epsilon_simul_A_OriginalModel_LoadIncNum0")

# # sigma_simul_B_OriginalModel = deserialise(master_folder_NN, 
# #                                       "sigma_simul_B_OriginalModel_LoadIncNum0")


# # component = ["xx", "xy", "xz", "yy", "yz", "zz"]
# # for j in range(6):
# #     # fig = plt.figure(figsize=(15, 10))
# #     # plt.scatter(epsilon_Expected_List[0][:,j], 
# #     #             epsilon_simul_A_OriginalModel[:,j], color = "red", 
# #     #             label = "Calculated", marker='x')
# #     # plt.plot(epsilon_Expected_List[0][:,j], 
# #     #                 epsilon_Expected_List[0][:,j], 
# #     #                 color = "blue", label = "Ideal")
# #     # plt.xlabel(r'Expected strain $(m/m)$')
# #     # plt.ylabel(r'Calculated strain $(m/m)$')
# #     # # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
# #     # #              + ". Base model. Epsilon_" + component[j], fontsize=20)
# #     # # plt.legend()
# #     # fig.savefig(master_folder_NN + "Plots/" + "Epsilon_" + component[j] 
# #     #             + "_BaseModel.png", bbox_inches='tight')
# #     # plt.close(fig)
# #     for i in range(TOTAL_NUMBER_PASSES):
# #         fig = plt.figure(figsize=(15, 10))
# #         plt.scatter(epsilon_Expected_List[i][:,j], epsilon_A_List[i][:,j], 
# #                     color = "green", label = "Calculated", marker='o')
# #         plt.scatter(epsilon_Expected_List[0][:,j], 
# #                     epsilon_simul_A_OriginalModel[:,j], color = "red", label = 
# #                     "Original", marker='+')
# #         plt.plot(epsilon_Expected_List[i][:,j], epsilon_Expected_List[0][:,j], 
# #                   color = "blue", label = "Ideal", alpha=0.99)
# #         plt.xlabel(r'Expected strain $(m/m)$')
# #         plt.ylabel(r'Calculated strain $(m/m)$')
# #         # plt.ylabel("Calculated strain ( m/m)")
        
# #         # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
# #         #              + ". Pass number " + str(i+1) + ". Epsilon_" 
# #         #              + component[j], fontsize=20)
# #         # plt.legend()
# #         fig.savefig(master_folder_NN + "Plots/" + "Epsilon_" + component[j] + 
# #                     "_Iter" + str(i+1) + ".png", bbox_inches='tight')
# #         plt.close(fig)

# # for j in range(6):
# #     # fig = plt.figure(figsize=(15, 10))
# #     # plt.scatter(sigma_Expected_List[0][:,j], 
# #     #                   sigma_simul_B_OriginalModel[:,j], color = "red", 
# #     #                   label = "Calculated", marker='x')
# #     # plt.plot(sigma_Expected_List[0][:,j], sigma_Expected_List[0][:,j], 
# #     #           color = "blue", label = "Ideal")
# #     # plt.xlabel(r'Expected stress $(Pa)$')
# #     # plt.ylabel(r'Calculated stress $(Pa)$')
# #     # # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
# #     # #               + ". Base model. Sigma_" + component[j], fontsize=20)
# #     # # plt.legend()
# #     # fig.savefig(master_folder_NN + "Plots/" + "Sigma_" + component[j] 
# #     #             + "_BaseModel.png", bbox_inches='tight')    
# #     # plt.close(fig)
# #     for i in range(TOTAL_NUMBER_PASSES):
# #         fig = plt.figure(figsize=(15, 10))
# #         plt.scatter(sigma_Expected_List[i][:,j], sigma_B_List[i][:,j], 
# #                     color = "green", label = "Calculated", marker='o')
# #         plt.scatter(sigma_Expected_List[0][:,j], 
# #                     sigma_simul_B_OriginalModel[:,j], color = "red", label = 
# #                     "Original", marker='+')
# #         plt.plot(sigma_Expected_List[i][:,j], sigma_Expected_List[0][:,j], 
# #                   color = "blue", label = "Ideal", alpha=0.99)
# #         plt.xlabel(r'Expected stress $(Pa)$')
# #         plt.ylabel(r'Calculated stress $(Pa)$')
# #         # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
# #         #               + ". Pass number " + str(i+1) + ". Sigma_" 
# #         #               + component[j], fontsize=20)
# #         # plt.legend()
# #         fig.savefig(master_folder_NN + "Plots/" + "Sigma_" + component[j] 
# #                     + "_Iter" + str(i+1) + ".png", bbox_inches='tight')
# #         plt.close(fig)


# # D_simul_A_OriginalModel_LoadIncNum0 = deserialise(master_folder_NN, 
# #                                           "D_simul_A_OriginalModel_LoadIncNum0")
# # D_simul_B_OriginalModel_LoadIncNum0 = deserialise(master_folder_NN, 
# #                                           "D_simul_B_OriginalModel_LoadIncNum0")

# # component = ["x", "y", "z"]
# # for j in range(3):
# #     # fig = plt.figure(figsize=(15, 10))
# #     # plt.scatter(D_Expected_List[0][:,j]/1e-6, 
# #     #                   D_simul_A_OriginalModel_LoadIncNum0[:,j]/1e-6, 
# #     #                     color = "red", label = "Calculated", marker='x')
# #     # plt.plot(D_Expected_List[0][:,j]/1e-6, 
# #     #                 D_Expected_List[0][:,j]/1e-6, 
# #     #                 color = "blue", label = "Ideal")
# #     # plt.xlabel(r'Expected displacement $(\mu m)$')
# #     # plt.ylabel(r'Calculated displacement $(\mu m)$')
# #     # # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
# #     # #               + ". Base model. D_" + component[j], fontsize=20)
# #     # # plt.legend()
# #     # fig.savefig(master_folder_NN + "Plots/" + "D_" + component[j] 
# #     #             + "_BaseModel.png", bbox_inches='tight')    
# #     # plt.close(fig)
# #     for i in range(TOTAL_NUMBER_PASSES):
# #         fig = plt.figure(figsize=(15, 10))
# #         plt.scatter(D_Expected_List[i][:,j]/1e-6, D_A_List[i][:,j]/1e-6, 
# #                     color = "green", label = "Calculated", marker='o')
# #         plt.scatter(D_Expected_List[0][:,j]/1e-6, 
# #                     D_simul_A_OriginalModel_LoadIncNum0[:,j]/1e-6, 
# #                     color = "red", label = "Original", marker='+')
# #         plt.plot(D_Expected_List[i][:,j]/1e-6, D_Expected_List[0][:,j]/1e-6, 
# #                   color = "blue", label = "Ideal", alpha = 0.3)
# #         plt.xlabel(r'Expected displacement $(\mu m)$')
# #         plt.ylabel(r'Calculated displacement $(\mu m)$')
# #         # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
# #         #               + ". Pass number " + str(i+1) + ". D_" + component[j], 
# #         #               fontsize=20)
# #         # plt.legend()
# #         fig.savefig(master_folder_NN + "Plots/" + "D_" + component[j] + "_Iter"
# #                     + str(i+1) + ".png", bbox_inches='tight')
# #         plt.close(fig)
