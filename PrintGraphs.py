import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------------------------- FED Tuning ----------------------------------------------------------------------------------------#

#---------- IID----------#

#---------- Import DataBase ----------#
tuning1 = pd.read_csv("mldl23fl-main/Results/Federated_Non-IID_False_LocalEpochs_1_Lr_0.01_momentum_0.0_wd_0.0_batchSize_32.csv")
tuning2 = pd.read_csv("mldl23fl-main/Results/Federated_Non-IID_False_LocalEpochs_1_Lr_0.001_momentum_0.0_wd_0.0_batchSize_32.csv")

tuning3 = pd.read_csv("mldl23fl-main/Results/Federated_Non-IID_False_LocalEpochs_1_Lr_0.01_momentum_0.0_wd_0.0_batchSize_64.csv")
tuning4 = pd.read_csv("mldl23fl-main/Results/Federated_Non-IID_False_LocalEpochs_1_Lr_0.001_momentum_0.0_wd_0.0_batchSize_64.csv")


#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(tuning3["Epochs"],tuning3["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'tuning3')
plt.plot(tuning4["Epochs"],tuning4["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'tuning4')
plt.ylim(bottom = 0)
plt.title("Comparison between methods with IID in FedAdp settings")
plt.xlabel("Rounds")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.savefig("im1.png", dpi=300)
plt.show()