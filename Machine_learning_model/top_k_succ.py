import pdb
import csv
import numpy as np
from numpy import linalg as LA
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model 
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import cross_validate
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn import metrics
from sklearn.svm import SVR,NuSVR,LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# files: 
# small_program_features.csv small_program_success_rate_label.csv small_program_sdc_rate_label.csv small_program_except_rate_label.csv
# hpc_program_features.csv hpc_success_rate_label.csv hpc_sdc_rate_label.csv hpc_except_rate_label.csv
# #############################################################################
# reading in X, Y

y=np.loadtxt(open("small_program_success_rate_label_90.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("small_program_sdc_rate_label_90.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("small_program_except_rate_label_90.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("combined_features_counter_label.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("y_labels_cut.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#x=np.loadtxt(open("x_features_cut2.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("small_program_features_90_unif.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("small_program_features_90.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("combined_features_counter.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
x=np.loadtxt(open("small_program_features_90.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#pdb.set_trace()
#x=x[:,:10]
# feature sort for success rate
# x=x[:,[3,16,7,23,11,21,13,27,17,1,26,6,2,22,19,15,5,25,12,0,20,10,14,29,4,24,9,28,8,18]]
x=x[:,[3,16,7,23,11]]

# x=x[:,[3,23,7,27,16,11,13,21,17,26,1,2,22,6,19,15,5,20,12,25,10,0,29,14,4,9,24,28,8,18]]
# x=x[:,[3,23,7,27,16,11,13,21,17,26,1,2,22,6,19,15,5,20,12,25,10,0,29,14,4,9,24,28,8,18]]
# feature sort for SDC rate
#x=x[:,[23,27,3,26,7,21,6,1,13,16,17,28,11,5,9,29,2,19,25,22,12,15,24,0,4,10,14,20,8,18]]
#x=x[:,[23,27]]
# feature sort for Interruption rate
#x=x[:,[13,17,3,7,26,23,27,6,15,29,5,16,25,0,2,11,9,10,12,1,20,18,19,22,4,14,21,24,8,28]]
#x=x[:,[13,17]]
#print(x)

# feature selection
#lda=LinearDiscriminantAnalysis(n_components=15)
#x_lda=lda.fit(x,y).transform(x)
#pca=PCA(n_components=15)
#x=pca.fit(x).transform(x)

y_hpc=np.loadtxt(open("hpc_success_rate_label.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y_hpc=np.loadtxt(open("hpc_sdc_rate_label.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y_hpc=np.loadtxt(open("hpc_except_rate_label.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#x_hpc=np.loadtxt(open("x_hpc_features_cut.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
x_hpc=np.loadtxt(open("hpc_program_features.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x_hpc=np.loadtxt(open("small_program_features_hpc_unif.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#pca1=PCA(n_components=20)
#x_hpc=pca1.fit(x_hpc).transform(x_hpc)
#x_hpc=x_hpc[:,:10]
# feature sort for success rate
# x_hpc=x_hpc[:,[3,16,7,23,11,21,13,27,17,1,26,6,2,22,19,15,5,25,12,0,20,10,14,29,4,24,9,28,8,18]]
x_hpc=x_hpc[:,[3,16,7,23,11]]

# x_hpc=x_hpc[:,[3,23,7,27,16,11,13,21,17,26,1,2,22,6,19,15,5,20,12,25,10,0,29,14,4,9,24,28,8,18]]
# x_hpc=x_hpc[:,[3,23,7,27,16,11,13,21,17,26,1,2,22,6,19,15,5,20,12,25,10,0,29,14,4,9,24,28,8,18]]
# feature sort for SDC rate
#x_hpc=x_hpc[:,[23,27,3,26,7,21,6,1,13,16,17,28,11,5,9,29,2,19,25,22,12,15,24,0,4,10,14,20,8,18]]
#x_hpc=x_hpc[:,[23,27]]
# feature sort for Interruption rate
#x_hpc=x_hpc[:,[13,17,3,7,26,23,27,6,15,29,5,16,25,0,2,11,9,10,12,1,20,18,19,22,4,14,21,24,8,28]]
#x_hpc=x_hpc[:,[13,17]]
#print(x_hpc)

# whitening
z_scaler = MinMaxScaler()
x=z_scaler.fit_transform(x)
x_hpc=z_scaler.fit_transform(x_hpc)

# #############################################################################
# Gradient Boosting Regressor *1*
#plt.figure(10)

print "Gradient Boosting Regression is On ..."


#lrate = np.arange(0.019, 0.031, 0.001)
#al = np.arange(0.1,0.9,0.01)
#n_est = np.arange(90, 125, 1)
for n_estimators in [0]:
    for alpha in [0]:
        print "**************************************"
#        print "lrate: ", learning_rate
#        print "alpha: ", alpha
#        print "n_est: ", n_estimators
	#gb=GradientBoostingRegressor(max_depth=1,learning_rate=0.04,n_estimators=100)
	#gb=GradientBoostingRegressor(loss='lad',max_depth=1,learning_rate=0.05,n_estimators=440)
        #gb=GradientBoostingRegressor(loss='huber',max_depth=1,learning_rate=0.45,n_estimators=200,alpha=0.45)
        #gb=GradientBoostingRegressor(loss='quantile',max_depth=1,learning_rate=0.028,n_estimators=109,alpha=0.36,criterion="friedman_mse")
        #gb=GradientBoostingRegressor(loss='quantile',max_depth=1,learning_rate=0.028,n_estimators=109,alpha=0.36,criterion="friedman_mse",subsample=0.6)
        gb=GradientBoostingRegressor()
        gb.fit(x,y)
	gbsc=gb.score(x,y)
	print "R2 score is: ", gbsc
	gbsc_hpc=gb.score(x_hpc,y_hpc)
	print "R2 score on hpc is: ", gbsc_hpc
	predictions = cross_val_predict(gb,x,y,cv=10)
	np.clip(predictions,0,1,out=predictions)
	predictions_h=gb.predict(x_hpc)
	np.clip(predictions_h,0,1,out=predictions_h)
	err=np.mean(abs((predictions-y)/y))
        var=np.var(abs((predictions-y)/y))
	print "Cross-Predicted Relative Error: ", err
        print "Cross-Predicted Var of Relative Error: ", var
	np.savetxt('result1.txt',predictions,delimiter='\n',fmt='%.3f')
	err=np.mean(abs((predictions-y)))
        var=np.var(abs((predictions-y)))
	print "Cross-Predicted Abs Error: ", err
        print "Cross-Predicted Var of Abs Error: ", var
	err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
        var_h=np.var(abs((predictions_h-y_hpc)/y_hpc))
	print predictions_h
	print y_hpc
	print "Cross-Predicted Relative Error for hpc: ", err_h
	print "Cross-Predicted Var of Relative Error for hpc: ", var_h
#        plt.scatter(y_hpc, predictions_h)
#        plt.title("GradientBoostingRegressor")
#        plt.xlabel("True Values")
#        plt.ylabel("Predictions")
#        plt.show()
	err_h=np.mean(abs((predictions_h-y_hpc)))
        var_h=np.var(abs((predictions_h-y_hpc)))
	print "Cross-Predicted Abs Error for hpc: ", err_h
	print "Cross-Predicted Var of Abs Error for hpc: ", var_h
	np.savetxt('result2.txt',predictions_h,delimiter='\n',fmt='%.3f')

print "Gradient Boosting  Regression is Off ..."
print "\n\n"

