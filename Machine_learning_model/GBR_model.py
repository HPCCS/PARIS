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

# #############################################################################
# reading in X, Y

y=np.loadtxt(open("small_program_except_rate_label.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("small_program_label.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("combined_features_counter_label.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("y_labels_cut.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#x=np.loadtxt(open("x_features_cut2.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("small_program_features_unif.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
x=np.loadtxt(open("small_program_features.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("combined_features_counter.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("small_program_features.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#pdb.set_trace()
#x=x[:,:10]
#x=x[:,[16,11,21,1,7,27,3,6,17,26,13,19,23,2,12,22,8,29,5,28,15,18,9,25,0,10,20,4,14,24]]
#print(x)

# feature selection
#lda=LinearDiscriminantAnalysis(n_components=15)
#x_lda=lda.fit(x,y).transform(x)
#pca=PCA(n_components=15)
#x=pca.fit(x).transform(x)

y_hpc=np.loadtxt(open("hpc_except_rate_label.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y_hpc=np.loadtxt(open("small_program_label_hpc.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#x_hpc=np.loadtxt(open("x_hpc_features_cut.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x_hpc=np.loadtxt(open("small_program_features_hpc.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
x_hpc=np.loadtxt(open("hpc_program_features.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#pca1=PCA(n_components=20)
#x_hpc=pca1.fit(x_hpc).transform(x_hpc)
#x_hpc=x_hpc[:,:10]
#x_hpc=x_hpc[:,[16,11,21,1,7,27,3,6,17,26,13,19,23,2,12,22,8,29,5,28,15,18,9,25,0,10,20,4,14,24]]
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
	#np.clip(predictions,0,1,out=predictions)
#        plt.scatter(y, predictions)
#        plt.title("GradientBoostingRegressor")
#        plt.xlabel("True Values")
#        plt.ylabel("Predictions")
#        plt.show()
	predictions_h=gb.predict(x_hpc)
	#np.clip(predictions_h,0,1,out=predictions_h)
	err=np.mean(abs((predictions-y)/y))
        var=np.var(abs((predictions-y)/y))
	print "Cross-Predicted Relative Error: ", err
        print "Cross-Predicted Var of Relative Error: ", var
	print(predictions)
	np.savetxt('result1.txt',predictions,delimiter='\n',fmt='%.3f')
	err=np.mean(abs((predictions-y)))
        var=np.var(abs((predictions-y)))
	print "Cross-Predicted Abs Error: ", err
        print "Cross-Predicted Var of Abs Error: ", var
	err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
        var_h=np.var(abs((predictions_h-y_hpc)/y_hpc))
	print "Cross-Predicted Relative Error for hpc: ", err_h
	print "Cross-Predicted Var of Relative Error for hpc: ", var_h
	err_h=np.mean(abs((predictions_h-y_hpc)))
        var_h=np.var(abs((predictions_h-y_hpc)))
	print "Cross-Predicted Abs Error for hpc: ", err_h
	print "Cross-Predicted Var of Abs Error for hpc: ", var_h
	print(predictions_h)
	np.savetxt('result2.txt',predictions_h,delimiter='\n',fmt='%.3f')

print "Gradient Boosting  Regression is Off ..."
print "\n\n"
