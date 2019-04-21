import pdb
import csv
import numpy as np
from numpy import linalg as LA
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn.kernel_ridge import KernelRidge
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

y=np.loadtxt(open("small_program_label.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("y_labels_cut.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#x=np.loadtxt(open("x_features_cut2.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("small_program_features_90_unif.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("small_program_features_90.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
x=np.loadtxt(open("small_program_features.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#pdb.set_trace()

# feature selection
#lda=LinearDiscriminantAnalysis(n_components=15)
#x_lda=lda.fit(x,y).transform(x)
#pca=PCA(n_components=15)
#x=pca.fit(x).transform(x)

y_hpc=np.loadtxt(open("small_program_label_hpc.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#x_hpc=np.loadtxt(open("x_hpc_features_cut.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
x_hpc=np.loadtxt(open("small_program_features_hpc.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
##x_hpc=np.loadtxt(open("small_program_features_hpc_unif.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#pca1=PCA(n_components=20)
#x_hpc=pca1.fit(x_hpc).transform(x_hpc)

z_scaler = StandardScaler()
#x=z_scaler.fit_transform(x)
#x_hpc=z_scaler.fit_transform(x_hpc)

# #############################################################################
# Fitting Regressions and an Original Linear Regression for comparison 

# Least Square Linear Regression
#'''
#ridgereg=linear_model.Ridge(alpha=.5)
#ridgereg.fit(x,y)

plt.figure(1)

print "Least Square Linear Regression is On ..."
oreg=linear_model.LinearRegression()
#selector=RFECV(oreg,step=1,cv=5)
#selector=selector.fit(x,y)
#print selector.support_
#print selector.ranking_
#print selector.n_features_
#print selector.grid_scores_
#print selector.estimator
oreg.fit(x,y)
print oreg.coef_
osc=oreg.score(x,y)
osc_hpc=oreg.score(x_hpc,y_hpc)
print "R2 score is: ", osc
print "R2 score on hpc is: ", osc_hpc
o_cv_res=cross_val_score(oreg,x,y,cv=10)
print o_cv_res
predictions = cross_val_predict(oreg,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
print predictions
plt.scatter(y, predictions)
plt.title("Least Square Linear")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=oreg.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "Least Square Linear Regression is Off ..."
print "\n\n"

# Ridge Regression

plt.figure(2)
print "Ridge Regression is On ..."

rr=linear_model.Ridge(alpha=0.5)
rr.fit(x,y)
print rr.coef_
rrsc=rr.score(x,y)
rrsc_hpc=rr.score(x_hpc,y_hpc)
print "R2 score is: ", rrsc
print "R2 score on hpc is: ", rrsc_hpc
rr_cv_res=cross_val_score(rr,x,y,cv=10)
print rr_cv_res
predictions = cross_val_predict(rr,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Ridge Regression")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=rr.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "Ridge Regression is Off ..."
print "\n\n"


# Support Vector Regression *3*

#svr=SVR(kernel='linear')
#selector=RFECV(svr,step=1,cv=5)
#selector=selector.fit(x,y)
#print selector.support_
#print selector.ranking_

#pdb.set_trace()
#plt.figure(0)
print "SV Regression is On ..."

C_range = np.arange(0, 1, 1)
gamma_range = np.logspace(-1, 0, 2)
degree_range = np.arange(1,2,1)
nu_range=np.arange(0.01,1,0.01)
for C in [0]:
    for nu in [3]:
#         for degree in degree_range:
#             print "**************************************"
#             print "C: ", C
#	     print "gamma: ", gamma
             print "nu: ", nu
             #sv=SVR(kernel='rbf', C=120, gamma=1e-11)
             #sv=NuSVR(nu=0.5, kernel='rbf', C=120, gamma=1e-11)
#sv=LinearSVR(C=1000, max_iter=1500,loss='epsilon_insensitive',epsilon=0.1)
#             #sv=SVR(kernel='sigmoid', C=C, gamma=gamma)
	     sv=SVR()
             sv.fit(x,y)
#print sv.coef_
             svsc=sv.score(x,y)
#             svsc_hpc=sv.score(x_hpc,y_hpc)
             print "R2 score is: ", svsc
#             print "R2 score on hpc is: ", svsc_hpc
             sv_cv_res=cross_val_score(sv,x,y,cv=10)
#print sv_cv_res
             predictions = cross_val_predict(sv,x,y,cv=10)
             #np.clip(predictions,0,1,out=predictions)
#plt.scatter(y, predictions)
#plt.title("SV Regression")
#plt.xlabel("True Values")
#plt.ylabel("Predictions")
#plt.show()
             predictions_h=sv.predict(x_hpc)
             #np.clip(predictions_h,0,1,out=predictions_h)
#             accuracy = metrics.r2_score(y, predictions)
#             print "Cross-Predicted Accuracy:", accuracy
             err=np.mean(abs((predictions-y)/y))
             err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
             print "Cross-Predicted Error: ", err
             print "Cross-Predicted Error for hpc: ", err_h

print "SV Regression is Off ..."
print "\n\n"



# Decision Tree Regressor *2*
#plt.figure(11)
print "Decision Tree Regression is On ..."
lrate = np.arange(0.015, 0.03, 0.001)
al = np.arange(0.1,0.9,0.1)
n_est = np.arange(90, 120, 1)
depth=np.arange(1,8,1)
for max_depth in [1]:
    for learning_rate in [1]:
        print "**************************************"
#        print "lrate: ", learning_rate
#        print "alpha: ", alpha
        print "depth: ", max_depth
	#dt=DecisionTreeRegressor(max_depth=2,criterion="friedman_mse",splitter="random",min_samples_split=4)
	dt=DecisionTreeRegressor()
        dt.fit(x,y)
#print gb.coef_
	dtsc=dt.score(x,y)
	print "R2 score is: ", dtsc
	dtsc_hpc=dt.score(x_hpc,y_hpc)
	print "R2 score on hpc is: ", dtsc_hpc
	dt_cv_res=cross_val_score(dt,x,y,cv=10)
	print dt_cv_res
	predictions = cross_val_predict(dt,x,y,cv=10)
	#np.clip(predictions,0,1,out=predictions)
#        plt.scatter(y, predictions)
#        plt.title("GradientBoostingRegressor")
#        plt.xlabel("True Values")
#        plt.ylabel("Predictions")
#        plt.show()
	accuracy = metrics.r2_score(y, predictions)
	print "Cross-Predicted Accuracy:", accuracy
	predictions_h=dt.predict(x_hpc)
	#np.clip(predictions_h,0,1,out=predictions_h)
	err=np.mean(abs((predictions-y)/y))
	print "Cross-Predicted Error: ", err
	err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
	print "Cross-Predicted Error for hpc: ", err_h

print "Gradient Boosting  Regression is Off ..."
print "\n\n"

# Passive Aggressive Regressor
#plt.figure(12)
print "Passive Aggressive Regression is On ..."
lrate = np.arange(0.015, 0.03, 0.001)
al = np.arange(0.1,0.9,0.1)
n_est = np.arange(90, 120, 1)
depth=np.arange(1,8,1)
for max_depth in [1]:
    for learning_rate in [1]:
        print "**************************************"
#        print "lrate: ", learning_rate
#        print "alpha: ", alpha
#        print "depth: ", max_depth
	#pa=linear_model.PassiveAggressiveRegressor(C=0.01,loss="epsilon_insensitive",max_iter=2000)
        pa=linear_model.PassiveAggressiveRegressor()
	pa.fit(x,y)
#print pa.coef_
	pasc=pa.score(x,y)
	print "R2 score is: ", pasc
	pasc_hpc=pa.score(x_hpc,y_hpc)
	print "R2 score on hpc is: ", pasc_hpc
	pa_cv_res=cross_val_score(pa,x,y,cv=10)
	print pa_cv_res
	predictions = cross_val_predict(pa,x,y,cv=10)
	#np.clip(predictions,0,1,out=predictions)
#        plt.scatter(y, predictions)
#        plt.title("GradientBoostingRegressor")
#        plt.xlabel("True Values")
#        plt.ylabel("Predictions")
#        plt.show()
	accuracy = metrics.r2_score(y, predictions)
	print "Cross-Predicted Accuracy:", accuracy
	predictions_h=pa.predict(x_hpc)
	#np.clip(predictions_h,0,1,out=predictions_h)
	err=np.mean(abs((predictions-y)/y))
	print "Cross-Predicted Error: ", err
	err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
	print "Cross-Predicted Error for hpc: ", err_h

print "Passive Aggressive Regression is Off ..."
print "\n\n"


#''' and None
# Gradient Boosting Regressor *1*
#plt.figure(10)
print "Gradient Boosting Regression is On ..."
lrate = np.arange(0.019, 0.031, 0.001)
al = np.arange(0.1,0.9,0.01)
n_est = np.arange(90, 125, 1)
for n_estimators in [0]:
    for alpha in [0]:
        print "**************************************"
#        print "lrate: ", learning_rate
        print "alpha: ", alpha
        print "n_est: ", n_estimators
	#gb=GradientBoostingRegressor(max_depth=1,learning_rate=0.04,n_estimators=100)
	#gb=GradientBoostingRegressor(loss='lad',max_depth=1,learning_rate=0.05,n_estimators=440)
	# -x- gb=GradientBoostingRegressor(loss='huber',max_depth=1,learning_rate=0.45,n_estimators=200,alpha=alpha)
        #gb=GradientBoostingRegressor(loss='quantile',max_depth=1,learning_rate=0.028,n_estimators=109,alpha=0.36,criterion="friedman_mse")
        #gb=GradientBoostingRegressor(loss='quantile',max_depth=1,learning_rate=0.028,n_estimators=109,alpha=0.36,criterion="friedman_mse",subsample=0.6)
        gb=GradientBoostingRegressor()
	gb.fit(x,y)
	gbsc=gb.score(x,y)
	print "R2 score is: ", gbsc
	gbsc_hpc=gb.score(x_hpc,y_hpc)
	print "R2 score on hpc is: ", gbsc_hpc
	gb_cv_res=cross_val_score(gb,x,y,cv=10)
	print gb_cv_res
	predictions = cross_val_predict(gb,x,y,cv=10)
	#np.clip(predictions,0,1,out=predictions)
#        plt.scatter(y, predictions)
#        plt.title("GradientBoostingRegressor")
#        plt.xlabel("True Values")
#        plt.ylabel("Predictions")
#        plt.show()
	accuracy = metrics.r2_score(y, predictions)
	print "Cross-Predicted Accuracy:", accuracy
	predictions_h=gb.predict(x_hpc)
	#np.clip(predictions_h,0,1,out=predictions_h)
	err=np.mean(abs((predictions-y)/y))
	print "Cross-Predicted Error: ", err
	err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
        print predictions_h
	print "Cross-Predicted Error for hpc: ", err_h

print "Gradient Boosting  Regression is Off ..."
print "\n\n"


# Random Forest Regressor
#plt.figure(13)
print "Random Forest Regression is On ..."
lrate = np.arange(0.019, 0.031, 0.001)
al = np.arange(0.1,0.9,0.01)
n_est = np.arange(10, 100, 2)
for n_estimators in [1]:
    for alpha in [0]:
        print "**************************************"
#        print "lrate: ", learning_rate
#        print "alpha: ", alpha
        print "n_est: ", n_estimators
	#gb=GradientBoostingRegressor(max_depth=1,learning_rate=0.04,n_estimators=100)
	#gb=GradientBoostingRegressor(loss='lad',max_depth=1,learning_rate=0.05,n_estimators=440)
	# -x- gb=GradientBoostingRegressor(loss='huber',max_depth=1,learning_rate=0.45,n_estimators=200,alpha=alpha)
        #gb=GradientBoostingRegressor(loss='quantile',max_depth=1,learning_rate=0.028,n_estimators=109,alpha=0.36,criterion="friedman_mse")
        #rf=RandomForestRegressor(n_estimators=88)
	rf=RandomForestRegressor()
	rf.fit(x,y)
#print rf.coef_
	rfsc=rf.score(x,y)
	print "R2 score is: ", rfsc
	rfsc_hpc=rf.score(x_hpc,y_hpc)
	print "R2 score on hpc is: ", rfsc_hpc
	rf_cv_res=cross_val_score(rf,x,y,cv=10)
	print rf_cv_res
	predictions = cross_val_predict(rf,x,y,cv=10)
	#np.clip(predictions,0,1,out=predictions)
#        plt.scatter(y, predictions)
#        plt.title("GradientBoostingRegressor")
#        plt.xlabel("True Values")
#        plt.ylabel("Predictions")
#        plt.show()
	accuracy = metrics.r2_score(y, predictions)
	print "Cross-Predicted Accuracy:", accuracy
	predictions_h=rf.predict(x_hpc)
	#np.clip(predictions_h,0,1,out=predictions_h)
	err=np.mean(abs((predictions-y)/y))
	print "Cross-Predicted Error: ", err
	err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
	print "Cross-Predicted Error for hpc: ", err_h
        print predictions_h

print "Gradient Boosting  Regression is Off ..."
print "\n\n"



#'''
# Lasso
plt.figure(3)
print "Lasso Regression is On ..."
#la=linear_model.Lasso(alpha=0.1)
la=linear_model.Lasso()
la.fit(x,y)
print la.coef_
lasc=la.score(x,y)
print "R2 score is: ", lasc
lasc_hpc=la.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", lasc_hpc
la_cv_res=cross_val_score(la,x,y,cv=10)
print la_cv_res
predictions = cross_val_predict(la,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Lasso")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=la.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "Lasso Regression is Off ..."
print "\n\n"

# Elastic Net
plt.figure(4)
print "Elastic Net Regression is On ..."
#en=linear_model.ElasticNet(alpha=0.5,l1_ratio=0.5)
en=linear_model.ElasticNet()
en.fit(x,y)
print(en.coef_)
ensc=en.score(x,y)
print "R2 score is: ", ensc
ensc_hpc=en.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", ensc_hpc
en_cv_res=cross_val_score(en,x,y,cv=10)
print en_cv_res
predictions = cross_val_predict(en,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Elastic Net")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=en.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "Elastic Net Regression is Off ..."
print "\n\n"

# Bayesian Ridge Lasso
plt.figure(5)
print "Bayesian Ridge Regression is On ..."
#br=linear_model.BayesianRidge(compute_score=True)
br=linear_model.BayesianRidge()
br.fit(x,y)
print br.coef_
brsc=br.score(x,y)
print "R2 score is: ", brsc
brsc_hpc=br.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", brsc_hpc
br_cv_res=cross_val_score(br,x,y,cv=10)
print br_cv_res
predictions = cross_val_predict(br,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Bayesian Ridge")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=br.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "Bayesian Ridge Regression is Off ..."
print "\n\n"

# RANSAC Regression 
plt.figure(6)
print "RANSAC Regression is On ..."
# rr is replacable
#RR=linear_model.RANSACRegressor(rr, random_state=42)
RR=linear_model.RANSACRegressor()
RR.fit(x,y)
#print RR.coef_
RRsc=RR.score(x,y)
print "R2 score is: ", RRsc
RRsc_hpc=RR.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", RRsc_hpc
RR_cv_res=cross_val_score(RR,x,y,cv=10)
print RR_cv_res
predictions = cross_val_predict(RR,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("RANSAC")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=RR.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "RANSAC Regression is Off ..."
print "\n\n"

# TheilSen Regression
plt.figure(7)
print "TheilSen Regression is On ..."
#tsr=linear_model.TheilSenRegressor(random_state=42)
tsr=linear_model.TheilSenRegressor()
tsr.fit(x,y)
print tsr.coef_
tsrsc=tsr.score(x,y)
print "R2 score is: ", tsrsc
tsrsc_hpc=tsr.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", tsrsc_hpc
tsr_cv_res=cross_val_score(tsr,x,y,cv=10)
print tsr_cv_res
predictions = cross_val_predict(tsr,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("TheilSen")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
predictions_h=tsr.predict(x_hpc)
print "Cross-Predicted Error: ", err
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "TheilSen Regression is Off ..."
print "\n\n"

# Huber Regression
plt.figure(8)
print "Huber Regression is On ..."
hr=linear_model.HuberRegressor()
hr.fit(x,y)
print hr.coef_
hrsc=hr.score(x,y)
print "R2 score is: ", hrsc
hrsc_hpc=hr.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", hrsc_hpc
hr_cv_res=cross_val_score(hr,x,y,cv=10)
print hr_cv_res
predictions = cross_val_predict(hr,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Huber")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=hr.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "TheilSen Regression is Off ..."
print "\n\n"


# SGD regression
print "SGD Regression is On ..."
hr=linear_model.SGDRegressor()
hr.fit(x,y)
print hr.coef_
hrsc=hr.score(x,y)
print "R2 score is: ", hrsc
hrsc_hpc=hr.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", hrsc_hpc
hr_cv_res=cross_val_score(hr,x,y,cv=10)
print hr_cv_res
predictions = cross_val_predict(hr,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Huber")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=hr.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "SGD Regression is Off ..."
print "\n\n"

# Kneighbors Regression
print "Kneighbors Regression is On ..."
hr=KNeighborsRegressor()
hr.fit(x,y)
#print hr.coef_
hrsc=hr.score(x,y)
print "R2 score is: ", hrsc
hrsc_hpc=hr.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", hrsc_hpc
hr_cv_res=cross_val_score(hr,x,y,cv=10)
print hr_cv_res
predictions = cross_val_predict(hr,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Huber")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=hr.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "Kneighbors Regression is Off ..."
print "\n\n"

# MLP Regression
print "MLP Regression is On ..."
hr=MLPRegressor(hidden_layer_sizes=3)
hr.fit(x,y)
#print hr.coef_
hrsc=hr.score(x,y)
print "R2 score is: ", hrsc
hrsc_hpc=hr.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", hrsc_hpc
hr_cv_res=cross_val_score(hr,x,y,cv=10)
print hr_cv_res
predictions = cross_val_predict(hr,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Huber")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=hr.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "MLP Regression is Off ..."
print "\n\n"

# KernelRidge Regression
print "KernelRidge Regression is On ..."
hr=KernelRidge()
hr.fit(x,y)
#print hr.coef_
hrsc=hr.score(x,y)
print "R2 score is: ", hrsc
#hrsc_hpc=hr.score(x_hpc,y_hpc)
#print "R2 score on hpc is: ", hrsc_hpc
#hr_cv_res=cross_val_score(hr,x,y,cv=10)
#print hr_cv_res
predictions = cross_val_predict(hr,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Huber")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=hr.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "KernelRidge Regression is Off ..."
print "\n\n"

# NuSVR regressor
print "NuSVR Regression is On ..."
hr=NuSVR()
hr.fit(x,y)
#print hr.coef_
hrsc=hr.score(x,y)
print "R2 score is: ", hrsc
hrsc_hpc=hr.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", hrsc_hpc
hr_cv_res=cross_val_score(hr,x,y,cv=10)
print hr_cv_res
predictions = cross_val_predict(hr,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Huber")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
err=np.mean(abs((predictions-y)/y))
print "Cross-Predicted Error: ", err
predictions_h=hr.predict(x_hpc)
err_h=np.mean(abs((predictions_h-y_hpc)/y_hpc))
print "Cross-Predicted Error for hpc: ", err_h
print "NuSVR Regression is Off ..."
print "\n\n"



# #############################################################################
# This preprocessor transforms an input data matrix into polynomial form
# such that X can be used into any model
'''
poly=preprocessing.PolynomialFeatures(degree=3)
x=poly.fit_transform(x)

# Least Square Linear Regression

#ridgereg=linear_model.Ridge(alpha=.5)
#ridgereg.fit(x,y)

plt.figure(9)

print "Least Square Linear Regression is On ..."
lr=linear_model.LinearRegression()
oreg=linear_model.LinearRegression()
oreg.fit(x,y)
print oreg.coef_
osc=oreg.score(x,y)
print "R2 score is: ", osc
osc_hpc=oreg.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", osc_hpc
o_cv_res=cross_val_score(lr,x,y,cv=10)
print o_cv_res
predictions = cross_val_predict(lr,x,y,cv=10)
#pdb.set_trace()
#np.clip(predictions,0,1,out=predictions)
print predictions
plt.scatter(y, predictions)
plt.title("Least Square Linear")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
print "Least Square Linear Regression is Off ..."
print "\n\n"

# Ridge Regression

plt.figure(10)
print "Ridge Regression is On ..."

rr=linear_model.Ridge(alpha=0.5)
rr.fit(x,y)
print rr.coef_
rrsc=rr.score(x,y)
print "R2 score is: ", rrsc
rrsc_hpc=rr.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", rrsc_hpc
rr_cv_res=cross_val_score(rr,x,y,cv=10)
print rr_cv_res
predictions = cross_val_predict(rr,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Ridge Regression")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
print "Ridge Regression is Off ..."
print "\n\n"

# Lasso
plt.figure(11)
print "Lasso Regression is On ..."
la=linear_model.Lasso(alpha=0.1)
la.fit(x,y)
print la.coef_
lasc=la.score(x,y)
print "R2 score is: ", lasc
lasc_hpc=la.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", lasc_hpc
la_cv_res=cross_val_score(la,x,y,cv=10)
print la_cv_res
predictions = cross_val_predict(la,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Lasso")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
print "Lasso Regression is Off ..."
print "\n\n"

# Elastic Net
plt.figure(12)
print "Elastic Net Regression is On ..."
en=linear_model.ElasticNet(alpha=0.5,l1_ratio=0.5)
en.fit(x,y)
print(en.coef_)
ensc=en.score(x,y)
print "R2 score is: ", ensc
ensc_hpc=en.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", ensc_hpc
en_cv_res=cross_val_score(en,x,y,cv=10)
print en_cv_res
predictions = cross_val_predict(en,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Elastic Net")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
print "Elastic Net Regression is Off ..."
print "\n\n"

# Bayesian Ridge Lasso
plt.figure(13)
print "Bayesian Ridge Regression is On ..."
br=linear_model.BayesianRidge(compute_score=True)
br.fit(x,y)
print br.coef_
brsc=br.score(x,y)
print "R2 score is: ", brsc
brsc_hpc=br.score(x_hpc,y_hpc)
print "R2 score on hpc is: ", brsc_hpc
br_cv_res=cross_val_score(br,x,y,cv=10)
print br_cv_res
predictions = cross_val_predict(br,x,y,cv=10)
#np.clip(predictions,0,1,out=predictions)
plt.scatter(y, predictions)
plt.title("Bayesian Ridge")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy
print "Bayesian Ridge Regression is Off ..."
print "\n\n"

''' and None

# #############################################################################
# Plot true weights, estimated weights, histogram of the weights, and
# predictions with standard deviations
#lw = 2
#plt.figure(figsize=(6, 5))
#plt.title("Weights of the model")
#plt.plot(oreg.coef_, color='navy', linestyle='--', label="OLR estimate")
#plt.xlabel("Features")
#plt.ylabel("Values of the weights")
#plt.legend(loc="best", prop=dict(size=12))
#plt.show()

#plt.figure(figsize=(6, 5))
#plt.title("Histogram of the weights")
#plt.hist(oreg.coef_, bins=29, color='gold', log=True, edgecolor='black')
#plt.ylabel("Features")
#plt.xlabel("Values of the weights")
#plt.legend(loc="upper left")
#plt.show()

#plt.figure(figsize=(6, 5))
#plt.title("Marginal log-likelihood")
#plt.plot(oreg.scores_, color='navy', linewidth=lw)
#plt.ylabel("Score")
#plt.xlabel("Iterations")
