import pdb
import statistics as stat
import csv
import numpy as np
from numpy import linalg as LA
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn import metrics
from sklearn.svm import SVR,NuSVR,LinearSVR
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# #############################################################################
# reading in X, Y

#y=np.loadtxt(open("small_program_success_rate_label_90.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("small_program_sdc_rate_label_90.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
y=np.loadtxt(open("small_program_except_rate_label_90.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("combined_features_counter_label.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#y=np.loadtxt(open("y_labels_cut.csv", "rb"), delimiter=",", skiprows=0, dtype="float")
#x=np.loadtxt(open("x_features_cut2.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("small_program_features_90_unif.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("small_program_features_90.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#x=np.loadtxt(open("combined_features_counter.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
x=np.loadtxt(open("small_program_features_90.csv", "rb"), delimiter=",", skiprows=0,dtype="float")
#

print("------ Mutual Info ------")
mi=mutual_info_regression(x,y)
print(mi)

print("------ Variance ------")
for i in range(0,30):
    print(stat.variance(x[:,i]))

print("------ F-score & P-value ------")
fr=f_regression(x,y)
print(fr)

'''
print(x[0])
print("---------------------")
pca=PCA(n_components=30)
pca.fit(x)

print("---------------------")
print(pca.get_covariance())
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
'''
