### This script has all of the machine learning methods and other tools that calculate results ###

import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import statsmodels.api as sm
import math
import pandas as pd
import os
import matplotlib.pyplot as plt

Location_to_Save_Matrices = 'Data_Matrices_19-06-03' # change this if necessary

### Multiple r**2 and error metric calculations ###
def get_r2_python(x, y):
    n = len(x)
    x_bar = sum(x)/n
    y_bar = sum(y)/n
    x_std = math.sqrt(sum([(xi-x_bar)**2 for xi in x])/(n-1))
    y_std = math.sqrt(sum([(yi-y_bar)**2 for yi in y])/(n-1))
    zx = [(xi-x_bar)/x_std for xi in x]
    zy = [(yi-y_bar)/y_std for yi in y]
    r = sum(zxi*zyi for zxi, zyi in zip(zx, zy))/(n-1)
    return r**2

def get_adjusted_r2(k,y,r2):
	n = len(y) # number of points in data sample
	adj_r2 = 1 - ((1-r2)*(n-1))/(n-k-1)
	return adj_r2

def get_mae_mape_mpe(actual,predicted):
	n = len(actual)
	mae_sum = 0
	mape_sum = 0
	mpe_sum = 0
	for x in range(n):
		mae_sum += abs(actual[x] - predicted[x])
		mape_sum += (abs(actual[x] - predicted[x]) / actual[x])
		mpe_sum += ((actual[x] - predicted[x]) / actual[x])
	mpe = mpe_sum / n
	mape = mape_sum / n
	mae = mae_sum / n
	return mae,mape,mpe


def classifier_measure(actual,predicted):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for i in range(len(actual)):
		if actual[i]==predicted[i]==True:
			TP += 1
		if actual[i]==predicted[i]==False:
			TN += 1
		if predicted[i]==True and actual[i]!=predicted[i]:
			FP += 1
		if predicted[i]==False and actual[i]!=predicted[i]:
			FN += 1
	return(TP,FP,TN,FN)

def classifier_measure_timeseries(actual,predicted):
	TP = []
	FP = []
	TN = []
	FN = []
	
	for i in range(len(actual)):
		if actual[i]==predicted[i]==True:
			TP.append(1)
			TN.append(0)
			FP.append(0)
			FN.append(0)
		if actual[i]==predicted[i]==False:
			TP.append(0)
			TN.append(1)
			FP.append(0)
			FN.append(0)
		if predicted[i]==True and actual[i]!=predicted[i]:
			TP.append(0)
			TN.append(0)
			FP.append(1)
			FN.append(0)
		if predicted[i]==False and actual[i]!=predicted[i]:
			TP.append(0)
			TN.append(0)
			FP.append(0)
			FN.append(1)
	return(TP,FP,TN,FN)


### Classification ###
def Classification(data,solution,test_data,test_solution,threshold,method):
	## Fix data structure ##
	data = data.values
	solution = solution.values
	test_data = test_data.values
	test_solution = test_solution.values

	## List of Method Options with Initialization ##
	if method == 'nst_nbr': # K Nearest Neighbors
		from sklearn.neighbors import KNeighborsClassifier
		clsfr = KNeighborsClassifier()
	elif method == 'log_reg': # Logistic Regression
		from sklearn.linear_model import LogisticRegression
		clsfr = LogisticRegression()
	elif method == 'svm_lin': # Linear SVM
		from sklearn.svm import SVC ##### maybe switch to LinearSVC??
		clsfr = SVC(kernel='linear') # has parameter C=0.025
	elif method == 'svm_2nd': # RBF SVM
		from sklearn.svm import SVC
		clsfr = SVC(kernel='poly',degree=2) # gamma=2
	elif method == 'svm_3rd': # RBF SVM
		from sklearn.svm import SVC
		clsfr = SVC(kernel='poly',degree=3) # gamma=2
	elif method == 'svm_rbf': # RBF SVM
		from sklearn.svm import SVC
		clsfr = SVC() # gamma=2
	elif method == 'gsn_prc': # Gaussian Process
		from sklearn.gaussian_process import GaussianProcessClassifier
		from sklearn.gaussian_process.kernels import RBF
		clsfr = GaussianProcessClassifier() # what is this??
	elif method == 'dcn_tre': # Decision Tree
		from sklearn.tree import DecisionTreeClassifier
		clsfr = DecisionTreeClassifier()
	elif method == 'rdm_for': # Random Forest
		from sklearn.ensemble import RandomForestClassifier
		clsfr = RandomForestClassifier(n_estimators=100)
	elif method == 'mlp_cls': # Neural Net
		from sklearn.neural_network import MLPClassifier
		clsfr = MLPClassifier() # alpha=1
	elif method == 'ada_bst': # Ada Boost
		from sklearn.ensemble import AdaBoostClassifier
		clsfr = AdaBoostClassifier()
	elif method == 'gsn_nbc': # Naive Bayes
		from sklearn.naive_bayes import GaussianNB
		clsfr = GaussianNB()
	elif method == 'qdr_dsc': # QDA
		from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
		clsfr = QuadraticDiscriminantAnalysis()
	elif method == 'grd_bst': # Gradient Boosting Classifier
		from sklearn.ensemble import GradientBoostingClassifier
		clsfr = GradientBoostingClassifier(random_state=3)
	elif method == 'rad_nbr_uni': # Radius Neighbors Classifier
		from sklearn.neighbors import RadiusNeighborsClassifier
		clsfr = RadiusNeighborsClassifier(weights='uniform')
	elif method == 'rad_nbr_dst': # Radius Neighbors Classifier
		from sklearn.neighbors import RadiusNeighborsClassifier
		clsfr = RadiusNeighborsClassifier(weights='distance')
	else:
		print('Error: Classification method not recognized. Please pick a valid method key (example: xxx_xxx).')

	## Preprocessing and Setup ##
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	data = scaler.fit_transform(data)
	scaler = StandardScaler()
	test_data = scaler.fit_transform(test_data)
	solution_binary = solution>=threshold
	test_solution_binary = test_solution>=threshold
	from sklearn.metrics import precision_recall_curve
	from sklearn.metrics import roc_curve
	solution_binary = solution_binary.reshape(-1,)
	test_solution_binary = test_solution_binary.reshape(-1,)

	## Fit the Data ##
	try: # this is necessary because some of the methods won't be able to fit if given weird parameters/data
		clsfr.fit(data,solution_binary)
	except:
		print('Classifier Error:',method,threshold)

	## Predict and Save to Matrix ##
	try:
		test_predictions = clsfr.predict(test_data)
		Matrix_to_save = pd.DataFrame()
		Matrix_to_save['Solution'] = test_solution_binary
		Matrix_to_save['Predictions'] = test_predictions
	except:
		print('Method was unable to predict:',method,threshold)
		Matrix_to_save = pd.DataFrame() # so that main script won't have error and stop code

	return Matrix_to_save



### Regression ### (Note: this section has not been verified to work with new 'Main_Running.py')
def Regression(train_data,train_solution,test_data,test_solution,method):
	## Fix Data Structure ##
	train_data = train_data.values
	train_solution = train_solution.values
	test_data = test_data.values
	test_solution = test_solution.values

	## List of Method Options with Initialization ##
	if method == 'lin_reg': # linear regression
		from sklearn.linear_model import LinearRegression
		reg = LinearRegression()
	elif method == 'ply_reg': # polynomial regression
		from sklearn.linear_model import LinearRegression
		reg = LinearRegression() 
		poly_features = PolynomialFeatures(degree=2)
	elif method == 'rdg_reg': # ridge regression
		from sklearn.linear_model import Ridge
		reg = Ridge()
	elif method == 'lso_reg': # lasso regression
		from sklearn.linear_model import Lasso
		reg = Lasso(alpha=0.00001)
	elif method == 'ela_net': # elastic net regression
		from sklearn.linear_model import ElasticNet
		reg = ElasticNet()
	elif method == 'svr_lin': # SVM regression
		from sklearn.svm import LinearSVR
		reg = LinearSVR(epsilon=0.01,max_iter=10000)
	elif method == 'svr_2nd': # SVR regression
		from sklearn.svm import SVR
		reg = SVR(kernel='poly',degree=2,epsilon=0.01) #C=100
	elif method == 'svr_3rd': # SVR regression
		from sklearn.svm import SVR
		reg = SVR(kernel='poly',degree=3,epsilon=0.01) #C=100
	elif method == 'dcn_tre': # decision tree
		from sklearn.tree import DecisionTreeRegressor
		reg = DecisionTreeRegressor()
	elif method == 'rdm_for': # random forests
		from sklearn.ensemble import RandomForestRegressor
		reg = RandomForestRegressor(n_estimators=100,random_state=3)
	elif method == 'ada_bst': # AdaBoost Regressor
		from sklearn.ensemble import AdaBoostRegressor
		reg = AdaBoostRegressor(n_estimators=100,random_state=3)
	elif method == 'grd_bst': # Gradient Boosting Regressor
		from sklearn.ensemble import GradientBoostingRegressor
		reg = GradientBoostingRegressor(random_state=3)
	elif method == 'gss_prc': # Gaussian Process Regressor
		from sklearn.gaussian_process import GaussianProcessRegressor
		reg = GaussianProcessRegressor(random_state=3)
	elif method == 'knl_rdg': # Kernel Ridge Regression
		from sklearn.kernel_ridge import KernelRidge 
		reg = KernelRidge()
	elif method == 'nst_nbr_uni': # K Nearest Neighbors Regressor
		from sklearn.neighbors import KNeighborsRegressor
		reg = KNeighborsRegressor(weights='uniform')
	elif method == 'nst_nbr_dst': # K Nearest Neighbors Regressor
		from sklearn.neighbors import KNeighborsRegressor
		reg = KNeighborsRegressor(weights='distance')	
	elif method == 'rad_nbr_uni': # Radius Neighbor Regressor
		from sklearn.neighbors import RadiusNeighborsRegressor
		reg = RadiusNeighborsRegressor(weights='uniform')
	elif method == 'rad_nbr_dst': # Radius Neighbor Regressor
		from sklearn.neighbors import RadiusNeighborsRegressor
		reg = RadiusNeighborsRegressor(weights='distance')
	elif method == 'mlp_reg':
		from sklearn.neural_network import MLPRegressor
		reg = MLPRegressor(random_state=3)
	else:
		print('Error: Regression method not recognized.\nPlease pick a valid method key (example: xxx_xxx).')
	
	## Preprocessing and Setup ##
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	data = scaler.fit_transform(train_data)
	scaler = StandardScaler()
	test_data = scaler.fit_transform(test_data)
	solution = train_solution.reshape(-1,)
	if method == 'ply_reg':
		data = poly_features.fit_transform(data)
	reg.fit(data,solution)

	if len(test_data) < 5:
		predictions = reg.predict(data)
	
	elif len(test_data) > 5:
		if method == 'ply_reg':
			test_data = poly_features.transform(test_data)
		test_solution = test_solution.reshape(-1,)
		predictions_test = reg.predict(test_data)
		solution = test_solution
		predictions = predictions_test
	
	else:
		print('Error: test_set undetermined.')
	
	Matrix_to_save = pd.DataFrame()
	Matrix_to_save['Solution'] = solution
	Matrix_to_save['Predictions'] = predictions

	return Matrix_to_save



### Remove Certain Number of Features from Data ###
def RemovingFeatures(data,solution,lead,window,num_features,sol_type,sc,test):

	try: # see if feature importances have already been calculated for this combination of parameters
		features = pd.read_csv('Removed_Features_Lists_Second_Run/Feature_Importances__'+sol_type+'__'+lead+'_'+window+'.csv', index_col=0)
		important_features_list = features.iloc[:num_features].index
		data_return = data[important_features_list]
	
	except FileNotFoundError: # if feature importances haven't already been calculated
		print('----Previously created reduced features lists not found. Please wait while one is generated.----')
		from sklearn.ensemble import RandomForestRegressor
		from sklearn.preprocessing import StandardScaler
		reg = RandomForestRegressor(n_estimators=1000,random_state=3) # how many trees are included can be changed to improve accuracy (more trees) or run time (fewer trees)
		features_list = data.columns
		data2 = data.values
		solution2 = solution.values

		scaler = StandardScaler()
		data2 = scaler.fit_transform(data2)
		solution2 = solution2.reshape(-1,)
		
		reg.fit(data2,solution2)
		features = reg.feature_importances_
		features_df = pd.DataFrame(features,index=features_list)
		features_df_ordered = features_df.sort_values([0],ascending=False)
		important_features_list = features_df_ordered.iloc[:num_features].index
		data_return = data[important_features_list]
		
		os.makedirs('Removed_Features_Lists_Second_Run', exist_ok=True)
		features_df_ordered.to_csv('Removed_Features_Lists_Second_Run/Feature_Importances__'+sol_type+'__'+lead+'_'+window+'.csv')
		data_return.to_csv('Removed_Features_Lists_Second_Run/Removed_Features__'+sol_type+'__'+lead+'_'+window+'.csv')

		if num_features == 500:
			if test == False:
				data_return.to_csv('%s/%s-Data_%s-REDUCED.csv'%(Location_to_Save_Matrices,sc,lead), index_col=0)
			elif test == True:
				data_return.to_csv('%s/%s-Testing_Data_%s-REDUCED.csv'%(Location_to_Save_Matrices,sc,lead), index_col=0)

	return data_return



### Correlation Matrices ### 
def Correlations(Lead,type):
	from Data_Formatting import Load_Data
	(Data_L0,Data_L1,Data_L5,Data_L10,Data_L20,
		Solution_L0_R10,Solution_L1_R10,Solution_L5_R10,Solution_L10_R10,Solution_L20_R10,
		Solution_L0_R20,Solution_L1_R20,Solution_L5_R20,Solution_L10_R20,Solution_L20_R20,
		Solution_L0_R30,Solution_L1_R30,Solution_L5_R30,Solution_L10_R30,Solution_L20_R30,
		Solution_L0_R40,Solution_L1_R40,Solution_L5_R40,Solution_L10_R40,Solution_L20_R40,
		Solution_L0_R50,Solution_L1_R50,Solution_L5_R50,Solution_L10_R50,Solution_L20_R50) = Load_Data()

	if Lead == 0:
		Lead_s = '_0_'
	if Lead == 1:
		Lead_s = '_1_'
	if Lead == 5:
		Lead_s = '_5_'

	Solution_L0_R50.columns = ['Reliability_50']
	Solution_L1_R50.columns = ['Reliability_50']
	Solution_L5_R50.columns = ['Reliability_50']
	Solution_L10_R50.columns = ['Reliability_50']
	Solution_L20_R50.columns = ['Reliability_50']

	matrix_L0 = Data_L0
	matrix_L0['Reliability_50'] = pd.Series(Solution_L0_R50['Reliability_50'], index=matrix_L0.index)
	matrix_L1 = Data_L1
	matrix_L1['Reliability_50'] = pd.Series(Solution_L1_R50['Reliability_50'], index=matrix_L1.index)
	matrix_L5 = Data_L5
	matrix_L5['Reliability_50'] = pd.Series(Solution_L5_R50['Reliability_50'], index=matrix_L5.index)
	matrix_L10 = Data_L10
	matrix_L10['Reliability_50'] = pd.Series(Solution_L10_R50['Reliability_50'], index=matrix_L10.index)
	matrix_L20 = Data_L20
	matrix_L20['Reliability_50'] = pd.Series(Solution_L20_R50['Reliability_50'], index=matrix_L20.index)

	C_matrix_L0 = matrix_L0.corr()
	C_matrix_L1 = matrix_L1.corr()
	C_matrix_L5 = matrix_L5.corr()
	C_matrix_L10 = matrix_L10.corr()
	C_matrix_L20 = matrix_L20.corr()

	corr_matrix_L0 = C_matrix_L0['Reliability_50'].sort_values(ascending=False)
	corr_matrix_L1 = C_matrix_L1['Reliability_50'].sort_values(ascending=False)
	corr_matrix_L5 = C_matrix_L5['Reliability_50'].sort_values(ascending=False)
	corr_matrix_L10 = C_matrix_L10['Reliability_50'].sort_values(ascending=False)
	corr_matrix_L20 = C_matrix_L20['Reliability_50'].sort_values(ascending=False)

	corr_matrix_L0.to_csv('corr_matrix_L0.csv')
	corr_matrix_L1.to_csv('corr_matrix_L1.csv')
	corr_matrix_L5.to_csv('corr_matrix_L5.csv')
	corr_matrix_L10.to_csv('corr_matrix_L10.csv')
	corr_matrix_L20.to_csv('corr_matrix_L20.csv')

	C_matrix_L0.to_csv('ALL_corr_matrix_L0.csv')
	C_matrix_L1.to_csv('ALL_corr_matrix_L1.csv')
	C_matrix_L5.to_csv('ALL_corr_matrix_L5.csv')
	C_matrix_L10.to_csv('ALL_corr_matrix_L10.csv')
	C_matrix_L20.to_csv('ALL_corr_matrix_L20.csv')



