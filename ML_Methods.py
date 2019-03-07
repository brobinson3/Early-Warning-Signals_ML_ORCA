import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import statsmodels.api as sm
import math
import pandas as pd


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


### Classification ###
def Classification(data,solution,threshold,method,result,test_set):
	data = data.values
	solution = solution.values 

	if method == 'ner_nbr': # K Nearest Neighbors
		from sklearn.neighbors import KNeighborsClassifier
		clsfr = KNeighborsClassifier()
	elif method == 'svm_lin': # Linear SVM
		from sklearn.svm import SVC
		clsfr = SVC(kernel='linear') # has parameter C=0.025
	elif method == 'svm_2nd': # RBF SVM
		from sklearn.svm import SVC
		clsfr = SVC() # gamma=2
	elif method == 'gsn_prc': # Gaussian Process
		from sklearn.gaussian_process import GaussianProcessClassifier
		from sklearn.gaussian_process.kernels import RBF
		clsfr = GaussianProcessClassifier(1.0 * RBF(1.0)) # what is this??
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
	else:
		print('Error: Regression method not recognized.\nPlease pick a valid method key (example: xxx_xxx).')

	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	data = scaler.fit_transform(data)

	if test_set == False:
	# 	if method == 'ply_reg':
	# 		data = poly_features.fit_transform(data)

		solution = solution.reshape(-1,)
		clsfr.fit(data,solution)
		predictions = clsfr.predict(data)
		reg_mse = mean_squared_error(solution,predictions)
		R2 = get_r2_python(solution,predictions)
		# if method == 'rdm_for':
		# 	features = reg.feature_importances_
	
	elif test_set == True:
		from sklearn.model_selection import train_test_split
		train_data,test_data,train_solution,test_solution = train_test_split(data,solution,test_size=0.3,random_state=0)
		# if method == 'ply_reg':
		# 	train_data = poly_features.fit_transform(train_data)
		# 	new_axis = len(train_solution)
		# 	train_data = train_data.reshape(new_axis,-1)
		# 	test_data = poly_features.transform(test_data)

		train_solution = train_solution.reshape(-1,)
		test_solution = test_solution.reshape(-1,)
		clsfr.fit(train_data,train_solution)
		predictions_t = clsfr.predict(test_data)
		reg_mse = mean_squared_error(test_solution,predictions_t)
		R2 = get_r2_python(test_solution,predictions_t)
	
	else:
		print('Error: test_set not True or False.')

	reg_rmse = np.sqrt(reg_mse)

	if result == 'R2':
		return R2
	else:
		print('Error in result option choice. Please chose one of result options.')



### Regression ###
def Regression(data,solution,method,result,test_set):
	data = data.values
	solution = solution.values 

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
	else:
		print('Error: Regression method not recognized.\nPlease pick a valid method key (example: xxx_xxx).')
	
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	data = scaler.fit_transform(data)

	if test_set == False:
		if method == 'ply_reg':
			data = poly_features.fit_transform(data)

		solution = solution.reshape(-1,)
		reg.fit(data,solution)
		predictions = reg.predict(data)
		reg_mse = mean_squared_error(solution,predictions)
		R2 = get_r2_python(solution,predictions)
		if method == 'rdm_for':
			features = reg.feature_importances_
	
	elif test_set == True:
		from sklearn.model_selection import train_test_split
		train_data,test_data,train_solution,test_solution = train_test_split(data,solution,test_size=0.3,random_state=0)
		if method == 'ply_reg':
			train_data = poly_features.fit_transform(train_data)
			new_axis = len(train_solution)
			train_data = train_data.reshape(new_axis,-1)
			test_data = poly_features.transform(test_data)

		train_solution = train_solution.reshape(-1,)
		test_solution = test_solution.reshape(-1,)
		reg.fit(train_data,train_solution)
		predictions_t = reg.predict(test_data)
		reg_mse = mean_squared_error(test_solution,predictions_t)
		R2 = get_r2_python(test_solution,predictions_t)
	
	else:
		print('Error: test_set not True or False.')

	reg_rmse = np.sqrt(reg_mse)
	
	if method == 'lso_reg':
		print('Non-zero features:',len(reg.coef_[reg.coef_!=0]))

	if result == 'rmse':
		return reg_rmse
	elif result == 'R2':
		return R2
	elif result == 'features':
		return features
	elif result == 'plot':
		import matplotlib.pyplot as plt
		if test_set == True:
			if method == 'ply_reg':
				data = poly_features.fit_transform(data)
			reg.fit(data,solution)
			predictions = reg.predict(data)
			plt.scatter(solution,predictions,alpha=0.3)
			plt.scatter(test_solution,predictions_t,color='red',alpha=0.3)
			# print(len(solution),len(test_solution))
		else:
			plt.scatter(solution,predictions,alpha=0.3)
		plt.title(method)
		plt.xlabel('Actual Reliability')
		plt.ylabel('Predicted Reliability')
		plt.show()
	else:
		print('Error in result option choice. Please chose one of result options.')



def RemovingFeatures(data,solution,num_features):
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.preprocessing import StandardScaler
	reg = RandomForestRegressor(n_estimators=100,random_state=3)
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

	return data_return




### Correlation Matrices ### still need to finish this
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

	# if type=='full'
	# if type=='Rel_only'

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

	# corr_matrix_L0.to_csv('corr_matrix_L0.csv')
	# corr_matrix_L1.to_csv('corr_matrix_L1.csv')
	# corr_matrix_L5.to_csv('corr_matrix_L5.csv')
	# corr_matrix_L10.to_csv('corr_matrix_L10.csv')
	# corr_matrix_L20.to_csv('corr_matrix_L20.csv')

	# C_matrix_L0.to_csv('ALL_corr_matrix_L0.csv')
	# C_matrix_L1.to_csv('ALL_corr_matrix_L1.csv')
	# C_matrix_L5.to_csv('ALL_corr_matrix_L5.csv')
	# C_matrix_L10.to_csv('ALL_corr_matrix_L10.csv')
	# C_matrix_L20.to_csv('ALL_corr_matrix_L20.csv')



