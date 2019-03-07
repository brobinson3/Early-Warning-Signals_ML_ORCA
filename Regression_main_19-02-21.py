import numpy as np 
import pandas as pd
from scipy import stats
from Data_Formatting import *
from ML_Methods import *
import cProfile


### Options ##############################################################
Data_Retrieval = False # if true will generate data matrices to feed into the scikitlearn library and save them (this takes about 1.5 mins), if false will load in already-made matrices that have been saved previously
Correlations = False # if true will generate correlation matrices
Corr_type = 'full' # choose one of the correlation option types below

ML = 'Regression' # choose 'Regression' or 'Classification'
# threshold = ? # Classification threshold; only used if ML='Classification'
method = 'svr_lin' # choose one of the methods options below
result_metric = 'R2' # choose one of the result options below
test_set = True # if true will return result based on 30% of data left out of training (the algorithm has not seen this data), if false returns result based on training data predictions (the algorithm has already seen this data)
num_features = 10 # max is 480 (all)
solution_type = 'change' # choose one of the solution type options below
##########################################################################

### Correlation Options ###
# Corr_type = 'full' for full correlation matrix
# Corr_type = 'Rel_only' for only correlations to Reliability

### Result Options ###
# result_metric = 'rmse' to get root mean squared error returned
# result_metric = 'R2' to get R^2 value returned
# result_metric = 'features' to get feature importances (method must be set to 'rdm_for', and test_set=False)
# result_metric = 'plot' to plot actual vs predicted values; if test_set = True will show test results in red on top of blue training results

### Method Options ###
# For method, chose one of the following:
# 'lin_reg' for linear regression
# 'ply_reg' for polynomial regression
# 'svr_lin' for SVR regression, optional parameter epsilon
# 'svr_2nd' for SVR regression, optional parameters epsilon, C, degree
# 'svr_3rd' for SVR regression, optional parameters epsilon, C, degree
# 'dcn_tre' for decision tree regression, optional parameter max_depth
# 'rdm_for' for random forest regression

### Solution Type Options ###
# solution_type = 'value' to predict Reliability value
# solution_type = 'change' to predict change in Reliability over lead time
# solution_type = 'sd' to predict standard deviation (change?) of Reliability ## NOT DONE YET


#####################################################################################################

if Data_Retrieval:
	Data_Compilation()

(columns,Data_L0,Data_L1,Data_L5,Data_L10,Data_L20,
	Solution_L0_R01,Solution_L1_R01,Solution_L5_R01,Solution_L10_R01,Solution_L20_R01,
	Solution_L0_R10,Solution_L1_R10,Solution_L5_R10,Solution_L10_R10,Solution_L20_R10,
	Solution_L0_R20,Solution_L1_R20,Solution_L5_R20,Solution_L10_R20,Solution_L20_R20,
	Solution_L0_R30,Solution_L1_R30,Solution_L5_R30,Solution_L10_R30,Solution_L20_R30,
	Solution_L0_R40,Solution_L1_R40,Solution_L5_R40,Solution_L10_R40,Solution_L20_R40,
	Solution_L0_R50,Solution_L1_R50,Solution_L5_R50,Solution_L10_R50,Solution_L20_R50) = Load_Data()

(Sol_change_L1_R01,Sol_change_L5_R01,Sol_change_L10_R01,Sol_change_L20_R01,
	Sol_change_L1_R10,Sol_change_L5_R10,Sol_change_L10_R10,Sol_change_L20_R10,
	Sol_change_L1_R20,Sol_change_L5_R20,Sol_change_L10_R20,Sol_change_L20_R20,
	Sol_change_L1_R30,Sol_change_L5_R30,Sol_change_L10_R30,Sol_change_L20_R30,
	Sol_change_L1_R40,Sol_change_L5_R40,Sol_change_L10_R40,Sol_change_L20_R40,
	Sol_change_L1_R50,Sol_change_L5_R50,Sol_change_L10_R50,Sol_change_L20_R50) = Load_Change_Sol()


### Correlation Matrices ###
if Correlations:
	Correlations(Lead=0,type=Corr_type)


### Data ###
Data_Matrices = [Data_L0,Data_L1,Data_L5,Data_L10,Data_L20]
Data_Matrices_names = ['Data_L0','Data_L1','Data_L5','Data_L10','Data_L20']

Solution_Matrices = [Solution_L0_R01,Solution_L1_R01,Solution_L5_R01,Solution_L10_R01,Solution_L20_R01,
	Solution_L0_R10,Solution_L1_R10,Solution_L5_R10,Solution_L10_R10,Solution_L20_R10,
	Solution_L0_R20,Solution_L1_R20,Solution_L5_R20,Solution_L10_R20,Solution_L20_R20,
	Solution_L0_R30,Solution_L1_R30,Solution_L5_R30,Solution_L10_R30,Solution_L20_R30,
	Solution_L0_R40,Solution_L1_R40,Solution_L5_R40,Solution_L10_R40,Solution_L20_R40,
	Solution_L0_R50,Solution_L1_R50,Solution_L5_R50,Solution_L10_R50,Solution_L20_R50]
Solution_Matrices_names = ['Solution_L0_R01','Solution_L1_R01','Solution_L5_R01','Solution_L10_R01','Solution_L20_R01',
	'Solution_L0_R10','Solution_L1_R10','Solution_L5_R10','Solution_L10_R10','Solution_L20_R10',
	'Solution_L0_R20','Solution_L1_R20','Solution_L5_R20','Solution_L10_R20','Solution_L20_R20',
	'Solution_L0_R30','Solution_L1_R30','Solution_L5_R30','Solution_L10_R30','Solution_L20_R30',
	'Solution_L0_R40','Solution_L1_R40','Solution_L5_R40','Solution_L10_R40','Solution_L20_R40',
	'Solution_L0_R50','Solution_L1_R50','Solution_L5_R50','Solution_L10_R50','Solution_L20_R50']

Change_Matrices = [Sol_change_L1_R01,Sol_change_L5_R01,Sol_change_L10_R01,Sol_change_L20_R01,
	Sol_change_L1_R10,Sol_change_L5_R10,Sol_change_L10_R10,Sol_change_L20_R10,
	Sol_change_L1_R20,Sol_change_L5_R20,Sol_change_L10_R20,Sol_change_L20_R20,
	Sol_change_L1_R30,Sol_change_L5_R30,Sol_change_L10_R30,Sol_change_L20_R30,
	Sol_change_L1_R40,Sol_change_L5_R40,Sol_change_L10_R40,Sol_change_L20_R40,
	Sol_change_L1_R50,Sol_change_L5_R50,Sol_change_L10_R50,Sol_change_L20_R50]
Change_Matrices_names = ['Sol_change_L1_R01','Sol_change_L5_R01','Sol_change_L10_R01','Sol_change_L20_R01',
	'Sol_change_L1_R10','Sol_change_L5_R10','Sol_change_L10_R10','Sol_change_L20_R10',
	'Sol_change_L1_R20','Sol_change_L5_R20','Sol_change_L10_R20','Sol_change_L20_R20',
	'Sol_change_L1_R30','Sol_change_L5_R30','Sol_change_L10_R30','Sol_change_L20_R30',
	'Sol_change_L1_R40','Sol_change_L5_R40','Sol_change_L10_R40','Sol_change_L20_R40',
	'Sol_change_L1_R50','Sol_change_L5_R50','Sol_change_L10_R50','Sol_change_L20_R50']

Leads = ['L0_','L1_','L5_','L10_','L20_']
Windows = ['R01','R10','R20','R30','R40','R50']



### Make loop for all methods ###
methods = ['lin_reg','ply_reg','svr_2nd','svr_3rd','dcn_tre','rdm_for']
test_sets = [True,False]
num_features_all = [10,5,3,2,1]
for method in methods:
	for test_set in test_sets:
		for num_features in num_features_all:

			if solution_type == 'change':
				Matrix = Change_Matrices
				Matrix_names = Change_Matrices_names

			elif solution_type == 'value':
				Matrix = Solution_Matrices
				Matrix_names = Solution_Matrices_names

			elif solution_type == 'sd': # need to work on this soon (getting SD matrices)
				Matrix = x
				Matrix_names = x

			else:
				print('Wrong solution_type input. Please input one of the available options.')


			results_array = pd.DataFrame(index=Windows,columns=Leads)

			### Main Regression ###
			for data_name,Lead in zip(Data_Matrices_names,Leads):
				data_index = Data_Matrices_names.index(data_name)
				for S in Matrix_names:
						if Lead in S:
							for W in Windows:
								if W in S:
									sol_index = Matrix_names.index(S)
									solution = Matrix[sol_index]
									print('--------------------')
									print(Data_Matrices_names[data_index])
									print(Matrix_names[sol_index])
									if num_features != 480:
										# cProfile.run('RemovingFeatures(data=Data_Matrices[data_index],solution=solution,num_features=num_features)')
										data = RemovingFeatures(data=Data_Matrices[data_index],solution=solution,num_features=num_features)
									else:
										data = Data_Matrices[data_index]
									if ML == 'Regression':
										result = Regression(data,solution,method=method,test_set=test_set,result=result_metric)
									elif ML == 'Classification':
										result = Classification(data,solution,method=method,threshold=threshold,test_set=test_set,result=result_metric)
									else:
										print('ML must be either Regression or Classification')
									print(result)
									if result_metric == 'features':
										np.savetxt('Feature_Importances_'+Lead+W+'.csv',result)
									else:
										results_array.loc[W,Lead] = result
			results_array.to_csv('Results_Outputs/Results-'+result_metric+'_Method-'+method+'_SolType-'+solution_type+'_Features-'+str(num_features)+'_TestSet-'+str(test_set)+'.csv')


### To print out RMSE or R2 for all permutations ###
# for data_name,Lead in zip(Data_Matrices_names,Leads):
# 	data_index = Data_Matrices_names.index(data_name)
# 	for S in Solution_Matrices_names:
# 		if Lead in S:
# 			sol_index = Solution_Matrices_names.index(S)
# 			print('--------------------')
# 			print(Data_Matrices_names[data_index])
# 			print(Solution_Matrices_names[sol_index])
# 			data = Data_Matrices[data_index]
# 			solution = Solution_Matrices[sol_index]
# 			result = Regression(data,solution,method='ply_reg',test_set=False,result='rmse')
# 			print(result)




### To print out only one permutation ###
# result = Regression(Data_L0,Solution_L0_R50,method='lso_reg',columns=columns,test_set=False,result='R2')
# print(result)




# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LinearRegression
# Solution_L0_R50 = Solution_L0_R50.reshape(-1,)
# result = SelectFromModel(LinearRegression())
# result.fit(Data_L0,Solution_L0_R50)
# result_support = result.get_support()
# # print(result_support)
# # print(Data_L0)
# Data_L0 = pd.DataFrame(Data_L0,columns=columns)
# # print(Data_L0)
# result_feature = Data_L0.loc[:,result_support].columns.tolist()
# print(result_feature)
# print(str(len(result_feature)), 'selected features')


### Check .corr() function to see what it actually does ###

# dta = pd.read_csv('C:/Users/BRadmin/Google Drive/Reservoir Modeling/ORCA/orca/data/input_climate_files/bcc-csm1-1_rcp85_r1i1p1_input_data.csv', index_col = 0, parse_dates = True)	
# col1 = dta.iloc[:,1]
# col3 = col1.rolling(10).mean()
# col5 = col1.rolling(50).mean()
# col2 = dta.iloc[:,2]
# col4 = col2.rolling(10).mean()
# col6 = col2.rolling(50).mean()

# cols = pd.concat([col1,col2,col3,col4,col5,col6],axis=1)
# cols.columns = ['FOL_yearly','MRC_yearly','FOL_10-yr','MRC_10-yr','FOL_50-yr','MRC_50-yr']
# print(cols)

# Corr_matrix = cols.corr()
# print(Corr_matrix)
# Corr_matrix.to_csv('Corr_matrix_test.csv')


