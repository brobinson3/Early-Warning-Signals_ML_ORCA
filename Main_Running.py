### Script for running classification and regression calculations ###

import numpy as np 
import pandas as pd
from scipy import stats
from Data_Formatting import *
from ML_Methods import *
from Compiling_Results import *
import cProfile
import warnings
import os
warnings.filterwarnings("ignore")

### Options ##############################################################
ML = 'Classification' # choose 'Regression' or 'Classification'
##########################################################################
with open('ORCA_data/scenario_names_all.txt') as f: # change file name here to run different scenarios
	scenarios = f.read().splitlines()
os.makedirs('Results_Raw_Outputs',exist_ok=True)

# scenarios = ['None'] # Use this for training. It runs all of the scenarios together instead of leave-one-out

Leads = ['L00','L01','L05','L10','L20']
num_features_all = [500,10,5,4,3,2,1] # if you want to run different or fewer features, change this line
thresholds = [0.8,0.7,0.6,0.72,0.74,0.76,0.78,0.68,0.66,0.64,0.62,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98] # if you want to run different or fewer thresholds, change this line

if ML == 'Regression':
	methods = ['lin_reg','ply_reg','svr_lin','svr_2nd','svr_3rd','dcn_tre','rdm_for','ada_bst','grd_bst','gss_prc','knl_rdg','nst_nbr_uni','nst_nbr_dst','mlp_reg']
if ML == 'Classification': # more methods are available (see grayed out text) than were used for final version of paper results
	methods = ['nst_nbr','log_reg','svm_3rd','rdm_for','mlp_cls','ada_bst','gsn_nbc']#,'rad_nbr_uni','rad_nbr_dst','svm_lin',,'svm_2nd','svm_rbf','gsn_prc','dcn_tre','qdr_dsc','grd_bst']

i = 0
## Calculate Results Using Leave-One-Out ## (training on all scenarios except the one selected)
for sc in scenarios:
	i += 1
	print('Scenario (#%s): %s'%(i,sc))
	
	try:
		(variables,Data_L00,Data_L01,Data_L05,Data_L10,Data_L20,
			Solution_L00,Solution_L01,Solution_L05,Solution_L10,Solution_L20,
			Testing_Data_L00,Testing_Data_L01,Testing_Data_L05,Testing_Data_L10,Testing_Data_L20,
			Testing_Solution_L00,Testing_Solution_L01,Testing_Solution_L05,Testing_Solution_L10,Testing_Solution_L20) = Load_Data_REDUCED(sc)
	except FileNotFoundError:
		try:
			(variables,Data_L00,Data_L01,Data_L05,Data_L10,Data_L20,
				Solution_L00,Solution_L01,Solution_L05,Solution_L10,Solution_L20,
				Testing_Data_L00,Testing_Data_L01,Testing_Data_L05,Testing_Data_L10,Testing_Data_L20,
				Testing_Solution_L00,Testing_Solution_L01,Testing_Solution_L05,Testing_Solution_L10,Testing_Solution_L20) = Load_Data(sc)
		except FileNotFoundError:
			print('----Data and Solution Matrices not found. Please wait while matrices are generated.----')
			(variables,Data_L00,Data_L01,Data_L05,Data_L10,Data_L20,
				Solution_L00,Solution_L01,Solution_L05,Solution_L10,Solution_L20,
				Testing_Data_L00,Testing_Data_L01,Testing_Data_L05,Testing_Data_L10,Testing_Data_L20,
				Testing_Solution_L00,Testing_Solution_L01,Testing_Solution_L05,Testing_Solution_L10,Testing_Solution_L20) = Data_Compilation(sc,'Annual_Rolling_Reliability_30yr')
	print('----Data Collected. Moving on to %s Calculations.----'%(ML))

	Data_Matrices = [Data_L00,Data_L01,Data_L05,Data_L10,Data_L20]
	Data_Matrices_names = ['Data_L00','Data_L01','Data_L05','Data_L10','Data_L20']
	Solution_Matrices = [Solution_L00,Solution_L01,Solution_L05,Solution_L10,Solution_L20]
	Solution_Matrices_names = ['Solution_L00','Solution_L01','Solution_L05','Solution_L10','Solution_L20']

	Testing_Data_Matrices = [Testing_Data_L00,Testing_Data_L01,Testing_Data_L05,Testing_Data_L10,Testing_Data_L20]
	Testing_Data_Matrices_names = ['Testing_Data_L00','Testing_Data_L01','Testing_Data_L05','Testing_Data_L10','Testing_Data_L20']
	Testing_Solution_Matrices = [Testing_Solution_L00,Testing_Solution_L01,Testing_Solution_L05,Testing_Solution_L10,Testing_Solution_L20]
	Testing_Solution_Matrices_names = ['Testing_Solution_L00','Testing_Solution_L01','Testing_Solution_L05','Testing_Solution_L10','Testing_Solution_L20']

	for num_features in num_features_all: # running through all feature options
		print('----Working on %s features.'%(num_features))
		for threshold in thresholds: # running through all threshold options
			for data_name,Lead in zip(Data_Matrices_names,Leads): # running through all Leads
				data_index = Data_Matrices_names.index(data_name)
				for sol_name in Solution_Matrices_names: # running through all solution matrixes
					if Lead in sol_name: # finding the correct solution matrix
						for method in methods: # running through all methods
							try: # See if this combination has already been run before continuing (allowing you to run this code in pieces)
								if ML == 'Classification':
									trying = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,'value','R30',method,threshold))
									print(Lead, method, threshold, num_features,'Found it!')
								elif ML == 'Regression':
									trying = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,num_features,'value','R30',method))
								else:
									print('Something has gone wrong with calculation type. Please check.')
									break
							except: # The results of this combination have not been found, so we need to run it
								solution = Solution_Matrices[Solution_Matrices_names.index(sol_name)] # get the training solution
								test_solution = Testing_Solution_Matrices[Testing_Solution_Matrices_names.index('Testing_%s'%(sol_name))] # get the testing solution
								test_data = Testing_Data_Matrices[Testing_Data_Matrices_names.index('Testing_%s'%(data_name))] # get the testing data
								data = RemovingFeatures(data=Data_Matrices[data_index],solution=solution,lead=Lead,window='R30',num_features=num_features,sol_type='value',sc=sc,test=False) # remove features if necessary (if not necessary, this won't do anything)
								test_data = RemovingFeatures(data=test_data,solution=test_solution,lead=Lead,window='R30',num_features=num_features,sol_type='value',sc=sc,test=True) # remove features if necessary (if not necessary, this won't do anything)
								
								## Get and Save Results ##
								if ML == 'Classification':
									Matrix_to_save = Classification(data,solution,test_data,test_solution,method=method,threshold=threshold) # CLASSIFICATION!
									Matrix_to_save.to_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,'value','R30',method,threshold))
								elif ML == 'Regression':
									Matrix_to_save = Regression(data,solution,test_data,test_solution,method=method) # REGRESSION!
									Matrix_to_save.to_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,num_features,'value','R30',method))
								else:
									print('Something has gone wrong. No results saved.')
									break # to make sure that your code doesn't run for 5 days and return nothing :)



### Plotting ### Note that additional plots (not used in final paper) are available in Compiling_Results.py
date = '19-19-19' # set a real date for plot file name
save = False # choose to save (True) or not save (False) plots
os.makedirs('All_Figures',exist_ok=True)

MAP(date,save=save) # PAPER FIGURE 1
CLASSIFICATION__Time_vs_TPR_RCP(date,save=save)
CLASSIFICATION__Thresholds_vs_TP_TN_subplots(date,save=save)
CLASSIFICATION__Leads_vs_TP_TN_subplots(date,save=save)
CLASSIFICATION__Features_vs_TPR_subplots(date,save=save)
CLASSIFICATION__Leads_vs_V_NV_switch_subplots(date,save=save)
CLASSIFICATION__Time_vs_TPR_subplots(date,save=save)
CLASSIFICATION__Correlation_heatmap(date,save=save)
CLASSIFICATION__Overview_Fig(date,save=save)
P_Value_Heatmap(date,save=save)