### This code takes all of the raw data, translates it into rolling averages and standard deviations, and combines it all into matrices for every lead time and every scenario for leave-one-out calculations

import numpy as np
import pandas as pd
import os

Reliability = ['Annual_Rolling_Reliability_30yr'] # you can also run this code for 10, 20, 40, and 50 -year rolling reliability
Location_of_Input_Data = 'ORCA_data/ORCA_inputs_19-06-03' # change this if necessary
Location_to_Save_Matrices = 'Data_Matrices_19-06-03' # change this if necessary

os.makedirs(Results_Raw_Outputs,exist_ok=True)
os.makedirs(Location_of_Input_Data,exist_ok=True)

new_cols = ['TLG_fnf','FOL_fnf','MRC_fnf','MIL_fnf','NML_fnf','ORO_fnf','MKM_fnf','BND_fnf','NHG_fnf','SHA_fnf','YRS_fnf',
			'BND_swe','ORO_swe','YRS_swe','FOL_swe',
			'SHA_pr','ORO_pr','FOL_pr',
			'SHA_tas','ORO_tas','FOL_tas','SHA_tasmax','ORO_tasmax','FOL_tasmax','SHA_tasmin','ORO_tasmin','FOL_tasmin',
			'SHA_storage','FOL_storage','ORO_storage',
			'SHA_out','FOL_out','ORO_out',
			'DEL_in','DEL_out','Total_DEL_SODD','Total_SODD_from_RES','Total_pumped_from_DEL','Total_Shortage','DEL_X2']


## To compile RCP data ##
def dat_file_to_df(raw_data):
	rcp26_cols = raw_data[38]
	cols = []
	word = []
	for spot in range(0,len(rcp26_cols)):
		if rcp26_cols[spot] is ' ':
			word = []
		elif rcp26_cols[spot]:
			word.append(rcp26_cols[spot])
			try:
				rcp26_cols[(spot+1)]
				if rcp26_cols[(spot+1)] is ' ':
					cols.append(''.join(word))
					word = []
			except IndexError:
				cols.append(''.join(word))
				word = []

	dates = list(range(1765,2501))
	rcp26_all_data = pd.DataFrame(index=dates,columns=cols)
	rcp26_data = raw_data[39:]

	for row in range(0,len(rcp26_data)):
		word = []
		df_col = 0
		for column in range(0,len(rcp26_data[0])):
			
			if rcp26_data[row][column] is ' ':
				word = []
			else:
				word.append(rcp26_data[row][column])
				try:
					if rcp26_data[row][column+1] is ' ':
						value = ''.join(word)
						rcp26_all_data.iloc[row,df_col] = value
						df_col += 1
						word = []
				except IndexError:
						value = ''.join(word)
						rcp26_all_data.iloc[row,df_col] = value
						word = []
	return rcp26_all_data



## To compile all data ##
def Data_Compilation(scenario_removed,rel='Annual_Rolling_Reliability_30yr',climate_data=False):

	# climate_data = True ## If you want ot add the RCP label as a variable, turn this line on
	if climate_data == True:
		with open('RCP3PD_MIDYR_CONC.dat') as f:
			rcp26_raw_data = f.read().splitlines()
		with open('RCP45_MIDYR_CONC.dat') as f:
			rcp45_raw_data = f.read().splitlines()
		with open('RCP6_MIDYR_CONC.dat') as f:
			rcp60_raw_data = f.read().splitlines()
		with open('RCP85_MIDYR_CONC.dat') as f:
			rcp85_raw_data = f.read().splitlines()

		RCP26_DATA = dat_file_to_df(rcp26_raw_data)
		RCP45_DATA = dat_file_to_df(rcp45_raw_data)
		RCP60_DATA = dat_file_to_df(rcp60_raw_data)
		RCP85_DATA = dat_file_to_df(rcp85_raw_data)

		new_rcp_cols = RCP85_DATA.columns

	with open('ORCA_data/scenario_names_all.txt') as f:
		scenarios = f.read().splitlines()

	if scenario_removed != 'None': # if not running all scenarios together (for training), take the selected scenario out to be tested
		scenarios.remove(scenario_removed)
		print('Removed One Scenario for Testing')

	dta = pd.read_csv('%s/%s-model_inputs_ann_avg_10.csv'%(Location_of_Input_Data,scenarios[1]), index_col = 0, parse_dates = True)
	dta_index = dta.index.year

	msd = ['avg','sd']
	whens = ['ann','m01','m02','m03','m04','m05','m06','m07','m08','m09','m10','m11','m12']
	windows = ['10','20','30','40','50']

	## Populate variables for all combinations above ##
	variables = []
	for v in dta.columns:
		for x in msd:
			for w in windows:
				for when in whens:
					l = []
					l.append(v)
					l.append('_')
					l.append(w)
					l.append('_')
					l.append(x)
					l.append('_')
					l.append(when)
					s = ''.join(l)
					variables.append(s)

	## Initiate Data Matrices for each Lead ##
	Data_L0 = pd.DataFrame()
	Data_L1 = pd.DataFrame()
	Data_L5 = pd.DataFrame()
	Data_L10 = pd.DataFrame()
	Data_L20 = pd.DataFrame()
	
	## Initiate Solution Matrices for each Lead ##
	Solution_L0 = pd.DataFrame(columns=['Reliability'])
	Solution_L1 = pd.DataFrame(columns=['Reliability'])
	Solution_L5 = pd.DataFrame(columns=['Reliability'])
	Solution_L10 = pd.DataFrame(columns=['Reliability'])
	Solution_L20 = pd.DataFrame(columns=['Reliability'])

	Sol_Change_L1 = pd.DataFrame(columns=['Reliability'])
	Sol_Change_L5 = pd.DataFrame(columns=['Reliability'])
	Sol_Change_L10 = pd.DataFrame(columns=['Reliability'])
	Sol_Change_L20 = pd.DataFrame(columns=['Reliability'])
	
	i = 0 
	for sc in scenarios:
			i += 1
			dta_all = pd.DataFrame(columns=variables)
			for w in windows:
					for x in msd:
						for when in whens:
							dta = pd.read_csv('%s/%s-model_inputs_%s_%s_%s.csv'%(Location_of_Input_Data,sc,when,x,w), index_col = 0, parse_dates = True)		
							dta.reset_index(inplace=True)
							
							for v in variables:
								if 'sd' in v:
									v2 = v[:-10] # remove 10 characters
								else:
									v2 = v[:-11] # remove 11 characters
								if v2 in dta.columns: # gets data from individual df to main df with original name for each
									if w in v:
										if when in v:
											if ('_'+x+'_') in v:
												dta_all[v] = dta[v2]
												test = dta_all[v]
												length = len(test)
												for lol in range(50,length):
													if np.isnan(test[lol]):
														print(v,v2,'for',w,when,x)
														print(dta_all[v])
			if climate_data == True:
				dta_all.index = dta_index
				if 'rcp26' in sc:
					rcp_matrix = RCP26_DATA
				if 'rcp45' in sc:
					rcp_matrix = RCP45_DATA
				if 'rcp60' in sc:
					rcp_matrix = RCP60_DATA
				if 'rcp85' in sc:
					rcp_matrix = RCP85_DATA
				
				for col_name in new_rcp_cols:
					dta_all[col_name] = rcp_matrix.loc[1951:2099,col_name]

			solution = pd.read_csv('ORCA_data/ORCA_outputs/{}/{}-reliability.csv'.format(sc,sc), index_col = 0, parse_dates = True)
			solution.drop(solution.index[0:49],axis=0,inplace=True)
			solution = solution.loc[:,rel]

			Solution_L0 = pd.concat([Solution_L0,solution])
			Solution_L1 = pd.concat([Solution_L1,solution[1:]])
			Solution_L5 = pd.concat([Solution_L5,solution[5:]])
			Solution_L10 = pd.concat([Solution_L10,solution[10:]])
			Solution_L20 = pd.concat([Solution_L20,solution[20:]])

			dta_all.drop(dta_all.index[0:49],axis=0,inplace=True)
			Data_L0 = pd.concat([Data_L0,dta_all],axis=0)
			Data_L1 = pd.concat([Data_L1,dta_all[:-1]],axis=0)
			Data_L5 = pd.concat([Data_L5,dta_all.iloc[:-5]],axis=0)
			Data_L10 = pd.concat([Data_L10,dta_all.iloc[:-10]],axis=0)
			Data_L20 = pd.concat([Data_L20,dta_all.iloc[:-20]],axis=0)
			print('Completed %s of %s'%(i,len(scenarios)))
			
	Solution_L0.drop('Reliability', axis=1,inplace=True)	
	Solution_L1.drop('Reliability', axis=1,inplace=True)
	Solution_L5.drop('Reliability', axis=1,inplace=True)
	Solution_L10.drop('Reliability', axis=1,inplace=True)
	Solution_L20.drop('Reliability', axis=1,inplace=True)

	Sol_Change_L1.drop('Reliability', axis=1,inplace=True)
	Sol_Change_L5.drop('Reliability', axis=1,inplace=True)
	Sol_Change_L10.drop('Reliability', axis=1,inplace=True)
	Sol_Change_L20.drop('Reliability', axis=1,inplace=True)

	Solution_L0.reset_index(inplace=True)
	Solution_L1.reset_index(inplace=True)
	Solution_L5.reset_index(inplace=True)
	Solution_L10.reset_index(inplace=True)
	Solution_L20.reset_index(inplace=True)
	
	Solution_L0.drop('index',axis=1,inplace=True)
	Solution_L1.drop('index',axis=1,inplace=True)
	Solution_L5.drop('index',axis=1,inplace=True)
	Solution_L10.drop('index',axis=1,inplace=True)
	Solution_L20.drop('index',axis=1,inplace=True)

	Data_L0.reset_index(inplace=True)
	Data_L1.reset_index(inplace=True)
	Data_L5.reset_index(inplace=True)
	Data_L10.reset_index(inplace=True)
	Data_L20.reset_index(inplace=True)

	Data_L0.drop('datetime',axis=1,inplace=True)
	Data_L1.drop('datetime',axis=1,inplace=True)
	Data_L5.drop('datetime',axis=1,inplace=True)
	Data_L10.drop('datetime',axis=1,inplace=True)
	Data_L20.drop('datetime',axis=1,inplace=True)
	
	Data_L0.to_csv('%s/%s-Data_L0-climate.csv'%(Location_to_Save_Matrices,scenario_removed))
	Data_L1.to_csv('%s/%s-Data_L1-climate.csv'%(Location_to_Save_Matrices,scenario_removed))
	Data_L5.to_csv('%s/%s-Data_L5-climate.csv'%(Location_to_Save_Matrices,scenario_removed))
	Data_L10.to_csv('%s/%s-Data_L10-climate.csv'%(Location_to_Save_Matrices,scenario_removed))
	Data_L20.to_csv('%s/%s-Data_L20-climate.csv'%(Location_to_Save_Matrices,scenario_removed))

	Solution_L0.to_csv('%s/%s-Solution_L0.csv'%(Location_to_Save_Matrices,scenario_removed))
	Solution_L1.to_csv('%s/%s-Solution_L1.csv'%(Location_to_Save_Matrices,scenario_removed))
	Solution_L5.to_csv('%s/%s-Solution_L5.csv'%(Location_to_Save_Matrices,scenario_removed))
	Solution_L10.to_csv('%s/%s-Solution_L10.csv'%(Location_to_Save_Matrices,scenario_removed))
	Solution_L20.to_csv('%s/%s-Solution_L20.csv'%(Location_to_Save_Matrices,scenario_removed))

	## Getting testing data ##
	if scenario_removed != 'None':
		Testing_Solution = pd.read_csv('ORCA_data/ORCA_outputs/{}/{}-reliability.csv'.format(scenario_removed,scenario_removed), index_col = 0, parse_dates = True)
		Testing_Solution.drop(Testing_Solution.index[0:49],axis=0,inplace=True)
		Testing_Solution.reset_index(inplace=True)
		Testing_Solution = Testing_Solution.loc[:,rel]
		Testing_Solution_L0 = Testing_Solution
		Testing_Solution_L1 = Testing_Solution[1:]
		Testing_Solution_L5 = Testing_Solution[5:]
		Testing_Solution_L10 = Testing_Solution[10:]
		Testing_Solution_L20 = Testing_Solution[20:]

		Testing_Data = pd.DataFrame(columns=variables)
		Testing_Data_L0 = pd.DataFrame(columns=variables)
		Testing_Data_L1 = pd.DataFrame(columns=variables)
		Testing_Data_L5 = pd.DataFrame(columns=variables)
		Testing_Data_L10 = pd.DataFrame(columns=variables)
		Testing_Data_L20 = pd.DataFrame(columns=variables)

		for w in windows:
				for x in msd:
					for when in whens:
						dta = pd.read_csv('%s/%s-model_inputs_%s_%s_%s.csv'%(Location_of_Input_Data,scenario_removed,when,x,w), index_col = 0, parse_dates = True)		
						dta.reset_index(inplace=True)
						
						for v in variables:
							if 'sd' in v:
								v2 = v[:-10] # remove 10 characters
							else:
								v2 = v[:-11] # remove 11 characters
							if v2 in dta.columns: # gets data from individual df to main df with original name for each
								if w in v:
									if when in v:
										if ('_'+x+'_') in v:
											Testing_Data[v] = dta[v2]

		Testing_Data.index = dta_index

		Testing_Data.drop(Testing_Data.index[0:49],axis=0,inplace=True)
		Testing_Data.reset_index(inplace=True)
		Testing_Data.drop('datetime',axis=1,inplace=True)

		Testing_Data_L0 = Testing_Data
		Testing_Data_L1 = Testing_Data[:-1]
		Testing_Data_L5 = Testing_Data[:-5]
		Testing_Data_L10 = Testing_Data[:-10]
		Testing_Data_L20 = Testing_Data[:-20]

		Testing_Data_L0.to_csv('%s/%s-Testing_Data_L0-climate.csv'%(Location_to_Save_Matrices,scenario_removed))
		Testing_Data_L1.to_csv('%s/%s-Testing_Data_L1-climate.csv'%(Location_to_Save_Matrices,scenario_removed))
		Testing_Data_L5.to_csv('%s/%s-Testing_Data_L5-climate.csv'%(Location_to_Save_Matrices,scenario_removed))
		Testing_Data_L10.to_csv('%s/%s-Testing_Data_L10-climate.csv'%(Location_to_Save_Matrices,scenario_removed))
		Testing_Data_L20.to_csv('%s/%s-Testing_Data_L20-climate.csv'%(Location_to_Save_Matrices,scenario_removed))

		Testing_Solution_L0.to_csv('%s/%s-Testing_Solution_L0.csv'%(Location_to_Save_Matrices,scenario_removed))
		Testing_Solution_L1.to_csv('%s/%s-Testing_Solution_L1.csv'%(Location_to_Save_Matrices,scenario_removed))
		Testing_Solution_L5.to_csv('%s/%s-Testing_Solution_L5.csv'%(Location_to_Save_Matrices,scenario_removed))
		Testing_Solution_L10.to_csv('%s/%s-Testing_Solution_L10.csv'%(Location_to_Save_Matrices,scenario_removed))
		Testing_Solution_L20.to_csv('%s/%s-Testing_Solution_L20.csv'%(Location_to_Save_Matrices,scenario_removed))
	
	## Only for a training run where no testing data is needed ##
	else:
		Testing_Data_L0 = 0
		Testing_Data_L1 = 0
		Testing_Data_L5 = 0
		Testing_Data_L10 = 0
		Testing_Data_L20 = 0
		Testing_Solution_L0 = 0
		Testing_Solution_L1 = 0
		Testing_Solution_L5 = 0
		Testing_Solution_L10 = 0
		Testing_Solution_L20 = 0

	return (variables,Data_L0,Data_L1,Data_L5,Data_L10,Data_L20,
			Solution_L0,Solution_L1,Solution_L5,Solution_L10,Solution_L20,
			Testing_Data_L0,Testing_Data_L1,Testing_Data_L5,Testing_Data_L10,Testing_Data_L20,
			Testing_Solution_L0,Testing_Solution_L1,Testing_Solution_L5,Testing_Solution_L10,Testing_Solution_L20)



## To load data that has already been compiled ##
def Load_Data(sc):

	Data_L00 = pd.read_csv('%s/%s-Data_L0.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L01 = pd.read_csv('%s/%s-Data_L1.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L05 = pd.read_csv('%s/%s-Data_L5.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L10 = pd.read_csv('%s/%s-Data_L10.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L20 = pd.read_csv('%s/%s-Data_L20.csv'%(Location_to_Save_Matrices,sc), index_col=0)

	Solution_L00 = pd.read_csv('%s/%s-Solution_L0.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L01 = pd.read_csv('%s/%s-Solution_L1.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L05 = pd.read_csv('%s/%s-Solution_L5.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L10 = pd.read_csv('%s/%s-Solution_L10.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L20 = pd.read_csv('%s/%s-Solution_L20.csv'%(Location_to_Save_Matrices,sc), index_col=0)

	variables = Data_L00.columns

	if sc != 'None':

		Testing_Data_L00 = pd.read_csv('%s/%s-Testing_Data_L0.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L01 = pd.read_csv('%s/%s-Testing_Data_L1.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L05 = pd.read_csv('%s/%s-Testing_Data_L5.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L10 = pd.read_csv('%s/%s-Testing_Data_L10.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L20 = pd.read_csv('%s/%s-Testing_Data_L20.csv'%(Location_to_Save_Matrices,sc), index_col=0)

		Testing_Solution_L00 = pd.read_csv('%s/%s-Testing_Solution_L0.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L01 = pd.read_csv('%s/%s-Testing_Solution_L1.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L05 = pd.read_csv('%s/%s-Testing_Solution_L5.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L10 = pd.read_csv('%s/%s-Testing_Solution_L10.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L20 = pd.read_csv('%s/%s-Testing_Solution_L20.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)

	else:
		Testing_Data_L00 = 0
		Testing_Data_L01 = 0
		Testing_Data_L05 = 0
		Testing_Data_L10 = 0
		Testing_Data_L20 = 0

		Testing_Solution_L00 = 0
		Testing_Solution_L01 = 0
		Testing_Solution_L05 = 0
		Testing_Solution_L10 = 0
		Testing_Solution_L20 = 0

	return (variables,Data_L00,Data_L01,Data_L05,Data_L10,Data_L20,
			Solution_L00,Solution_L01,Solution_L05,Solution_L10,Solution_L20,
			Testing_Data_L00,Testing_Data_L01,Testing_Data_L05,Testing_Data_L10,Testing_Data_L20,
			Testing_Solution_L00,Testing_Solution_L01,Testing_Solution_L05,Testing_Solution_L10,Testing_Solution_L20)	



## To load data that has already been reduced to 500 features (to save time and space) ##
def Load_Data_REDUCED(sc):

	Data_L00 = pd.read_csv('%s/%s-Data_L00-REDUCED.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L01 = pd.read_csv('%s/%s-Data_L01-REDUCED.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L05 = pd.read_csv('%s/%s-Data_L05-REDUCED.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L10 = pd.read_csv('%s/%s-Data_L10-REDUCED.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L20 = pd.read_csv('%s/%s-Data_L20-REDUCED.csv'%(Location_to_Save_Matrices,sc), index_col=0)

	Solution_L00 = pd.read_csv('%s/%s-Solution_L0.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L01 = pd.read_csv('%s/%s-Solution_L1.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L05 = pd.read_csv('%s/%s-Solution_L5.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L10 = pd.read_csv('%s/%s-Solution_L10.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L20 = pd.read_csv('%s/%s-Solution_L20.csv'%(Location_to_Save_Matrices,sc), index_col=0)

	variables = Data_L00.columns

	if sc != 'None':

		Testing_Data_L00 = pd.read_csv('%s/%s-Testing_Data_L00-REDUCED.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L01 = pd.read_csv('%s/%s-Testing_Data_L01-REDUCED.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L05 = pd.read_csv('%s/%s-Testing_Data_L05-REDUCED.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L10 = pd.read_csv('%s/%s-Testing_Data_L10-REDUCED.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L20 = pd.read_csv('%s/%s-Testing_Data_L20-REDUCED.csv'%(Location_to_Save_Matrices,sc), index_col=0)

		Testing_Solution_L00 = pd.read_csv('%s/%s-Testing_Solution_L0.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L01 = pd.read_csv('%s/%s-Testing_Solution_L1.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L05 = pd.read_csv('%s/%s-Testing_Solution_L5.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L10 = pd.read_csv('%s/%s-Testing_Solution_L10.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L20 = pd.read_csv('%s/%s-Testing_Solution_L20.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)

	else:
		Testing_Data_L00 = 0
		Testing_Data_L01 = 0
		Testing_Data_L05 = 0
		Testing_Data_L10 = 0
		Testing_Data_L20 = 0

		Testing_Solution_L00 = 0
		Testing_Solution_L01 = 0
		Testing_Solution_L05 = 0
		Testing_Solution_L10 = 0
		Testing_Solution_L20 = 0

	return (variables,Data_L00,Data_L01,Data_L05,Data_L10,Data_L20,
			Solution_L00,Solution_L01,Solution_L05,Solution_L10,Solution_L20,
			Testing_Data_L00,Testing_Data_L01,Testing_Data_L05,Testing_Data_L10,Testing_Data_L20,
			Testing_Solution_L00,Testing_Solution_L01,Testing_Solution_L05,Testing_Solution_L10,Testing_Solution_L20)	



## To load data (including additional climate data) that has already been reduced to 500 features (to save time and space) ##
def Load_Data_REDUCED_climate(sc):

	Data_L00 = pd.read_csv('%s/%s-Data_L00-REDUCED-climate.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L01 = pd.read_csv('%s/%s-Data_L01-REDUCED-climate.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L05 = pd.read_csv('%s/%s-Data_L05-REDUCED-climate.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L10 = pd.read_csv('%s/%s-Data_L10-REDUCED-climate.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Data_L20 = pd.read_csv('%s/%s-Data_L20-REDUCED-climate.csv'%(Location_to_Save_Matrices,sc), index_col=0)

	Solution_L00 = pd.read_csv('%s/%s-Solution_L0.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L01 = pd.read_csv('%s/%s-Solution_L1.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L05 = pd.read_csv('%s/%s-Solution_L5.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L10 = pd.read_csv('%s/%s-Solution_L10.csv'%(Location_to_Save_Matrices,sc), index_col=0)
	Solution_L20 = pd.read_csv('%s/%s-Solution_L20.csv'%(Location_to_Save_Matrices,sc), index_col=0)

	variables = Data_L00.columns

	if sc != 'None':

		Testing_Data_L00 = pd.read_csv('%s/%s-Testing_Data_L00-REDUCED-climate.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L01 = pd.read_csv('%s/%s-Testing_Data_L01-REDUCED-climate.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L05 = pd.read_csv('%s/%s-Testing_Data_L05-REDUCED-climate.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L10 = pd.read_csv('%s/%s-Testing_Data_L10-REDUCED-climate.csv'%(Location_to_Save_Matrices,sc), index_col=0)
		Testing_Data_L20 = pd.read_csv('%s/%s-Testing_Data_L20-REDUCED-climate.csv'%(Location_to_Save_Matrices,sc), index_col=0)

		Testing_Solution_L00 = pd.read_csv('%s/%s-Testing_Solution_L0.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L01 = pd.read_csv('%s/%s-Testing_Solution_L1.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L05 = pd.read_csv('%s/%s-Testing_Solution_L5.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L10 = pd.read_csv('%s/%s-Testing_Solution_L10.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)
		Testing_Solution_L20 = pd.read_csv('%s/%s-Testing_Solution_L20.csv'%(Location_to_Save_Matrices,sc), index_col=0, header=None)

	else:
		Testing_Data_L00 = 0
		Testing_Data_L01 = 0
		Testing_Data_L05 = 0
		Testing_Data_L10 = 0
		Testing_Data_L20 = 0

		Testing_Solution_L00 = 0
		Testing_Solution_L01 = 0
		Testing_Solution_L05 = 0
		Testing_Solution_L10 = 0
		Testing_Solution_L20 = 0

	return (variables,Data_L00,Data_L01,Data_L05,Data_L10,Data_L20,
			Solution_L00,Solution_L01,Solution_L05,Solution_L10,Solution_L20,
			Testing_Data_L00,Testing_Data_L01,Testing_Data_L05,Testing_Data_L10,Testing_Data_L20,
			Testing_Solution_L00,Testing_Solution_L01,Testing_Solution_L05,Testing_Solution_L10,Testing_Solution_L20)	

