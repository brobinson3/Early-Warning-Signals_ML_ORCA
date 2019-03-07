import numpy as np
import pandas as pd


Reliability = ['Rolling_Reliability_50yr','Rolling_Reliability_40yr','Rolling_Reliability_30yr','Rolling_Reliability_20yr','Rolling_Reliability_10yr','Reliability']


def Data_Compilation():

	with open('ORCA_data/scenario_names_all.txt') as f: # change file name here to run different scenarios
		scenarios = f.read().splitlines()

	dta = pd.read_csv('ORCA_data/ORCA_inputs/%s-modinputs_m_10.csv'%(scenarios[1]), index_col = 0, parse_dates = True)

	msd = ['m','sd']
	variables = []
	windows = ['10','20','30','40','50']
	

	for v in dta.columns:
		for x in msd:
			for w in windows:
				l = []
				l.append(v)
				l.append('_')
				l.append(w)
				l.append('_')
				l.append(x)
				s = ''.join(l)
				variables.append(s)

	Data_L0 = pd.DataFrame()
	Data_L1 = pd.DataFrame()
	Data_L5 = pd.DataFrame()
	Data_L10 = pd.DataFrame()
	Data_L20 = pd.DataFrame()
	
	for R in Reliability:
		Solution_L0 = pd.DataFrame(columns=['Reliability'])
		Solution_L1 = pd.DataFrame(columns=['Reliability'])
		Solution_L5 = pd.DataFrame(columns=['Reliability'])
		Solution_L10 = pd.DataFrame(columns=['Reliability'])
		Solution_L20 = pd.DataFrame(columns=['Reliability'])

		Sol_Change_L1 = pd.DataFrame(columns=['Reliability'])
		Sol_Change_L5 = pd.DataFrame(columns=['Reliability'])
		Sol_Change_L10 = pd.DataFrame(columns=['Reliability'])
		Sol_Change_L20 = pd.DataFrame(columns=['Reliability'])
		
		for sc in scenarios:
			if R == Reliability[4]:
				dta_all = pd.DataFrame(index=dta.index,columns=variables)
				for w in windows:
					for x in msd:
						dta = pd.read_csv('ORCA_data/ORCA_inputs/%s-modinputs_%s_%s.csv'%(sc,x,w), index_col = 0, parse_dates = True)		
						for v in variables:
							if x == 'sd':
								v2 = v[:-6] # remove 6 characters
							if x == 'm':
								v2 = v[:-5] # remove 5 characters
							if v2 in dta.columns: # gets data from individual df to main df with original name for each
								if w in v:
									dta_all[v] = dta[v2]

			solution = pd.read_csv('ORCA_data/ORCA_outputs/{}/{}-results2-annual.csv'.format(sc,sc), index_col = 0, parse_dates = True)
			solution.drop(solution.index[0:49],axis=0,inplace=True)
			solution = solution.loc[:,R]
			Solution_L0 = pd.concat([Solution_L0,solution])
			Solution_L1 = pd.concat([Solution_L1,solution[1:]])
			Solution_L5 = pd.concat([Solution_L5,solution[5:]])
			Solution_L10 = pd.concat([Solution_L10,solution[10:]])
			Solution_L20 = pd.concat([Solution_L20,solution[20:]])

			solution = solution.values
			change_L20 = np.subtract(solution[20:],solution[:-20])
			change_L20 = pd.DataFrame(change_L20)
			change_L10 = np.subtract(solution[10:],solution[:-10])
			change_L10 = pd.DataFrame(change_L10)
			change_L5 = np.subtract(solution[5:],solution[:-5])
			change_L5 = pd.DataFrame(change_L5)
			change_L1 = np.subtract(solution[1:],solution[:-1])
			change_L1 = pd.DataFrame(change_L1)

			Sol_Change_L1 = pd.concat([Sol_Change_L1,change_L1])
			Sol_Change_L5 = pd.concat([Sol_Change_L5,change_L5])
			Sol_Change_L10 = pd.concat([Sol_Change_L10,change_L10])
			Sol_Change_L20 = pd.concat([Sol_Change_L20,change_L20])

			if R == Reliability[4]:
				dta_all.drop(dta.index[0:49],axis=0,inplace=True)
				Data_L0 = pd.concat([Data_L0,dta_all],axis=0)
				Data_L1 = pd.concat([Data_L1,dta_all[:-1]],axis=0)
				Data_L5 = pd.concat([Data_L5,dta_all.iloc[:-5]],axis=0)
				Data_L10 = pd.concat([Data_L10,dta_all.iloc[:-10]],axis=0)
				Data_L20 = pd.concat([Data_L20,dta_all.iloc[:-20]],axis=0)
			
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

		if R == Reliability[4]:
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

			### Now get rid of cheaters! (i.e. results of high/low reliability, not potential causes)
			Cheaters = ['Total_pumped_from_DEL_10_m','Total_pumped_from_DEL_20_m','Total_pumped_from_DEL_30_m','Total_pumped_from_DEL_40_m','Total_pumped_from_DEL_50_m',
					'Total_DEL_SODD_10_sd','Total_DEL_SODD_20_sd','Total_DEL_SODD_30_sd','Total_DEL_SODD_40_sd','Total_DEL_SODD_50_sd',
					'Total_pumped_from_DEL_10_sd','Total_pumped_from_DEL_20_sd','Total_pumped_from_DEL_30_sd','Total_pumped_from_DEL_40_sd','Total_pumped_from_DEL_50_sd',
					'Total_SODD_from_RES_10_sd','Total_SODD_from_RES_20_sd','Total_SODD_from_RES_30_sd','Total_SODD_from_RES_40_sd','Total_SODD_from_RES_50_sd',
					'Total_Shortage_10_sd','Total_Shortage_20_sd','Total_Shortage_30_sd','Total_Shortage_40_sd','Total_Shortage_50_sd',
					'Total_SODD_from_RES_10_m','Total_SODD_from_RES_20_m','Total_SODD_from_RES_30_m','Total_SODD_from_RES_40_m','Total_SODD_from_RES_50_m',
					'Total_DEL_SODD_10_m','Total_DEL_SODD_20_m','Total_DEL_SODD_30_m','Total_DEL_SODD_40_m','Total_DEL_SODD_50_m',
					'Total_Shortage_10_m','Total_Shortage_20_m','Total_Shortage_30_m','Total_Shortage_40_m','Total_Shortage_50_m']

			Data_L0.drop(Cheaters,axis=1,inplace=True)
			Data_L1.drop(Cheaters,axis=1,inplace=True)
			Data_L5.drop(Cheaters,axis=1,inplace=True)
			Data_L10.drop(Cheaters,axis=1,inplace=True)
			Data_L20.drop(Cheaters,axis=1,inplace=True)

			# Data_L0.to_csv('Data_Matrices/Data_L0.csv')
			# Data_L1.to_csv('Data_Matrices/Data_L1.csv')
			# Data_L5.to_csv('Data_Matrices/Data_L5.csv')
			# Data_L10.to_csv('Data_Matrices/Data_L10.csv')
			# Data_L20.to_csv('Data_Matrices/Data_L20.csv')

		# Solution_L0.to_csv('Data_Matrices/Solution_L0_%s.csv'%(R))
		# Solution_L1.to_csv('Data_Matrices/Solution_L1_%s.csv'%(R))
		# Solution_L5.to_csv('Data_Matrices/Solution_L5_%s.csv'%(R))
		# Solution_L10.to_csv('Data_Matrices/Solution_L10_%s.csv'%(R))
		# Solution_L20.to_csv('Data_Matrices/Solution_L20_%s.csv'%(R))

		Sol_Change_L1.to_csv('Data_Matrices/Sol_Change_L1_%s.csv'%(R))
		Sol_Change_L5.to_csv('Data_Matrices/Sol_Change_L5_%s.csv'%(R))
		Sol_Change_L10.to_csv('Data_Matrices/Sol_Change_L10_%s.csv'%(R))
		Sol_Change_L20.to_csv('Data_Matrices/Sol_Change_L20_%s.csv'%(R))



def Load_Data():

	Data_L0 = pd.read_csv('Data_Matrices/Data_L0.csv', index_col=0)
	columns = Data_L0.columns
	Data_L1 = pd.read_csv('Data_Matrices/Data_L1.csv', index_col=0)
	Data_L5 = pd.read_csv('Data_Matrices/Data_L5.csv', index_col=0)
	Data_L10 = pd.read_csv('Data_Matrices/Data_L10.csv', index_col=0)
	Data_L20 = pd.read_csv('Data_Matrices/Data_L20.csv', index_col=0)
	
	Solution_L0_R01 = pd.read_csv('Data_Matrices/Solution_L0_'+Reliability[5]+'.csv', index_col=0)
	Solution_L1_R01 = pd.read_csv('Data_Matrices/Solution_L1_'+Reliability[5]+'.csv', index_col=0)
	Solution_L5_R01 = pd.read_csv('Data_Matrices/Solution_L5_'+Reliability[5]+'.csv', index_col=0)
	Solution_L10_R01 = pd.read_csv('Data_Matrices/Solution_L10_'+Reliability[5]+'.csv', index_col=0)
	Solution_L20_R01 = pd.read_csv('Data_Matrices/Solution_L20_'+Reliability[5]+'.csv', index_col=0)
	
	Solution_L0_R10 = pd.read_csv('Data_Matrices/Solution_L0_'+Reliability[4]+'.csv', index_col=0)
	Solution_L1_R10 = pd.read_csv('Data_Matrices/Solution_L1_'+Reliability[4]+'.csv', index_col=0)
	Solution_L5_R10 = pd.read_csv('Data_Matrices/Solution_L5_'+Reliability[4]+'.csv', index_col=0)
	Solution_L10_R10 = pd.read_csv('Data_Matrices/Solution_L10_'+Reliability[4]+'.csv', index_col=0)
	Solution_L20_R10 = pd.read_csv('Data_Matrices/Solution_L20_'+Reliability[4]+'.csv', index_col=0)

	Solution_L0_R20 = pd.read_csv('Data_Matrices/Solution_L0_'+Reliability[3]+'.csv', index_col=0)
	Solution_L1_R20 = pd.read_csv('Data_Matrices/Solution_L1_'+Reliability[3]+'.csv', index_col=0)
	Solution_L5_R20 = pd.read_csv('Data_Matrices/Solution_L5_'+Reliability[3]+'.csv', index_col=0)
	Solution_L10_R20 = pd.read_csv('Data_Matrices/Solution_L10_'+Reliability[3]+'.csv', index_col=0)
	Solution_L20_R20 = pd.read_csv('Data_Matrices/Solution_L20_'+Reliability[3]+'.csv', index_col=0)

	Solution_L0_R30 = pd.read_csv('Data_Matrices/Solution_L0_'+Reliability[2]+'.csv', index_col=0)
	Solution_L1_R30 = pd.read_csv('Data_Matrices/Solution_L1_'+Reliability[2]+'.csv', index_col=0)
	Solution_L5_R30 = pd.read_csv('Data_Matrices/Solution_L5_'+Reliability[2]+'.csv', index_col=0)
	Solution_L10_R30 = pd.read_csv('Data_Matrices/Solution_L10_'+Reliability[2]+'.csv', index_col=0)
	Solution_L20_R30 = pd.read_csv('Data_Matrices/Solution_L20_'+Reliability[2]+'.csv', index_col=0)

	Solution_L0_R40 = pd.read_csv('Data_Matrices/Solution_L0_'+Reliability[1]+'.csv', index_col=0)
	Solution_L1_R40 = pd.read_csv('Data_Matrices/Solution_L1_'+Reliability[1]+'.csv', index_col=0)
	Solution_L5_R40 = pd.read_csv('Data_Matrices/Solution_L5_'+Reliability[1]+'.csv', index_col=0)
	Solution_L10_R40 = pd.read_csv('Data_Matrices/Solution_L10_'+Reliability[1]+'.csv', index_col=0)
	Solution_L20_R40 = pd.read_csv('Data_Matrices/Solution_L20_'+Reliability[1]+'.csv', index_col=0)

	Solution_L0_R50 = pd.read_csv('Data_Matrices/Solution_L0_'+Reliability[0]+'.csv', index_col=0)
	Solution_L1_R50 = pd.read_csv('Data_Matrices/Solution_L1_'+Reliability[0]+'.csv', index_col=0)
	Solution_L5_R50 = pd.read_csv('Data_Matrices/Solution_L5_'+Reliability[0]+'.csv', index_col=0)
	Solution_L10_R50 = pd.read_csv('Data_Matrices/Solution_L10_'+Reliability[0]+'.csv', index_col=0)
	Solution_L20_R50 = pd.read_csv('Data_Matrices/Solution_L20_'+Reliability[0]+'.csv', index_col=0)

	return (columns,Data_L0,Data_L1,Data_L5,Data_L10,Data_L20,
			Solution_L0_R01,Solution_L1_R01,Solution_L5_R01,Solution_L10_R01,Solution_L20_R01,
			Solution_L0_R10,Solution_L1_R10,Solution_L5_R10,Solution_L10_R10,Solution_L20_R10,
			Solution_L0_R20,Solution_L1_R20,Solution_L5_R20,Solution_L10_R20,Solution_L20_R20,
			Solution_L0_R30,Solution_L1_R30,Solution_L5_R30,Solution_L10_R30,Solution_L20_R30,
			Solution_L0_R40,Solution_L1_R40,Solution_L5_R40,Solution_L10_R40,Solution_L20_R40,
			Solution_L0_R50,Solution_L1_R50,Solution_L5_R50,Solution_L10_R50,Solution_L20_R50)



def Load_Change_Sol():

	Sol_Change_L1_R01 = pd.read_csv('Data_Matrices/Sol_change_L1_'+Reliability[5]+'.csv', index_col=0)
	Sol_Change_L5_R01 = pd.read_csv('Data_Matrices/Sol_change_L5_'+Reliability[5]+'.csv', index_col=0)
	Sol_Change_L10_R01 = pd.read_csv('Data_Matrices/Sol_change_L10_'+Reliability[5]+'.csv', index_col=0)
	Sol_Change_L20_R01 = pd.read_csv('Data_Matrices/Sol_change_L20_'+Reliability[5]+'.csv', index_col=0) 

	Sol_Change_L1_R10 = pd.read_csv('Data_Matrices/Sol_change_L1_'+Reliability[4]+'.csv', index_col=0)
	Sol_Change_L5_R10 = pd.read_csv('Data_Matrices/Sol_change_L5_'+Reliability[4]+'.csv', index_col=0)
	Sol_Change_L10_R10 = pd.read_csv('Data_Matrices/Sol_change_L10_'+Reliability[4]+'.csv', index_col=0)
	Sol_Change_L20_R10 = pd.read_csv('Data_Matrices/Sol_change_L20_'+Reliability[4]+'.csv', index_col=0) 

	Sol_Change_L1_R20 = pd.read_csv('Data_Matrices/Sol_change_L1_'+Reliability[3]+'.csv', index_col=0)
	Sol_Change_L5_R20 = pd.read_csv('Data_Matrices/Sol_change_L5_'+Reliability[3]+'.csv', index_col=0)
	Sol_Change_L10_R20 = pd.read_csv('Data_Matrices/Sol_change_L10_'+Reliability[3]+'.csv', index_col=0)
	Sol_Change_L20_R20 = pd.read_csv('Data_Matrices/Sol_change_L20_'+Reliability[3]+'.csv', index_col=0) 

	Sol_Change_L1_R30 = pd.read_csv('Data_Matrices/Sol_change_L1_'+Reliability[2]+'.csv', index_col=0)
	Sol_Change_L5_R30 = pd.read_csv('Data_Matrices/Sol_change_L5_'+Reliability[2]+'.csv', index_col=0)
	Sol_Change_L10_R30 = pd.read_csv('Data_Matrices/Sol_change_L10_'+Reliability[2]+'.csv', index_col=0)
	Sol_Change_L20_R30 = pd.read_csv('Data_Matrices/Sol_change_L20_'+Reliability[2]+'.csv', index_col=0) 

	Sol_Change_L1_R40 = pd.read_csv('Data_Matrices/Sol_change_L1_'+Reliability[1]+'.csv', index_col=0)
	Sol_Change_L5_R40 = pd.read_csv('Data_Matrices/Sol_change_L5_'+Reliability[1]+'.csv', index_col=0)
	Sol_Change_L10_R40 = pd.read_csv('Data_Matrices/Sol_change_L10_'+Reliability[1]+'.csv', index_col=0)
	Sol_Change_L20_R40 = pd.read_csv('Data_Matrices/Sol_change_L20_'+Reliability[1]+'.csv', index_col=0) 

	Sol_Change_L1_R50 = pd.read_csv('Data_Matrices/Sol_change_L1_'+Reliability[0]+'.csv', index_col=0)
	Sol_Change_L5_R50 = pd.read_csv('Data_Matrices/Sol_change_L5_'+Reliability[0]+'.csv', index_col=0)
	Sol_Change_L10_R50 = pd.read_csv('Data_Matrices/Sol_change_L10_'+Reliability[0]+'.csv', index_col=0)
	Sol_Change_L20_R50 = pd.read_csv('Data_Matrices/Sol_change_L20_'+Reliability[0]+'.csv', index_col=0) 


	return (Sol_Change_L1_R01,Sol_Change_L5_R01,Sol_Change_L10_R01,Sol_Change_L20_R01,
			Sol_Change_L1_R10,Sol_Change_L5_R10,Sol_Change_L10_R10,Sol_Change_L20_R10,
			Sol_Change_L1_R20,Sol_Change_L5_R20,Sol_Change_L10_R20,Sol_Change_L20_R20,
			Sol_Change_L1_R30,Sol_Change_L5_R30,Sol_Change_L10_R30,Sol_Change_L20_R30,
			Sol_Change_L1_R40,Sol_Change_L5_R40,Sol_Change_L10_R40,Sol_Change_L20_R40,
			Sol_Change_L1_R50,Sol_Change_L5_R50,Sol_Change_L10_R50,Sol_Change_L20_R50)




