### All vizualization/plotting scripts ###

import numpy as np 
import pandas as pd
from scipy import stats
from Data_Formatting import *
from ML_Methods import *
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error


## Initial Setup with common variables, names, parameters ##
with open('ORCA_data/scenario_names_all.txt') as f: # change file name here to run different scenarios
	scenarios = f.read().splitlines()

Leads = ['L00','L01','L05','L10','L20']
sol_type = 'value'
num_features_all = [1,2,3,4,5,10,500]

thresholds_all = [0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86]

REG_methods = ['lin_reg','ply_reg','svr_lin','svr_3rd','dcn_tre','rdm_for','ada_bst','grd_bst','knl_rdg','nst_nbr_uni']
REG_method_names = ['Linear','Polynomial','Linear SVR','3rd Degree SVR','Decision Tree','Random Forest','AdaBoost','Gradient Boosting','Kernel Ridge','K Nearest Neighbors']

CLS_methods = ['nst_nbr','log_reg','svm_3rd','rdm_for','mlp_cls','ada_bst','gsn_nbc']#,'grd_bst','gsn_prc','dcn_tre','svm_rbf','svm_lin','rad_nbr_uni','rad_nbr_dst','qdr_dsc','svm_2nd']
CLS_method_names = ['K Nearest Neighbors','Logistic Regression','3rd Degree SVM','Random Forest','Multi-layer Perceptron','AdaBoost','Naive Bayes']

colors_to_use = ['blue','red','brown','green','cyan','m','lawngreen','tan','grey','darkcyan','violet','cornflowerblue','lightpink','purple']
Leads_to_plot = [0,1,5,10,20]

REG_custom_legend = []
for i in range(len(REG_methods)):
	REG_custom_legend.append(Line2D([0], [0], color='w', markerfacecolor=colors_to_use[i], marker='s', markersize=10))

CLS_custom_legend = []
for i in range(len(CLS_methods)):
	CLS_custom_legend.append(Line2D([0], [0], color=colors_to_use[i]))#, marker='s', markersize=10))



### REGRESSION: LEADS VS R2 ###
def REGRESSION__Leads_vs_R2(date,save=False):
	i=0
	num_features = 500
	for method in REG_methods:
		R2s = []
		R2s_max = []
		R2s_min = []
		R2s_median = []
		print(method)
		for Lead in Leads:
			Years = 99-int(Lead[1:])
			comp_R2s = []
			all_predictions = []
			all_solutions = []
			for sc in scenarios:
				results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,num_features,sol_type,'R30',method),index_col=0)
				results.columns = ['Solution','Predictions']
				x = get_r2_python(results['Solution'],results['Predictions'])
				for year in range(Years):
					all_solutions.append(results.loc[year,'Solution'])
					all_predictions.append(results.loc[year,'Predictions'])
				comp_R2s.append(x)
			y = get_r2_python(all_solutions,all_predictions)
			all_yearly_R2s = []
			for year in range(Years):
				current_year_prediction = []
				current_year_solution = []
				for sc in scenarios:
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,num_features,sol_type,'R30',method),index_col=0)
					results.columns = ['Solution','Predictions']
					current_year_prediction.append(results.loc[year,'Predictions'])
					current_year_solution.append(results.loc[year,'Solution'])
				yearly_R2 = get_r2_python(current_year_solution,current_year_prediction)
				all_yearly_R2s.append(yearly_R2)

			R2s.append(np.mean(comp_R2s))
			R2s_max.append(np.percentile(comp_R2s,90))
			R2s_min.append(np.percentile(comp_R2s,10))
			R2s_median.append(np.median(comp_R2s))
			print('---',Lead)
			print('---R2 aggregated by scenario:',R2s[-1])
			print('---R2 aggregated by year:',np.mean(all_yearly_R2s))
			print('---R2 taken after all data is aggregated:',y)

		plt.plot(Leads_to_plot,R2s_median,color=colors_to_use[i])
		plt.scatter(Leads_to_plot,R2s_max,color=colors_to_use[i],marker=7)
		plt.scatter(Leads_to_plot,R2s_min,color=colors_to_use[i],marker=6)
		i += 1
	plt.xlabel('Lead Times (years)',fontweight='bold',fontsize=12)
	plt.ylabel('R-squared',fontweight='bold',fontsize=12)
	plt.xticks(Leads_to_plot,fontsize=12)
	plt.grid(linestyle='--')
	plt.xlim(-0.5,20.5)
	plt.ylim(0,1)
	
	types_of_lines = [Line2D([0],[0],color='k'), Line2D([0],[0],color='w',marker=7,markerfacecolor='k'), Line2D([0],[0],color='w',marker=6,markerfacecolor='k')]
	legend2 = plt.legend(handles=types_of_lines,labels=['Median','90th Percentile','10th Percentile'],title='Marker Types',loc='lower left',ncol=1,bbox_to_anchor=(1.01,0))
	plt.legend(handles=REG_custom_legend,labels=REG_method_names,title='Type of Regression',loc='center left',bbox_to_anchor=(1.01,0.6))
	plt.gca().add_artist(legend2)
	if save == True:
		plt.savefig('All_Figures/Regression_Leads-vs-R2__with_med_bounds_%s.svg'%(date),bbox_inches='tight')
	plt.show()



### REGRESSION: LEADS VS RMSE ###
def REGRESSION__Leads_vs_RMSE(date,save=False):
	i=0
	num_features = 500
	for method in REG_methods:
		RMSEs = []
		RMSEs_max = []
		RMSEs_min = []
		RMSEs_median = []
		for Lead in Leads:
			comp_RMSEs = []
			for sc in scenarios:
				results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,num_features,sol_type,'R30',method),index_col=0)
				results.columns = ['Solution','Predictions']
				x = (mean_squared_error(results['Solution'],results['Predictions']))**(0.5)
				comp_RMSEs.append(x)
			RMSEs.append(np.mean(comp_RMSEs))
			RMSEs_max.append(np.percentile(comp_RMSEs,90))
			RMSEs_min.append(np.percentile(comp_RMSEs,10))
			RMSEs_median.append(np.median(comp_RMSEs))
		plt.plot(Leads_to_plot,RMSEs,label=[method],color=colors_to_use[i])
		plt.plot(Leads_to_plot,RMSEs_median,color=colors_to_use[i],linestyle='--')
		plt.scatter(Leads_to_plot,RMSEs_max,color=colors_to_use[i],marker=7)
		plt.scatter(Leads_to_plot,RMSEs_min,color=colors_to_use[i],marker=6)
		i += 1
	plt.xlabel('Lead Times (years)',fontweight='bold',fontsize=12)
	plt.ylabel('RMSE',fontweight='bold',fontsize=12)
	plt.xticks(Leads_to_plot,fontsize=12)
	plt.grid(linestyle='--')
	plt.xlim(-0.5,20.5)
	plt.ylim(-0.05,1)
	
	types_of_lines = [Line2D([0],[0],color='k'), Line2D([0],[0],color='k',linestyle='--'), Line2D([0],[0],color='w',marker=7,markerfacecolor='k'), Line2D([0],[0],color='w',marker=6,markerfacecolor='k')]
	legend2 = plt.legend(handles=types_of_lines,labels=['Mean','Median','90th Percentile','10th Percentile'],title='Marker Types',loc='lower left',ncol=1,bbox_to_anchor=(1.01,0))
	plt.legend(handles=REG_custom_legend,labels=REG_method_names,title='Type of Regression',loc='center left',bbox_to_anchor=(1.01,0.65))
	plt.gca().add_artist(legend2)

	if save == True:
		plt.savefig('All_Figures/Regression_Leads-vs-RMSE__with_med_bounds_%s.svg'%(date),bbox_inches='tight')
	plt.show()



### CLASSIFICATION: THRESHOLDS VS TP/TN RATIOS ###
def CLASSIFICATION__Thresholds_vs_TP_TN(date,save=False):
	num_features = 500
	CLS_custom_legend.append(Line2D([0],[0],color='k',linewidth=2))
	CLS_method_names.append('Ratio of Possible Positives')
	
	for Lead in Leads:
		i=0
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		for method in CLS_methods:
			TP_ratio_to_plot = []
			TN_ratio_to_plot = []
			min_95 = []
			max_95 = []
			Possible_Positives_Ratio = []
			for threshold in thresholds_all:
				TN = 0
				FP = 0
				TP = 0
				FN = 0
				Scenario_TP_ratio = []
				Scenario_TN_ratio = []
				for sc in scenarios:
					Scenario_FN = 0
					Scenario_TP = 0
					Scenario_TN = 0
					Scenario_FP = 0
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
					results.columns = ['Solution','Predictions']
					for row in range(0,len(results['Solution'])):
						if results['Solution'][row] == True and results['Predictions'][row] == True: # both are above the threshold (not vulnerable)
							TN += 1
							Scenario_TN += 1
						elif results['Solution'][row] == True and results['Predictions'][row] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							FP += 1
							Scenario_FP += 1
						elif results['Solution'][row] == False and results['Predictions'][row] == False: # both are below the threshold (vulnerable)
							TP += 1
							Scenario_TP += 1
						elif results['Solution'][row] == False and results['Predictions'][row] == True: # solution is below threshold but prediction is above the threshold
							FN += 1
							Scenario_FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
					try:
						Scenario_TP_ratio.append(Scenario_TP/(Scenario_TP+Scenario_FN))
					except:
						Scenario_TP_ratio.append(0)
					try:
						Scenario_TN_ratio.append(Scenario_TN/(Scenario_TN+Scenario_FP))
					except:
						Scenario_TN_ratio.append(0)
				TP_ratio = TP/(TP+FN)
				TN_ratio = TN/(TN+FP)
				TP_ratio_to_plot.append(np.median(Scenario_TP_ratio))
				TN_ratio_to_plot.append(np.median(Scenario_TN_ratio))
				Possible_Positives_Ratio.append((TP+FN)/(TP+FN+FP+TN))
				min_95.append(np.percentile(Scenario_TP_ratio,10))
				max_95.append(np.percentile(Scenario_TP_ratio,90))
			plt.plot(thresholds_all,TP_ratio_to_plot,c=colors_to_use[i])
			plt.plot(thresholds_all,TN_ratio_to_plot,c=colors_to_use[i],linestyle='--')
			plt.scatter(thresholds_all,min_95,c=colors_to_use[i],marker=6)
			plt.scatter(thresholds_all,max_95,c=colors_to_use[i],marker=7)
			i += 1
		plt.plot(thresholds_all,Possible_Positives_Ratio,color='k',linewidth=2)
		plt.title('TP and TN Classification Ratios for Lead %s'%(Lead[lead_cut:]),fontsize=14,fontweight='bold')
		plt.ylabel('Ratio',fontweight='bold',fontsize=12)
		plt.xlabel('Thresholds',fontweight='bold',fontsize=12)
		plt.xticks(thresholds_all,fontsize=10)
		plt.grid(linestyle='--')
		legend2 = plt.legend(handles=[Line2D([0],[0],color='k'),Line2D([0],[0],color='k',linestyle='--'),Line2D([0],[0],color='w',marker=6,markerfacecolor='k'),Line2D([0],[0],color='w',marker=7,markerfacecolor='k')],labels=['TPR','TNR','10th Percentile','90th Percentile'],title='Line Types',loc='lower left',bbox_to_anchor=(1.01,0))
		plt.legend(handles=CLS_custom_legend,labels=CLS_method_names,title='Type of Classification',loc='center left',bbox_to_anchor=(1.01,0.6))
		plt.gca().add_artist(legend2)
		if save == True:
			plt.savefig('All_Figures/Classification_Thresholds-vs-TP-TN_%s_%s.svg'%(Lead,date),bbox_inches='tight')
		plt.show()



### REGRESSION: FEATURES VS R2 ###
def REGRESSION__Features_vs_R2(date,save=False):
	breaks = [15,490]
	for Lead in Leads:
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		i = 0
		fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,gridspec_kw={'hspace':0,'wspace':0.02})
		for method in REG_methods:
			R2s = []
			R2s_max = []
			R2s_min = []
			R2s_median = []
			for num_features in num_features_all:
				comp_R2s = []
				for sc in scenarios:
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,num_features,sol_type,'R30',method),index_col=0)
					results.columns = ['Solution','Predictions']
					x = get_r2_python(results['Solution'],results['Predictions'])
					comp_R2s.append(x)
				R2s.append(np.mean(comp_R2s))
				R2s_max.append(np.percentile(comp_R2s,90))
				R2s_min.append(np.percentile(comp_R2s,10))
				R2s_median.append(np.median(comp_R2s))
			# one line of calcs
			R2s = R2s_median
			mid_line_slope = (R2s[-1]-R2s[-2])/(num_features_all[-1]-num_features_all[-2])
			left_line = R2s[:-1]
			left_line.append((breaks[0]-num_features_all[-2])*mid_line_slope + R2s[-2])
			left_line_x = num_features_all[:-1]
			left_line_x.append(breaks[0])
			right_line = []
			right_line.append(R2s[-1] - (num_features_all[-1]-breaks[1])*mid_line_slope)
			right_line.append(R2s[-1])
			right_line_x = []
			right_line_x.append(breaks[1])
			right_line_x.append(num_features_all[-1])
			ax1.plot(left_line_x,left_line,color=colors_to_use[i])
			ax2.plot(right_line_x,right_line,color=colors_to_use[i])
			ax1.scatter([1,2,3,4,5,10],R2s_max[:6],color=colors_to_use[i],marker=7)
			ax2.scatter([500],R2s_max[-1],color=colors_to_use[i],marker=7)
			ax1.scatter([1,2,3,4,5,10],R2s_min[:6],color=colors_to_use[i],marker=6)
			ax2.scatter([500],R2s_min[-1],color=colors_to_use[i],marker=6)
			i += 1
		ax1.spines['right'].set_visible(False)
		ax2.spines['left'].set_visible(False)
		ax1.yaxis.tick_left()
		ax1.tick_params(labelright='off')
		ax2.yaxis.tick_right()
		d = 0.015
		kwargs = dict(transform=ax1.transAxes,color='k',clip_on=False)
		ax1.plot((1-d,1+d),(-d,+d),**kwargs)
		ax1.plot((1-d,1+d),(1-d,1+d),**kwargs)
		kwargs.update(transform=ax2.transAxes)
		ax2.plot((-d,+d),(1-d,1+d),**kwargs)
		ax2.plot((-d,+d),(-d,+d),**kwargs)

		ax1.set_xticks([1,2,3,4,5,10])
		ax2.set_xticks([500])
		ax1.grid(linestyle='--')
		ax2.grid(linestyle='--')
		plt.ylim(0,1)
		fig.text(0.5,0.03,'Number of Features Used',ha='center',fontweight='bold',fontsize=12)
		fig.text(0.04,0.5,'R-squared',fontweight='bold',fontsize=12,va='center',rotation='vertical')
		fig.text(0.5,0.92,'Performance for Lead %s'%(Lead[lead_cut:]),fontweight='bold',fontsize=14,ha='center')
		
		types_of_lines = [Line2D([0],[0],color='k'), Line2D([0],[0],color='w',marker=7,markerfacecolor='k'), Line2D([0],[0],color='w',marker=6,markerfacecolor='k')]
		legend2 = plt.legend(handles=types_of_lines,labels=['Median','90th Percentile','10th Percentile'],title='Marker Types',loc='lower left',ncol=1,bbox_to_anchor=(1.01,0))
		plt.legend(REG_custom_legend,REG_method_names,title='Type of Regression',loc='center left',bbox_to_anchor=(1.01,0.6))
		plt.gca().add_artist(legend2)
		if save == True:
			plt.savefig('All_Figures/Regression_Features-vs-R2_%s_with-percentiles_%s.svg'%(Lead,date),bbox_inches='tight')
		plt.show()



### CLASSIFICATION: FEATURES VS TPR ###
def CLASSIFICATION__Features_vs_TPR(date,save=False):
	breaks = [15,490]
	threshold = 0.76
	for Lead in Leads:
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		i = 0
		fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,gridspec_kw={'hspace':0,'wspace':0.02})
		for method in CLS_methods:
			R2s = []
			TP_ratio_to_plot = []
			TN_ratio_to_plot = []
			for num_features in num_features_all:
				TN = 0
				FP = 0
				TP = 0
				FN = 0
				for sc in scenarios:
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
					results.columns = ['Solution','Predictions']
					for row in range(0,len(results['Solution'])):
						if results['Solution'][row] == True and results['Predictions'][row] == True: # both are above the threshold (not vulnerable)
							TN += 1
						elif results['Solution'][row] == True and results['Predictions'][row] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							FP += 1
						elif results['Solution'][row] == False and results['Predictions'][row] == False: # both are below the threshold (vulnerable)
							TP += 1
						elif results['Solution'][row] == False and results['Predictions'][row] == True: # solution is below threshold but prediction is above the threshold
							FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
				TP_ratio = TP/(TP+FN)
				TN_ratio = TN/(TN+FP)
				TP_ratio_to_plot.append(TP_ratio)
				TN_ratio_to_plot.append(TN_ratio)

			mid_line_slope_TP = (TP_ratio_to_plot[-1]-TP_ratio_to_plot[-2])/(num_features_all[-1]-num_features_all[-2])
			mid_line_slope_TN = (TN_ratio_to_plot[-1]-TN_ratio_to_plot[-2])/(num_features_all[-1]-num_features_all[-2])
			left_line_TP = TP_ratio_to_plot[:-1]
			left_line_TP.append((breaks[0]-num_features_all[-2])*mid_line_slope_TP + TP_ratio_to_plot[-2])
			left_line_TN = TN_ratio_to_plot[:-1]
			left_line_TN.append((breaks[0]-num_features_all[-2])*mid_line_slope_TN + TN_ratio_to_plot[-2])
			left_line_x = num_features_all[:-1]
			left_line_x.append(breaks[0])
			right_line_TP = []
			right_line_TP.append(TP_ratio_to_plot[-1] - (num_features_all[-1]-breaks[1])*mid_line_slope_TP)
			right_line_TP.append(TP_ratio_to_plot[-1])
			right_line_TN = []
			right_line_TN.append(TN_ratio_to_plot[-1] - (num_features_all[-1]-breaks[1])*mid_line_slope_TN)
			right_line_TN.append(TN_ratio_to_plot[-1])
			right_line_x = []
			right_line_x.append(breaks[1])
			right_line_x.append(num_features_all[-1])
			ax1.plot(left_line_x,left_line_TP,color=colors_to_use[i])
			ax2.plot(right_line_x,right_line_TP,color=colors_to_use[i])
			ax1.plot(left_line_x,left_line_TN,color=colors_to_use[i],linestyle='--')
			ax2.plot(right_line_x,right_line_TN,color=colors_to_use[i],linestyle='--')
			i += 1
		ax1.spines['right'].set_visible(False)
		ax2.spines['left'].set_visible(False)
		ax1.yaxis.tick_left()
		ax1.tick_params(labelright='off')
		ax2.yaxis.tick_right()
		d = 0.015
		kwargs = dict(transform=ax1.transAxes,color='k',clip_on=False)
		ax1.plot((1-d,1+d),(-d,+d),**kwargs)
		ax1.plot((1-d,1+d),(1-d,1+d),**kwargs)
		kwargs.update(transform=ax2.transAxes)
		ax2.plot((-d,+d),(1-d,1+d),**kwargs)
		ax2.plot((-d,+d),(-d,+d),**kwargs)

		ax1.set_xticks([1,2,3,4,5,10])
		ax2.set_xticks([500])
		ax1.grid(linestyle='--')
		ax2.grid(linestyle='--')

		fig.text(0.5,0.03,'Number of Features Used',ha='center',fontweight='bold',fontsize=12)
		fig.text(0.04,0.5,'Ratio',fontweight='bold',fontsize=12,va='center',rotation='vertical')
		fig.text(0.5,0.92,'Performance for Lead %s'%(Lead[lead_cut:]),fontweight='bold',fontsize=14,ha='center')
		legend2 = plt.legend(handles=[Line2D([0],[0],color='k'),Line2D([0],[0],color='k',linestyle='--')],labels=['TPR','TNR'],title='Line Types',loc='lower left',bbox_to_anchor=(1.01,0))
		plt.legend(CLS_custom_legend,CLS_method_names,title='Type of Classification',loc='center left',bbox_to_anchor=(1.01,0.6))
		plt.gca().add_artist(legend2)
		if save == True:
			plt.savefig('All_Figures/Classification_Features-vs-TPR_%s_%s.svg'%(Lead,date),bbox_inches='tight')
		plt.show()



### REGRESSION: TIME VS R2 ###
def REGRESSION__Time_vs_R2():
	num_features = 500
	for Lead in Leads:
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		Years = 99-int(Lead[1:])
		i=0
		for method in REG_methods:
			R2s = []
			century_solution = []
			century_predictions = []
			for year in range(Years):
				current_year_solution = []
				current_year_prediction = []
				
				for sc in scenarios:
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,num_features,sol_type,'R30',method),index_col=0)
					results.columns = ['Solution','Predictions']
					current_year_solution.append(results.loc[year,'Solution'])
					current_year_prediction.append(results.loc[year,'Predictions'])
					century_solution.append(results.loc[year,'Solution'])
					century_predictions.append(results.loc[year,'Predictions'])
				x = get_r2_python(current_year_solution,current_year_prediction)
				R2s.append(x)

			Year_labels = list(range((2100-Years),2100))
			plt.plot(Year_labels,R2s,color=colors_to_use[i])
			i += 1

		plt.xlabel('Years',fontweight='bold',fontsize=12)
		plt.ylabel('R-squared',fontweight='bold',fontsize=12)
		plt.title('Performance for Lead %s'%(Lead[lead_cut:]),fontweight='bold',fontsize=14)
		plt.grid(linestyle='--')
		plt.legend(handles=REG_custom_legend,labels=REG_method_names,title='Type of Regression',loc='center left',bbox_to_anchor=(1.01,0.5))
		if save == True:
			plt.savefig('All_Figures/Regression_Time_vs_R2_%s_%s.svg'%(Lead,date),bbox_inches='tight')
		plt.show()



### REGRESSION: TIME VS RMSE ###
def REGRESSION__Time_vs_RMSE():
	num_features = 500
	for Lead in Leads:
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		Years = 99-int(Lead[1:])
		i=0
		for method in REG_methods:
			RMSEs = []
			for year in range(Years):
				current_year_solution = []
				current_year_prediction = []
				
				for sc in scenarios:
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,num_features,sol_type,'R30',method),index_col=0)
					results.columns = ['Solution','Predictions']
					current_year_solution.append(results.loc[year,'Solution'])
					current_year_prediction.append(results.loc[year,'Predictions'])

				x = (mean_squared_error(current_year_solution,current_year_prediction))**(0.5)
				RMSEs.append(x)
			Year_labels = list(range((2100-Years),2100))
			plt.plot(Year_labels,RMSEs,color=colors_to_use[i])
			i += 1
		plt.xlabel('Years',fontweight='bold',fontsize=12)
		plt.ylabel('RMSE',fontweight='bold',fontsize=12)
		plt.title('Performance for Lead %s'%(Lead[lead_cut:]),fontweight='bold',fontsize=14)
		plt.grid(linestyle='--')
		plt.legend(handles=REG_custom_legend,labels=REG_method_names,title='Type of Regression',loc='center left',bbox_to_anchor=(1.01,0.5))
		plt.ylim(0,1)
		if save == True:
			plt.savefig('All_Figures/Regression_Time_vs_RMSE_%s_%s.svg'%(Lead,date),bbox_inches='tight')
		plt.show()



### CLASSIFICATION: TIME VS TPR ###
def CLASSIFICATION__Time_vs_TPR():
	threshold = 0.76
	num_features = 500
	CLS_custom_legend.append(Line2D([0],[0],color='k',linewidth=2))
	CLS_method_names.append('Ratio of Possible Positives')

	for Lead in Leads:
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		Years = 99-int(Lead[1:])
		i=0
		plt.grid(linestyle='--')
		for method in CLS_methods:
			TPRs = []
			TNRs = []
			Total_Poss = []
			for year in range(Years):
				current_year_solution = []
				current_year_prediction = []
				TN = 0
				FP = 0
				TP = 0
				FN = 0
				for sc in scenarios:
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
					results.columns = ['Solution','Predictions']
					current_year_solution.append(results.loc[year,'Solution'])
					current_year_prediction.append(results.loc[year,'Predictions'])
				for row in range(0,len(current_year_prediction)): # JUST STARTED WORKING ON THIS SECTION
						if current_year_solution[row] == True and current_year_prediction[row] == True: # both are above the threshold (not vulnerable)
							TN += 1
						elif current_year_solution[row] == True and current_year_prediction[row] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							FP += 1
						elif current_year_solution[row] == False and current_year_prediction[row] == False: # both are below the threshold (vulnerable)
							TP += 1
						elif current_year_solution[row] == False and current_year_prediction[row] == True: # solution is below threshold but prediction is above the threshold
							FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
				try:
					TP_ratio = TP/(TP+FN)
				except:
					TP_ratio = 0
				try:
					TN_ratio = TN/(TN+FP)
				except:
					TN_ratio = 0
				Total_Poss.append((TP+FN)/(TN+FP+TP+FN))
				TPRs.append(TP_ratio)
				TNRs.append(TN_ratio)

			Year_labels = list(range((2100-Years),2100))
			plt.plot(Year_labels,TPRs,color=colors_to_use[i])
			plt.plot(Year_labels,TNRs,color=colors_to_use[i],linestyle='--')
			plt.plot(Year_labels,Total_Poss,color='k',linewidth=2)
			i += 1
		plt.xlabel('Years',fontweight='bold',fontsize=12)
		plt.ylabel('Ratio',fontweight='bold',fontsize=12)
		plt.title('Performance for Lead %s with Threshold of %s'%(Lead[lead_cut:],str(threshold)),fontweight='bold',fontsize=14)
		legend2 = plt.legend(handles=[Line2D([0],[0],color='k'),Line2D([0],[0],color='k',linestyle='--')],labels=['TPR','TNR'],title='Line Types',loc='lower left',bbox_to_anchor=(1.01,0.1))
		plt.legend(handles=CLS_custom_legend,labels=CLS_method_names,title='Type of Classification',loc='center left',bbox_to_anchor=(1.01,0.6))
		plt.gca().add_artist(legend2)
		plt.xlim(2000,2100)
		if save == True:
			plt.savefig('All_Figures/Classification_Time_vs_TPR_%s_threshold-%s_%s.svg'%(Lead,threshold,date),bbox_inches='tight')
		plt.show()



def REGRESSION__Time_vs_Diff(date,save=False):
	method = 'rdm_for'
	for Lead in Leads:
		fig = plt.figure(figsize=(10,5))
		fig.patch.set_visible(False)
		ax = fig.add_subplot(111)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)
		ax1.grid(linestyle='--')
		ax2.grid(linestyle='--')
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		group_26 = pd.DataFrame()
		group_45 = pd.DataFrame()
		group_60 = pd.DataFrame()
		group_85 = pd.DataFrame()
		for sc in scenarios:
			results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,500,sol_type,'R30',method),index_col=0)
			results.columns = ['Solution','Predictions']
			diff = results['Predictions'] - results['Solution']
			if 'rcp26' in sc:
				c = '#ffb3b3'
				group_26[sc] = diff
			elif 'rcp45' in sc:
				c = '#ff3333'
				group_45[sc] = diff
			elif 'rcp60' in sc:
				c = '#b30000'
				group_60[sc] = diff
			elif 'rcp85' in sc:
				c = '#330000'
				group_85[sc] = diff
			ax1.plot(diff, color=c)
		group_26_avg = group_26.mean(axis=1)
		group_45_avg = group_45.mean(axis=1)
		group_60_avg = group_60.mean(axis=1)
		group_85_avg = group_85.mean(axis=1)
		ax2.plot(group_26_avg,color='#ffb3b3',linewidth=2)
		ax2.plot(group_45_avg,color='#ff3333',linewidth=2)
		ax2.plot(group_60_avg,color='#b30000',linewidth=2)
		ax2.plot(group_85_avg,color='#330000',linewidth=2)
		group_26_std = group_26.std(axis=1)
		group_45_std = group_45.std(axis=1)
		group_60_std = group_60.std(axis=1)
		group_85_std = group_85.std(axis=1)
		x_values = range(len(group_26_avg))
		ax2.fill_between(x_values, group_26_avg-group_26_std, group_26_avg+group_26_std, alpha=0.3,facecolor='#ffb3b3')
		ax2.fill_between(x_values, group_45_avg-group_45_std, group_45_avg+group_45_std, alpha=0.3,facecolor='#ff3333')
		ax2.fill_between(x_values, group_60_avg-group_60_std, group_60_avg+group_60_std, alpha=0.3,facecolor='#b30000')
		ax2.fill_between(x_values, group_85_avg-group_85_std, group_85_avg+group_85_std, alpha=0.3,facecolor='#330000')
		if Lead[lead_cut:] == '0':
			for axes in [ax1,ax2]:
				axes.set_xticks([0,24,49,74,98])
				axes.set_xticklabels([2001,2025,2050,2075,2098])
		elif Lead[lead_cut:] == '1':
			for axes in [ax1,ax2]:
				axes.set_xticks([0,23,48,73,97])
				axes.set_xticklabels([2002,2025,2050,2075,2097])
		elif Lead[lead_cut:] == '5':
			for axes in [ax1,ax2]:
				axes.set_xticks([0,20,45,70,93])
				axes.set_xticklabels([2006,2025,2050,2075,2093])
		elif Lead[lead_cut:] == '10':
			for axes in [ax1,ax2]:
				axes.set_xticks([0,14,39,64,88])
				axes.set_xticklabels([2011,2025,2050,2075,2088])
		elif Lead[lead_cut:] == '20':
			for axes in [ax1,ax2]:
				axes.set_xticks([0,4,29,54,78])
				axes.set_xticklabels([2021,2025,2050,2075,2078])
		else:
			print('Problem with identifying the Lead for X axis labels.')
		ax.set_yticks([])
		ax.set_xticks([])
		ax.set_xlabel('Years',fontweight='bold',fontsize=12,labelpad=20)
		ax.set_ylabel('Predictions minus Solutions',fontweight='bold',fontsize=12,labelpad=40)
		ax.set_title('Random Forest Regression for Lead %s'%(Lead[lead_cut:]),fontweight='bold',fontsize=14)
		ax1.legend([Line2D([0],[0],c='#ffb3b3'),Line2D([0],[0],c='#ff3333'),Line2D([0],[0],c='#b30000'),Line2D([0],[0],c='#330000')],['RCP 2.6','RCP 4.5','RCP 6.0','RCP 8.5'])
		if save == True:
			fig.savefig('All_Figures/Regression_Time_vs_Diff-RCPs_%s_method-%s_SUBPLOTS_%s.svg'%(Lead,method,date))#,bbox_inches='tight')
		plt.show()



### REGRESSION: TIME VS AVG DIFF ###
def REGRESSION__Time_vs_AvgDiff(date,save=False):
	for Lead in Leads:
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		agg_diff = pd.DataFrame()
		i=0
		for method in REG_methods:
			for sc in scenarios:
				results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,500,sol_type,'R30',method),index_col=0)
				results.columns = ['Solution','Predictions']
				diff = results['Predictions'] - results['Solution']
				agg_diff[sc] = diff
			agg_diff_avg = agg_diff.mean(axis=1)
			plt.plot(agg_diff_avg,c=colors_to_use[i])
			i += 1
		plt.xlabel('Years',fontweight='bold',fontsize=12)
		plt.ylabel('Predictions minus Solutions',fontweight='bold',fontsize=12)
		plt.title('Difference between Predictions and Solutions for Lead %s'%(Lead[lead_cut:]),fontweight='bold',fontsize=14)
		plt.grid(linestyle='--')
		plt.legend(handles=REG_custom_legend,labels=REG_method_names,title='Type of Regression',loc='center left',bbox_to_anchor=(1.01,0.5))
		if save == True:
			plt.savefig('All_Figures/Regression_Time_vs_AvgDiff_%s_%s.svg'%(Lead,date),bbox_inches='tight')
		plt.show()



### REGRESSION: LEADS VS FEATURES FOR R2 ABOVE ###
def REGRESSION__Leads_vs_Features_for_R2aboveX(date,save=False):
	R2_above = 0.25
	i=0
	for method in REG_methods:
		for num_features in num_features_all:
			for Lead in Leads:
				if Lead[1] == '0':
					lead_cut = 2
				else:
					lead_cut = 1
				Years = 99-int(Lead[1:])
				for year in range(Years):
					current_year_solution = []
					current_year_prediction = []
					for sc in scenarios:
						results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s.csv'%(sc,'REG',Lead,num_features,sol_type,'R30',method),index_col=0)
						results.columns = ['Solution','Predictions']
						current_year_solution.append(results.loc[year,'Solution'])
						current_year_prediction.append(results.loc[year,'Predictions'])
					x = get_r2_python(current_year_solution,current_year_prediction)
					if x >= R2_above:
						color_value = 2099-Years+year
						break
				plt.scatter(Lead,num_features,c=color_value,edgecolor='k',cmap='rainbow',marker='o',s=60,vmin=2000,vmax=2100)
		plt.title('Average Year R2 Exceeded 0.25 for %s'%(method))	
		cbar = plt.colorbar()	
		plt.ylim(0,11)
		i += 1
		if save == True:
			plt.savefig('All_Figures/Regression_Leads_vs_Features_for_%s_R2above_%s_%s.svg'%(method,R2_above,date))
		plt.show()



### CLASSIFICATION: LEADS VS FEATURES TPR ABOVE ###
def CLASSIFICATION__Leads_vs_Features_for_TPRaboveX(date,save=False):
	TPR_above = 0.8
	threshold = 0.76
	i=0
	for method in CLS_methods:
		for num_features in num_features_all:
			for Lead in Leads:
				if Lead[1] == '0':
					lead_cut = 2
				else:
					lead_cut = 1
				Years = 99-int(Lead[1:])
				for year in range(Years):
					current_year_solution = []
					current_year_prediction = []
					TN = 0
					FP = 0
					TP = 0
					FN = 0
					for sc in scenarios:
						results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
						results.columns = ['Solution','Predictions']
						current_year_solution.append(results.loc[year,'Solution'])
						current_year_prediction.append(results.loc[year,'Predictions'])
					for row in range(0,len(current_year_prediction)):
						if current_year_solution[row] == True and current_year_prediction[row] == True: # both are above the threshold (not vulnerable)
							TN += 1
						elif current_year_solution[row] == True and current_year_prediction[row] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							FP += 1
						elif current_year_solution[row] == False and current_year_prediction[row] == False: # both are below the threshold (vulnerable)
							TP += 1
						elif current_year_solution[row] == False and current_year_prediction[row] == True: # solution is below threshold but prediction is above the threshold
							FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
					try:
						TP_ratio = TP/(TP+FN)
					except:
						TP_ratio = 0
					try:
						TN_ratio = TN/(TN+FP)
					except:
						TN_ratio = 0
					if TP_ratio >= TPR_above:
						color_value = 2099-Years+year
						break
				plt.scatter(Lead,num_features,c=color_value,edgecolor='k',cmap='rainbow',marker='o',s=60,vmin=2050,vmax=2070)
		plt.title('Average Year R2 Exceeded %s for %s'%(TPR_above,method))	
		cbar = plt.colorbar()	
		plt.ylim(0,11)
		i += 1
		if save == True:
			plt.savefig('All_Figures/Classification_Leads_vs_Features_for_%s_threshold-%s_R2above_%s_%s.svg'%(method,threshold,TPR_above,date))
		plt.show()



### CLASSIFICATION: LEADS VS TP & TN ###
def CLASSIFICATION__Leads_vs_TP_TN(date,save=False):
	i=0
	threshold = 0.76
	plt.grid(linestyle='--')
	num_features = 500
	for method in CLS_methods:
		TPRs = []
		TNRs = []
		Total_Poss = []
		for Lead in Leads:
			Years = 99-int(Lead[1:])
			TN = 0
			TP = 0
			FN = 0
			FP = 0
			for sc in scenarios:
				results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
				results.columns = ['Solution','Predictions']
				for row in range(0,len(results['Solution'])):
						if results.loc[row,'Solution'] == True and results.loc[row,'Predictions'] == True: # both are above the threshold (not vulnerable)
							TN += 1
						elif results.loc[row,'Solution'] == True and results.loc[row,'Predictions'] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							FP += 1
						elif results.loc[row,'Solution'] == False and results.loc[row,'Predictions'] == False: # both are below the threshold (vulnerable)
							TP += 1
						elif results.loc[row,'Solution'] == False and results.loc[row,'Predictions'] == True: # solution is below threshold but prediction is above the threshold
							FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
			try:
				TP_ratio = TP/(TP+FN)
			except:
				TP_ratio = 0
			try:
				TN_ratio = TN/(TN+FP)
			except:
				TN_ratio = 0
			Total_Poss.append((TP+FN)/(TN+FP+TP+FN))
			TPRs.append(TP_ratio)
			TNRs.append(TN_ratio)
		plt.plot(Leads_to_plot,TPRs,color=colors_to_use[i])
		plt.plot(Leads_to_plot,TNRs,color=colors_to_use[i],linestyle='--')
		i += 1
	plt.xlabel('Lead Times (years)',fontweight='bold',fontsize=12)
	plt.ylabel('Ratio',fontweight='bold',fontsize=12)
	plt.xticks(Leads_to_plot,fontsize=12)
	legend2 = plt.legend(handles=[Line2D([0],[0],color='k'),Line2D([0],[0],color='k',linestyle='--')],labels=['TPR','TNR'],title='Line Types',loc='lower left',bbox_to_anchor=(1.01,-0.02))
	plt.legend(handles=CLS_custom_legend,labels=CLS_method_names,title='Type of Classification',loc='center left',bbox_to_anchor=(1.01,0.6))
	plt.gca().add_artist(legend2)
	if save == True:
		plt.savefig('All_Figures/Classification_Leads-vs-TP-TN_%s.svg'%(date),bbox_inches='tight')
	plt.show()



### THE MAP ###
def MAP(date,save=False):
	# create map background 
	from mpl_toolkits.basemap import Basemap
	m = Basemap(llcrnrlon=-123, llcrnrlat=36.5, urcrnrlon=-118, urcrnrlat=41.1, projection='cyl', resolution='f', area_thresh=1) # whole US is 24-53 and -125.6--60
	m.drawmapboundary(fill_color='steelblue', zorder=-99)
	m.arcgisimage(service='World_Physical_Map', xpixels=1000, dpi=1000, verbose=False) # World_Shaded_Relief (original), World_Terrain_Base (not much color), World_Physical_Map (I like this one best so far),
	m.drawstates(color='gray')
	m.drawcountries(color='k')
	m.drawcoastlines(color='gray')
	m.drawrivers(linewidth=0.5, linestyle='solid', color='b')

	# load reservoir data and scatterplot (lat,lon,dams)
	df = pd.read_csv('sites_list.csv')
	lons = df.iloc[:,3].values
	lats = df.iloc[:,2].values
	data = df.iloc[:,5]

	markers = []
	for i in data:
		if i == 'all':
			markers.append('*')
		elif i == 'swe':
			markers.append('X')
		elif i == 'cmip':
			markers.append('p')
		else:
			markers.append('o')
	
	# Folsom Dam
	# m.scatter(-121.1565,38.7077,s=130,c='k',marker='v',zorder=6) # Folsom Dam
	m.scatter(-121.1565-.05,38.7077+.05,s=35,c='b', marker='o',edgecolor='None',zorder=6)
	m.scatter(-121.1565+.05,38.7077+.05,s=35,c='white', marker='o',edgecolor='None',zorder=6)
	m.scatter(-121.1565,38.7077-.05,s=35,c='r', marker='o',edgecolor='None',zorder=6)
	plt.text(-121.1565+.15,38.7077-.05,'Folsom Dam', color='k',weight='semibold',size=8)
	
	# Shasta Dam
	# m.scatter(-122.418889,40.718611,s=130,c='k',marker='v',zorder=6) # Shasta Dam
	m.scatter(-122.418889-.05,40.718611,s=35,c='b', marker='o',edgecolor='None',zorder=6)
	m.scatter(-122.418889+.05,40.718611,s=35,c='r', marker='o',edgecolor='None',zorder=6)
	plt.text(-122.418889+.15,40.718611-.05,'Shasta Dam', color='k',weight='semibold',size=8)
	
	# Oroville Dam
	# m.scatter(-121.485556,39.538889,s=130,c='k',marker='v',zorder=6) # Oroville Dam
	m.scatter(-121.485556-.05,39.538889+.05,s=35,c='b', marker='o',edgecolor='None',zorder=6)
	m.scatter(-121.485556+.05,39.538889+.05,s=35,c='white', marker='o',edgecolor='None',zorder=6)
	m.scatter(-121.485556,39.538889-.05,s=35,c='r', marker='o',edgecolor='None',zorder=6)
	plt.text(-121.485556+.15,39.538889-.05,'Oroville Dam', color='k',weight='semibold',size=8)

	# white can be snowpack
	# blue can be streamflow
	# temperature and precipitation can be red

	# 0 La Grange Dam
	m.scatter(lons[0],lats[0],s=35,c='b', marker='o',edgecolor='None',zorder=6)
	plt.text(lons[0]+.15, lats[0]-.05, df.iloc[0,1], color='k', weight='semibold',size=8)

	# 1 Merced R near Merced Falls
	m.scatter(lons[1],lats[1],s=35,c='b', marker='o',edgecolor='None',zorder=6)
	plt.text(lons[1]+.15, lats[1]-.05, df.iloc[1,1], color='k', weight='semibold',size=8)
	
	# 2 Friant Dam
	m.scatter(lons[2],lats[2],s=35,c='b', marker='o',edgecolor='None',zorder=6)
	plt.text(lons[2]+.15, lats[2]-.05, df.iloc[2,1], color='k', weight='semibold',size=8)
	
	# 3 New Melones Reservoir
	m.scatter(lons[3],lats[3],s=35,c='b', marker='o',edgecolor='None',zorder=6)
	plt.text(lons[3]+.15, lats[3]-.05, df.iloc[3,1], color='k', weight='semibold',size=8)

	# 4 Mokelumne Hill
	m.scatter(lons[4],lats[4],s=35,c='b', marker='o',edgecolor='None',zorder=6)
	plt.text(lons[4]+.15, lats[4]-.05, df.iloc[4,1], color='k', weight='semibold',size=8)

	# 5 Sacramento R at Bend Bridge
	m.scatter(lons[5]-.05,lats[5],s=35,c='b', marker='o',edgecolor='None',zorder=6)
	m.scatter(lons[5]+.05,lats[5],s=35,c='white', marker='o',edgecolor='None',zorder=6)
	plt.text(lons[5]+.15, lats[5]-.05, df.iloc[5,1], color='k', weight='semibold',size=8)

	# 6 New Hogan Lake
	m.scatter(lons[6],lats[6],s=35,c='b', marker='o',edgecolor='None',zorder=6)
	plt.text(lons[6]+.15, lats[6]-.05, df.iloc[6,1], color='k', weight='semibold',size=8)

	# 7 Yuba R near Smartville
	m.scatter(lons[7]-.05,lats[7],s=35,c='b', marker='o',edgecolor='None',zorder=6)
	m.scatter(lons[7]+.05,lats[7],s=35,c='white', marker='o',edgecolor='None',zorder=6)
	plt.text(lons[7]+.15, lats[7]-.05, df.iloc[7,1], color='k', weight='semibold',size=8)

	bbox_props = dict(boxstyle='round', fc='lightgray', alpha=0.9, pad=1)
	a = m.scatter(-122.6, 37.6, s=35, c='b', marker='o', edgecolor='None', zorder=7)
	b = m.scatter(-122.6, 37.3, s=35, c='white', marker='o', edgecolor='None', zorder=7)
	b = m.scatter(-122.6, 37, s=35, c='r', marker='o', edgecolor='None', zorder=7)
	plt.text(-122.7,37.38,'Data Types Available\n\n     Streamflow\n\n     Snow\n\n     Precipitation\n     and Temperature', weight='semibold',size=8, va='center',ha='left', bbox=bbox_props)
	if save == True:
		plt.savefig('All_Figures/Map_%s.svg'%(date),bbox_inches='tight')
	plt.show()



### CLASSIFICATION: LEADS VS SWITCHES ###
def CLASSIFICATION__Leads_vs_V_NV_switch(date,save=False):
	i=0
	threshold = 0.76
	plt.grid(linestyle='--')
	num_features = 500
	for method in CLS_methods:
		TPRs = []
		TNRs = []
		Total_Poss = []
		for Lead in Leads[1::]:
			Years = 99-int(Lead[1:])
			TN = 0
			TP = 0
			FN = 0
			FP = 0
			switch_count_v_to_nv = 0
			switch_count_nv_to_v = 0
			for sc in scenarios:
				results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
				results.columns = ['Solution','Predictions']
				for row in range(int(Lead[1:]),len(results['Solution'])):
					if results.loc[row,'Solution'] == True and results.loc[row-int(Lead[1:]),'Solution'] == False:
						switch_count_v_to_nv += 1
						if results.loc[row,'Solution'] == True and results.loc[row,'Predictions'] == True: # both are above the threshold (not vulnerable)
							TN += 1
						elif results.loc[row,'Solution'] == True and results.loc[row,'Predictions'] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							FP += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))	
					if results.loc[row,'Solution'] == False and results.loc[row-int(Lead[1:]),'Solution'] == True:
						switch_count_nv_to_v += 1
						if results.loc[row,'Solution'] == False and results.loc[row,'Predictions'] == False: # both are below the threshold (vulnerable)
							TP += 1
						elif results.loc[row,'Solution'] == False and results.loc[row,'Predictions'] == True: # solution is below threshold but prediction is above the threshold
							FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
			print(switch_count_nv_to_v,switch_count_v_to_nv)
			try:
				TP_ratio = TP/(TP+FN)
			except:
				TP_ratio = 0
			try:
				TN_ratio = TN/(TN+FP)
			except:
				TN_ratio = 0
			Total_Poss.append((TP+FN)/(TN+FP+TP+FN))
			TPRs.append(TP_ratio)
			TNRs.append(TN_ratio)
			
		plt.plot(Leads_to_plot[1::],TPRs,color=colors_to_use[i])
		plt.plot(Leads_to_plot[1::],TNRs,color=colors_to_use[i],linestyle='--')
		i += 1
	plt.xlabel('Lead Times (years)',fontweight='bold',fontsize=12)
	plt.ylabel('Ratio',fontweight='bold',fontsize=12)
	plt.xticks(Leads_to_plot[1::],fontsize=12)
		
	legend2 = plt.legend(handles=[Line2D([0],[0],color='k'),Line2D([0],[0],color='k',linestyle='--')],labels=['TPR','TNR'],title='Line Types',loc='lower left',bbox_to_anchor=(1.01,-0.02))
	plt.legend(handles=CLS_custom_legend,labels=CLS_method_names,title='Type of Classification',loc='center left',bbox_to_anchor=(1.01,0.6))
	plt.gca().add_artist(legend2)
	if save == True:
		plt.savefig('All_Figures/Classification_Leads-vs-V-NV-switch_%s.svg'%(date),bbox_inches='tight')
	plt.show()	



### CLASSIFICATION: TIME VS TPR WITH RCPS ###
def CLASSIFICATION__Time_vs_TPR_RCP(date,save=False):
	threshold = 0.76
	num_features = 500
	CLS_custom_legend.append(Line2D([0],[0],color='k',linewidth=2))
	CLS_method_names.append('Ratio of Possible Positives')

	for Lead in Leads:
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		Years = 99-int(Lead[1:])
		print(Years)
		i=0
		plt.grid(linestyle='--')
		Total_Poss = []
		method = 'rdm_for'
		for sc in scenarios:
				if 'rcp26' in sc:
					c = '#ffb3b3'
				elif 'rcp45' in sc:
					c = '#ff3333'
				elif 'rcp60' in sc:
					c = '#b30000'
				elif 'rcp85' in sc:
					c = '#330000'
				else:
					print('Cannot identify RCP')
				TPRs = []
				TNRs = []
				current_sc_solution = []
				current_sc_prediction = []
				TN = 0
				FP = 0
				TP = 0
				FN = 0
				TNs = []
				FPs = []
				TPs = []
				FNs = []
				for year in range(Years):
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
					results.columns = ['Solution','Predictions']
					if results.loc[year,'Solution'] == True and results.loc[year,'Predictions'] == True: # both are above the threshold (not vulnerable)
						TN += 1
					elif results.loc[year,'Solution'] == True and results.loc[year,'Predictions'] == False: # solution is above the threshold but prediction is below the threshold (false positive)
						FP += 1
					elif results.loc[year,'Solution'] == False and results.loc[year,'Predictions'] == False: # both are below the threshold (vulnerable)
						TP += 1
					elif results.loc[year,'Solution'] == False and results.loc[year,'Predictions'] == True: # solution is below threshold but prediction is above the threshold
						FN += 1
					else:
						print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
					try:
						TP_ratio = TP/(TP+FN)
					except:
						TP_ratio = 0
					try:
						TN_ratio = TN/(TN+FP)
					except:
						TN_ratio = 0
					Total_Poss.append((TP+FN)/(TN+FP+TP+FN))
					TPRs.append(TP_ratio)
					TNRs.append(TN_ratio)
				Year_labels = list(range((2100-Years),2100))
				plt.plot(Year_labels,TPRs,color=c)
				plt.plot(Year_labels,TNRs,color=c,linestyle='--')
				i += 1
		plt.xlabel('Years',fontweight='bold',fontsize=12)
		plt.ylabel('Ratio',fontweight='bold',fontsize=12)
		plt.title('Performance for Lead %s with Threshold of %s'%(Lead[lead_cut:],str(threshold)),fontweight='bold',fontsize=14)
		
		legend2 = plt.legend(handles=[Line2D([0],[0],color='k'),Line2D([0],[0],color='k',linestyle='--')],labels=['TPR','TNR'],title='Line Types',loc='lower left',bbox_to_anchor=(1.01,-0.02))
		plt.legend(handles=CLS_custom_legend,labels=CLS_method_names,title='Type of Classification',loc='center left',bbox_to_anchor=(1.01,0.6))
		plt.gca().add_artist(legend2)
		plt.xlim(2000,2100)
		if save == True:
			plt.savefig('All_Figures/Classification_Time_vs_TPR_%s_threshold-%s_%s.svg'%(Lead,threshold,date),bbox_inches='tight')
		plt.show()



### CLASSIFICATION: THRESHOLDS VS TP & TN (SUBPLOTS) ###
def CLASSIFICATION__Thresholds_vs_TP_TN_subplots(date,save=False):
	num_features = 500

	CLS_custom_legend.append(Line2D([0],[0],color='k'))
	CLS_custom_legend.append(Line2D([0],[0],color='k',linestyle='--'))
	CLS_custom_legend.append(Line2D([0],[0],color='w',marker=6,markersize=8,markerfacecolor='k'))
	CLS_custom_legend.append(Line2D([0],[0],color='w',marker=7,ms=8,markerfacecolor='k'))
	CLS_method_names.append('Benchmark')
	CLS_method_names.append('Possible Classifications')
	CLS_method_names.append('10th Percentile')
	CLS_method_names.append('90th Percentile')
	
	for Lead in Leads:
		fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,sharex=True,figsize=(10,5),gridspec_kw={'hspace':0,'wspace':0.02})
		i=0
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		for method in CLS_methods:
			TP_ratio_to_plot = []
			TN_ratio_to_plot = []
			TP_min_CI = []
			TP_max_CI = []
			TN_min_CI = []
			TN_max_CI = []
			Possible_Positives_Ratio = []
			Possible_Negatives_Ratio = []
			Positives_Benchmark = []
			Negatives_Benchmark = []
			for threshold in thresholds_all:
				TN = 0
				FP = 0
				TP = 0
				FN = 0
				Scenario_TP_ratio = []
				Scenario_TN_ratio = []
				for sc in scenarios:
					Scenario_FN = 0
					Scenario_TP = 0
					Scenario_TN = 0
					Scenario_FP = 0
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
					results.columns = ['Solution','Predictions']
					for row in range(0,len(results['Solution'])):
						if results['Solution'][row] == True and results['Predictions'][row] == True: # both are above the threshold (not vulnerable)
							TN += 1
							Scenario_TN += 1
						elif results['Solution'][row] == True and results['Predictions'][row] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							FP += 1
							Scenario_FP += 1
						elif results['Solution'][row] == False and results['Predictions'][row] == False: # both are below the threshold (vulnerable)
							TP += 1
							Scenario_TP += 1
						elif results['Solution'][row] == False and results['Predictions'][row] == True: # solution is below threshold but prediction is above the threshold
							FN += 1
							Scenario_FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
					try:
						Scenario_TP_ratio.append(Scenario_TP/(Scenario_TP+Scenario_FN))
					except:
						Scenario_TP_ratio.append(0)
					try:
						Scenario_TN_ratio.append(Scenario_TN/(Scenario_TN+Scenario_FP))
					except:
						Scenario_TN_ratio.append(0)
				TP_ratio = TP/(TP+FN)
				TN_ratio = TN/(TN+FP)

				TP_ratio_to_plot.append(np.median(Scenario_TP_ratio))
				TN_ratio_to_plot.append(np.median(Scenario_TN_ratio))
				Possible_Positives_Ratio.append((TP+FN)/(TP+FN+FP+TN))
				Possible_Negatives_Ratio.append((TN+FP)/(TP+FN+FP+TN))
				Positives_Benchmark.append(((TP+FN)/(TP+FN+FP+TN))**2)
				Negatives_Benchmark.append(((TN+FP)/(TP+FN+FP+TN))**2)

				TP_min_CI.append(np.percentile(Scenario_TP_ratio,10))
				TP_max_CI.append(np.percentile(Scenario_TP_ratio,90))
				TN_min_CI.append(np.percentile(Scenario_TN_ratio,10))
				TN_max_CI.append(np.percentile(Scenario_TN_ratio,90))

			ax1.plot(thresholds_all,TP_ratio_to_plot,c=colors_to_use[i])
			ax2.plot(thresholds_all,TN_ratio_to_plot,c=colors_to_use[i])
			ax1.scatter(thresholds_all,TP_min_CI,c=colors_to_use[i],marker=6)
			ax1.scatter(thresholds_all,TP_max_CI,c=colors_to_use[i],marker=7)
			ax2.scatter(thresholds_all,TN_min_CI,c=colors_to_use[i],marker=6)
			ax2.scatter(thresholds_all,TN_max_CI,c=colors_to_use[i],marker=7)
			i += 1
		ax1.plot(thresholds_all,Positives_Benchmark,color='k')
		ax2.plot(thresholds_all,Negatives_Benchmark,color='k')
		ax1.plot(thresholds_all,Possible_Positives_Ratio,color='k',linestyle='--')
		ax2.plot(thresholds_all,Possible_Negatives_Ratio,color='k',linestyle='--')

		fig.text(0.5,0.03,'Thresholds',ha='center',fontweight='bold',fontsize=12)
		fig.text(0.07,0.5,'Ratio',fontweight='bold',fontsize=12,va='center',rotation='vertical')
		fig.text(0.5,0.9,'TP/TN Ratios, Lead Time = %s Years'%(Lead[lead_cut:]),fontweight='bold',fontsize=14,ha='center')
		fig.text(0.28,0.2,'A) True Positive Ratios',fontweight='bold',fontsize=12)
		fig.text(0.55,0.2,'B) True Negative Ratios',fontweight='bold',fontsize=12)
		thresholds_every_other = [0.6,'',0.64,'',0.68,'',0.72,'',0.76,'',0.8,'',0.84]
		plt.xticks(thresholds_all,fontsize=10)
		ax1.set_xticklabels(thresholds_every_other,fontsize=10)
		ax2.set_xticklabels(thresholds_every_other,fontsize=10)
		ax1.grid(linestyle='--')
		ax2.grid(linestyle='--')
		legend2 = plt.legend(handles=[Line2D([0],[0],color='w',marker=6,markersize=8,markerfacecolor='k'),Line2D([0],[0],color='w',marker=7,ms=8,markerfacecolor='k')],labels=['10th Percentile','90th Percentile'],title='Markers',loc='lower left',bbox_to_anchor=(1.01,0.14))
		plt.legend(handles=CLS_custom_legend,labels=CLS_method_names,title='Legend',loc='center left',bbox_to_anchor=(1.01,0.5))
		if save == True:
			plt.savefig('All_Figures/Classification_Thresholds-vs-TP-TN_subplots_%s_%s.svg'%(Lead,date),bbox_inches='tight')
		plt.show()



### CLASSIFICATION: LEADS VS TP & TN (SUBPLOTS) ###
def CLASSIFICATION__Leads_vs_TP_TN_subplots(date,save=False):
	i=0
	threshold = 0.76
	plt.grid(linestyle='--')
	num_features = 500
	fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,sharex=True,figsize=(10,5),gridspec_kw={'hspace':0,'wspace':0.02})
	CLS_custom_legend.append(Line2D([0],[0],color='k'))
	CLS_custom_legend.append(Line2D([0],[0],color='w',marker=6,ms=8,markerfacecolor='k'))
	CLS_custom_legend.append(Line2D([0],[0],color='w',marker=7,ms=8,markerfacecolor='k'))
	CLS_method_names.append('Benchmark')
	CLS_method_names.append('10th Percentile')
	CLS_method_names.append('90th Percentile')
	for method in CLS_methods:
		TPRs = []
		TNRs = []
		TPRs_switch = []
		TNRs_switch = []
		TP_min_CI = []
		TP_max_CI = []
		TN_min_CI = []
		TN_max_CI = []
		Possible_Positives_Ratio = []
		Possible_Negatives_Ratio = []
		Positives_Benchmark = []
		Negatives_Benchmark = []
		for Lead in Leads:
			Years = 99-int(Lead[1:])
			TN = 0
			TP = 0
			FN = 0
			FP = 0
			Scenario_TP_ratio = []
			Scenario_TN_ratio = []
			Scenario_TP_switch_ratio = []
			Scenario_TN_switch_ratio = []
			for sc in scenarios:
				Scenario_FN = 0
				Scenario_TP = 0
				Scenario_TN = 0
				Scenario_FP = 0
				TN_switch = 0
				TP_switch = 0
				FN_switch = 0
				FP_switch = 0
				results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
				results.columns = ['Solution','Predictions']

				for row in range(int(Lead[1:]),len(results['Solution'])):
						if results.loc[row,'Solution'] == True and results.loc[row,'Predictions'] == True: # both are above the threshold (not vulnerable)
							TN += 1
							Scenario_TN += 1
						elif results.loc[row,'Solution'] == True and results.loc[row,'Predictions'] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							FP += 1
							Scenario_FP += 1
						elif results.loc[row,'Solution'] == False and results.loc[row,'Predictions'] == False: # both are below the threshold (vulnerable)
							TP += 1
							Scenario_TP += 1
						elif results.loc[row,'Solution'] == False and results.loc[row,'Predictions'] == True: # solution is below threshold but prediction is above the threshold
							FN += 1
							Scenario_FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
						if results.loc[row,'Solution'] == True and results.loc[row-int(Lead[1:]),'Solution'] == False:
							if results.loc[row,'Solution'] == True and results.loc[row,'Predictions'] == True: # both are above the threshold (not vulnerable)
								TN_switch += 1
							elif results.loc[row,'Solution'] == True and results.loc[row,'Predictions'] == False: # solution is above the threshold but prediction is below the threshold (false positive)
								FP_switch += 1
							else:
								print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))	
						if results.loc[row,'Solution'] == False and results.loc[row-int(Lead[1:]),'Solution'] == True:
							if results.loc[row,'Solution'] == False and results.loc[row,'Predictions'] == False: # both are below the threshold (vulnerable)
								TP_switch += 1
							elif results.loc[row,'Solution'] == False and results.loc[row,'Predictions'] == True: # solution is below threshold but prediction is above the threshold
								FN_switch += 1
							else:
								print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
				try:
					Scenario_TP_ratio.append(Scenario_TP/(Scenario_TP+Scenario_FN))
				except:
					Scenario_TP_ratio.append(0)
				try:
					Scenario_TN_ratio.append(Scenario_TN/(Scenario_TN+Scenario_FP))
				except:
					Scenario_TN_ratio.append(0)
				try:
					Scenario_TP_switch_ratio.append(TP_switch/(TP_switch+FN_switch))
				except:
					Scenario_TP_switch_ratio.append(0)
				try:
					Scenario_TN_switch_ratio.append(TN_switch/(TN_switch+FP_switch))
				except:
					Scenario_TN_switch_ratio.append(0)
			try:
				TP_ratio = TP/(TP+FN)
			except:
				TP_ratio = 0
			try:
				TN_ratio = TN/(TN+FP)
			except:
				TN_ratio = 0

			TPRs.append(np.median(Scenario_TP_ratio))
			TNRs.append(np.median(Scenario_TN_ratio))
			TPRs_switch.append(np.median(Scenario_TP_switch_ratio))
			TNRs_switch.append(np.median(Scenario_TN_switch_ratio))

			Possible_Positives_Ratio.append((TP+FN)/(TP+FN+FP+TN))
			Possible_Negatives_Ratio.append((TN+FP)/(TP+FN+FP+TN))
			Positives_Benchmark.append(((TP+FN)/(TP+FN+FP+TN))**2)
			Negatives_Benchmark.append(((TN+FP)/(TP+FN+FP+TN))**2)

			TP_min_CI.append(np.percentile(Scenario_TP_ratio,10))
			TP_max_CI.append(np.percentile(Scenario_TP_ratio,90))
			TN_min_CI.append(np.percentile(Scenario_TN_ratio,10))
			TN_max_CI.append(np.percentile(Scenario_TN_ratio,90))
		ax1.plot(Leads_to_plot,TPRs,color=colors_to_use[i])
		ax2.plot(Leads_to_plot,TNRs,color=colors_to_use[i])
		ax1.scatter(Leads_to_plot,TP_min_CI,c=colors_to_use[i],marker=6)
		ax1.scatter(Leads_to_plot,TP_max_CI,c=colors_to_use[i],marker=7)
		ax2.scatter(Leads_to_plot,TN_min_CI,c=colors_to_use[i],marker=6)
		ax2.scatter(Leads_to_plot,TN_max_CI,c=colors_to_use[i],marker=7)
		i += 1

	ax1.plot(Leads_to_plot,Positives_Benchmark,color='k')
	ax2.plot(Leads_to_plot,Negatives_Benchmark,color='k')
	fig.text(0.5,0.03,'Lead Times (years)',ha='center',fontweight='bold',fontsize=12)
	fig.text(0.07,0.5,'Ratio',fontweight='bold',fontsize=12,va='center',rotation='vertical')
	fig.text(0.5,0.9,'TP/TN Ratios, Threshold = %s'%(threshold),fontweight='bold',fontsize=14,ha='center')
	fig.text(0.165,0.2,'A) True Positive Ratios',fontweight='bold',fontsize=12)
	fig.text(0.555,0.2,'B) True Negative Ratios',fontweight='bold',fontsize=12)

	plt.xticks(Leads_to_plot,fontsize=10)
	ax1.grid(linestyle='--')
	ax2.grid(linestyle='--')
	legend2 = plt.legend(handles=[Line2D([0],[0],color='w',marker=6,ms=8,markerfacecolor='k'),Line2D([0],[0],color='w',marker=7,ms=8,markerfacecolor='k')],labels=['10th Percentile','90th Percentile'],title='Markers',loc='lower left',bbox_to_anchor=(1.01,0.14))
	plt.legend(handles=CLS_custom_legend,labels=CLS_method_names,title='Legend',loc='center left',bbox_to_anchor=(1.01,0.5))
	if save == True:
		plt.savefig('All_Figures/Classification_Leads-vs-TP-TN_subplots_%s.svg'%(date),bbox_inches='tight')
	plt.show()



### CLASSIFICATION: FEATURES VS TPR (SUBPLOTS) ###
def CLASSIFICATION__Features_vs_TPR_subplots(date,save=False):
	breaks = [15,490]
	threshold = 0.76

	CLS_custom_legend.append(Line2D([0],[0],color='k'))
	CLS_custom_legend.append(Line2D([0],[0],color='w',marker=6,ms=8,markerfacecolor='k'))
	CLS_custom_legend.append(Line2D([0],[0],color='w',marker=7,ms=8,markerfacecolor='k'))
	CLS_method_names.append('Benchmark')
	CLS_method_names.append('10th Percentile')
	CLS_method_names.append('90th Percentile')

	for Lead in Leads:
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		i = 0
		fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,sharey=True,gridspec_kw={'hspace':0,'wspace':0.02},figsize=(10,5))
		for method in CLS_methods:
			TP_ratio_to_plot = []
			TN_ratio_to_plot = []
			TP_min_CI = []
			TP_max_CI = []
			TN_min_CI = []
			TN_max_CI = []
			Positives_Benchmark = []
			Negatives_Benchmark = []
			for num_features in num_features_all:
				TN = 0
				FP = 0
				TP = 0
				FN = 0
				Scenario_TP_ratio = []
				Scenario_TN_ratio = []
				for sc in scenarios:
					Scenario_FN = 0
					Scenario_TP = 0
					Scenario_TN = 0
					Scenario_FP = 0					
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
					results.columns = ['Solution','Predictions']
					for row in range(0,len(results['Solution'])):
						if results['Solution'][row] == True and results['Predictions'][row] == True: # both are above the threshold (not vulnerable)
							TN += 1
							Scenario_TN += 1
						elif results['Solution'][row] == True and results['Predictions'][row] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							FP += 1
							Scenario_FP += 1
						elif results['Solution'][row] == False and results['Predictions'][row] == False: # both are below the threshold (vulnerable)
							TP += 1
							Scenario_TP += 1
						elif results['Solution'][row] == False and results['Predictions'][row] == True: # solution is below threshold but prediction is above the threshold
							FN += 1
							Scenario_FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
					try:
						Scenario_TP_ratio.append(Scenario_TP/(Scenario_TP+Scenario_FN))
					except:
						Scenario_TP_ratio.append(0)
					try:
						Scenario_TN_ratio.append(Scenario_TN/(Scenario_TN+Scenario_FP))
					except:
						Scenario_TN_ratio.append(0)
				TP_ratio = TP/(TP+FN)
				TN_ratio = TN/(TN+FP)
				TP_ratio_to_plot.append(np.median(Scenario_TP_ratio))
				TN_ratio_to_plot.append(np.median(Scenario_TN_ratio))
				Positives_Benchmark.append(((TP+FN)/(TP+FN+FP+TN))**2)
				Negatives_Benchmark.append(((TN+FP)/(TP+FN+FP+TN))**2)
				TP_min_CI.append(np.percentile(Scenario_TP_ratio,10))
				TP_max_CI.append(np.percentile(Scenario_TP_ratio,90))
				TN_min_CI.append(np.percentile(Scenario_TN_ratio,10))
				TN_max_CI.append(np.percentile(Scenario_TN_ratio,90))

			mid_line_slope_TP = (TP_ratio_to_plot[-1]-TP_ratio_to_plot[-2])/(num_features_all[-1]-num_features_all[-2])
			mid_line_slope_TN = (TN_ratio_to_plot[-1]-TN_ratio_to_plot[-2])/(num_features_all[-1]-num_features_all[-2])
			left_line_TP = TP_ratio_to_plot[:-1]
			left_line_TP.append((breaks[0]-num_features_all[-2])*mid_line_slope_TP + TP_ratio_to_plot[-2])
			left_line_TN = TN_ratio_to_plot[:-1]
			left_line_TN.append((breaks[0]-num_features_all[-2])*mid_line_slope_TN + TN_ratio_to_plot[-2])
			left_line_x = num_features_all[:-1]
			left_line_x.append(breaks[0])
			right_line_TP = []
			right_line_TP.append(TP_ratio_to_plot[-1] - (num_features_all[-1]-breaks[1])*mid_line_slope_TP)
			right_line_TP.append(TP_ratio_to_plot[-1])
			right_line_TN = []
			right_line_TN.append(TN_ratio_to_plot[-1] - (num_features_all[-1]-breaks[1])*mid_line_slope_TN)
			right_line_TN.append(TN_ratio_to_plot[-1])
			right_line_x = []
			right_line_x.append(breaks[1])
			right_line_x.append(num_features_all[-1])
			ax1.plot(left_line_x,left_line_TP,color=colors_to_use[i])
			ax2.plot(right_line_x,right_line_TP,color=colors_to_use[i])
			ax3.plot(left_line_x,left_line_TN,color=colors_to_use[i])
			ax4.plot(right_line_x,right_line_TN,color=colors_to_use[i])

			## Plotting Benchmark ##
			if method == CLS_methods[-1]:
				ax1.plot(left_line_x,Positives_Benchmark[0:7],color='k')
				ax2.plot(right_line_x,Positives_Benchmark[-2:],color='k')
				ax3.plot(left_line_x,Negatives_Benchmark[0:7],color='k')
				ax4.plot(right_line_x,Negatives_Benchmark[-2:],color='k')

			ax1.scatter(left_line_x[0:6],TP_min_CI[0:6],c=colors_to_use[i],marker=6)
			ax2.scatter(right_line_x[-1],TP_min_CI[-1],c=colors_to_use[i],marker=6)
			ax1.scatter(left_line_x[0:6],TP_max_CI[0:6],c=colors_to_use[i],marker=7)
			ax2.scatter(right_line_x[-1],TP_max_CI[-1],c=colors_to_use[i],marker=7)
			
			ax3.scatter(left_line_x[0:6],TN_min_CI[0:6],c=colors_to_use[i],marker=6)
			ax4.scatter(right_line_x[-1],TN_min_CI[-1],c=colors_to_use[i],marker=6)
			ax3.scatter(left_line_x[0:6],TN_max_CI[0:6],c=colors_to_use[i],marker=7)
			ax4.scatter(right_line_x[-1],TN_max_CI[-1],c=colors_to_use[i],marker=7)
			i += 1
		ax1.spines['right'].set_visible(False)
		ax2.spines['left'].set_visible(False)
		ax3.spines['right'].set_visible(False)
		ax4.spines['left'].set_visible(False)
		ax1.yaxis.tick_left()
		ax1.tick_params(labelright='off')
		ax2.yaxis.tick_right()
		d = 0.015
		kwargs = dict(transform=ax1.transAxes,color='k',clip_on=False)
		ax1.plot((1-d,1+d),(-d,+d),**kwargs)
		ax1.plot((1-d,1+d),(1-d,1+d),**kwargs)
		kwargs.update(transform=ax2.transAxes)
		ax2.plot((-d,+d),(1-d,1+d),**kwargs)
		ax2.plot((-d,+d),(-d,+d),**kwargs)

		ax3.yaxis.tick_left()
		ax3.tick_params(labelright='off')
		ax4.yaxis.tick_right()
		d = 0.015
		kwargs = dict(transform=ax3.transAxes,color='k',clip_on=False)
		ax3.plot((1-d,1+d),(-d,+d),**kwargs)
		ax3.plot((1-d,1+d),(1-d,1+d),**kwargs)
		kwargs.update(transform=ax4.transAxes)
		ax4.plot((-d,+d),(1-d,1+d),**kwargs)
		ax4.plot((-d,+d),(-d,+d),**kwargs)

		ax1.set_xticks([1,2,3,4,5,10])
		ax2.set_xticks([500])
		ax1.grid(linestyle='--')
		ax2.grid(linestyle='--')
		ax1.set_xlim([0,15])
		ax2.set_xlim([490,501])

		ax3.set_xticks([1,2,3,4,5,10])
		ax4.set_xticks([500])
		ax3.grid(linestyle='--')
		ax4.grid(linestyle='--')
		ax3.set_xlim(0,15)
		ax4.set_xlim(490,501)

		fig.text(0.5,0.03,'Number of Features Used',ha='center',fontweight='bold',fontsize=12)
		fig.text(0.5,0.9,'Feature Reduction, Lead Time = %s Years'%(Lead[lead_cut:]),fontweight='bold',fontsize=14,ha='center')
		fig.text(0.07,0.5,'Ratio',fontweight='bold',fontsize=12,va='center',rotation='vertical')
		fig.text(0.262,0.2,'A) True Positive Ratios',fontweight='bold',fontsize=12)
		fig.text(0.652,0.2,'B) True Negative Ratios',fontweight='bold',fontsize=12)
		legend2 = plt.legend(handles=[Line2D([0],[0],color='w',marker=6,ms=8,markerfacecolor='k'),Line2D([0],[0],color='w',marker=7,ms=8,markerfacecolor='k')],labels=['10th Percentile','90th Percentile'],title='Markers',loc='lower left',bbox_to_anchor=(1.01,0.17))
		plt.legend(CLS_custom_legend,CLS_method_names,title='Legend',loc='center left',bbox_to_anchor=(1.01,0.5))
		if save == True:
			plt.savefig('All_Figures/Classification_Features-vs-TPR_%s_subplots_Threshold-%s_%s.svg'%(Lead,threshold,date),bbox_inches='tight')
		plt.show()



### CLASSIFICATION: LEADS VS SWITCHES (SUBPLOTS) ###
def CLASSIFICATION__Leads_vs_V_NV_switch_subplots(date,save=False):
	i=0
	threshold = 0.76
	plt.grid(linestyle='--')
	num_features = 500
	fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,sharex=True,figsize=(10,5),gridspec_kw={'hspace':0,'wspace':0.02})
	CLS_custom_legend.append(Line2D([0],[0],color='k'))
	CLS_custom_legend.append(Line2D([0],[0],color='w',marker=6,ms=8,markerfacecolor='k'))
	CLS_custom_legend.append(Line2D([0],[0],color='w',marker=7,ms=8,markerfacecolor='k'))
	CLS_method_names.append('Benchmark')
	CLS_method_names.append('10th Percentile')
	CLS_method_names.append('90th Percentile')
	ax1.grid(linestyle='--')
	ax2.grid(linestyle='--')
	for method in CLS_methods:
		TPRs = []
		TNRs = []
		TP_min_CI = []
		TP_max_CI = []
		TN_min_CI = []
		TN_max_CI = []
		Total_Poss = []
		Possible_Positives_Ratio = []
		Possible_Negatives_Ratio = []
		Positives_Benchmark = []
		Negatives_Benchmark = []
		for Lead in Leads[1::]:
			Years = 99-int(Lead[1:])
			TN = 0
			TP = 0
			FN = 0
			FP = 0
			switch_count_v_to_nv = 0
			switch_count_nv_to_v = 0
			Scenario_TP_ratio = []
			Scenario_TN_ratio = []
			count_all = 0
			for sc in scenarios:
				Scenario_FN = 0
				Scenario_TP = 0
				Scenario_TN = 0
				Scenario_FP = 0	
				try:
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
				except:
					continue
				results.columns = ['Solution','Predictions']
				for row in range(0,len(results['Solution'])):
					if results.loc[row,'Solution'] == False: # this is first year that the solution falls below the threshold
						threshold_index = row
						break
				if (threshold_index+int(Lead[1:])) <= len(results['Solution']):
					stop = threshold_index+int(Lead[1:])
				else:
					stop = len(results['Solution'])

				for row in range(int(Lead[1:]),stop):
					count_all += 1
					if results.loc[row,'Solution'] == True and results.loc[row,'Predictions'] == True: # both are above the threshold (not vulnerable)
						TN += 1
						Scenario_TN += 1
					elif results.loc[row,'Solution'] == True and results.loc[row,'Predictions'] == False: # solution is above the threshold but prediction is below the threshold (false positive)
						FP += 1
						Scenario_FP += 1
					elif results.loc[row,'Solution'] == False and results.loc[row,'Predictions'] == False: # both are below the threshold (vulnerable)
						TP += 1
						Scenario_TP += 1
					elif results.loc[row,'Solution'] == False and results.loc[row,'Predictions'] == True: # solution is below threshold but prediction is above the threshold
						FN += 1
						Scenario_FN += 1
				try:
					Scenario_TP_ratio.append(Scenario_TP/(Scenario_TP+Scenario_FN))
				except:
					Scenario_TP_ratio.append(0)
				try:
					Scenario_TN_ratio.append(Scenario_TN/(Scenario_TN+Scenario_FP))
				except:
					Scenario_TN_ratio.append(0)
			try:
				Possible_Positives_Ratio.append((TP+FN)/(TP+TN+FN+FP))
			except:
				Possible_Positives_Ratio.append(0)
			try:
				Possible_Negatives_Ratio.append((TN+FP)/(TP+TN+FN+FP))
			except:
				Possible_Negatives_Ratio.append(0)
			try:
				Positives_Benchmark.append(((TP+FN)/(TP+TN+FN+FP))**2)
			except:
				Positives_Benchmark.append(0)
			try:
				Negatives_Benchmark.append(((TN+FP)/(TP+TN+FN+FP))**2)
			except:
				Negatives_Benchmark.append(0)
			TPRs.append(np.median(Scenario_TP_ratio))
			TNRs.append(np.median(Scenario_TN_ratio))
			TP_min_CI.append(np.percentile(Scenario_TP_ratio,10))
			TP_max_CI.append(np.percentile(Scenario_TP_ratio,90))
			TN_min_CI.append(np.percentile(Scenario_TN_ratio,10))
			TN_max_CI.append(np.percentile(Scenario_TN_ratio,90))
			print(Lead,switch_count_nv_to_v,switch_count_v_to_nv,count_all) # tells you how many instances are used to generate statistics (sometimes very few)
		
		ax1.plot(Leads_to_plot[1::],TPRs,color=colors_to_use[i])
		ax2.plot(Leads_to_plot[1::],TNRs,color=colors_to_use[i])
		ax1.scatter(Leads_to_plot[1::],TP_min_CI,c=colors_to_use[i],marker=6)#,alpha=.5)
		ax1.scatter(Leads_to_plot[1::],TP_max_CI,c=colors_to_use[i],marker=7)#,alpha=.5)
		ax2.scatter(Leads_to_plot[1::],TN_min_CI,c=colors_to_use[i],marker=6)#,alpha=.5)
		ax2.scatter(Leads_to_plot[1::],TN_max_CI,c=colors_to_use[i],marker=7)#,alpha=.5)
		i += 1

	ax1.plot(Leads_to_plot[1::],Positives_Benchmark,color='k')
	ax2.plot(Leads_to_plot[1::],Negatives_Benchmark,color='k')
	fig.text(0.5,0.03,'Lead Times (years)',ha='center',fontweight='bold',fontsize=12)
	fig.text(0.07,0.5,'Ratio',fontweight='bold',fontsize=12,va='center',rotation='vertical')
	fig.text(0.5,0.9,'TP/TN Ratios, Threshold = %s'%(threshold),fontweight='bold',fontsize=14,ha='center')
	fig.text(0.165,0.79,'A) True Positive Ratios',fontweight='bold',fontsize=12)
	fig.text(0.555,0.2,'B) True Negative Ratios',fontweight='bold',fontsize=12)

	plt.xticks(Leads_to_plot[1::],fontsize=12)
	plt.legend(handles=CLS_custom_legend,labels=CLS_method_names,title='Legend',loc='center left',bbox_to_anchor=(1.01,0.5))
	if save == True:
		plt.savefig('All_Figures/Classification_Leads-vs-V-NV-switch_subplots_%s.svg'%(date),bbox_inches='tight')
	plt.show()	



### CLASSIFICATION: TIME VS TPR (SUBPLOTS) ###
def CLASSIFICATION__Time_vs_TPR_subplots(date,save=False):
	threshold = 0.76
	num_features = 500
	CLS_custom_legend.append(Line2D([0],[0],color='k'))
	CLS_method_names.append('Benchmark')

	for Lead in Leads:
		fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,sharex=True,figsize=(10,5),gridspec_kw={'hspace':0,'wspace':0.02})
		ax1.grid(linestyle='--')
		ax2.grid(linestyle='--')
		if Lead[1] == '0':
			lead_cut = 2
		else:
			lead_cut = 1
		Years = 99-int(Lead[1:])
		i=0
		for method in CLS_methods:
			TPRs = []
			TNRs = []
			Total_Poss = []
			Total_Neg = []
			Positives_Benchmark = []
			Negatives_Benchmark = []
			for year in range(Years):
				current_year_solution = []
				current_year_prediction = []
				TN = 0
				FP = 0
				TP = 0
				FN = 0
				for sc in scenarios:
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
					results.columns = ['Solution','Predictions']
					current_year_solution.append(results.loc[year,'Solution'])
					current_year_prediction.append(results.loc[year,'Predictions'])

				for row in range(0,len(current_year_prediction)): # JUST STARTED WORKING ON THIS SECTION
						if current_year_solution[row] == True and current_year_prediction[row] == True: # both are above the threshold (not vulnerable)
							# print('True Negative',results['Solution'][row],results['Predictions'][row])
							TN += 1
						elif current_year_solution[row] == True and current_year_prediction[row] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							# print('False Positive',results['Solution'][row],results['Predictions'][row])
							FP += 1
						elif current_year_solution[row] == False and current_year_prediction[row] == False: # both are below the threshold (vulnerable)
							# print('True Positive',results['Solution'][row],results['Predictions'][row])
							TP += 1
						elif current_year_solution[row] == False and current_year_prediction[row] == True: # solution is below threshold but prediction is above the threshold
							# print('False Negative',results['Solution'][row],results['Predictions'][row])
							FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
				try:
					TP_ratio = TP/(TP+FN)
				except:
					TP_ratio = 0
				try:
					TN_ratio = TN/(TN+FP)
				except:
					TN_ratio = 0
				if TN_ratio < 0.3:
					print(year,TN_ratio,TN,FP)
				Total_Poss.append((TP+FN)/(TN+FP+TP+FN))
				Total_Neg.append((TN+FP)/(TN+FP+TP+FN))
				Positives_Benchmark.append(((TP+FN)/(TN+FP+TP+FN))**2)
				Negatives_Benchmark.append(((TN+FP)/(TN+FP+TP+FN))**2)
				TPRs.append(TP_ratio)
				TNRs.append(TN_ratio)
			
			Min_markers_TPR = []
			Max_markers_TPR = []
			Min_markers_TNR = []
			Max_markers_TNR = []
			Point_1 = (19-int(Lead[1:])) # to 2020
			Point_2 = (39-int(Lead[1:])) # to 2040
			Point_3 = (59-int(Lead[1:])) # to 2060
			Point_4 = (79-int(Lead[1:])) # to 2080
			Point_5 = (99-int(Lead[1:])) # to end (2100)
			All_Points = [Point_1,Point_2,Point_3,Point_4,Point_5]
			Year_labels = list(range((2100-Years),2100))
			ax1.plot(Year_labels,TPRs,color=colors_to_use[i])
			ax2.plot(Year_labels,TNRs,color=colors_to_use[i])
			ax1.plot(Year_labels,Positives_Benchmark,color='k')
			ax2.plot(Year_labels,Negatives_Benchmark,color='k')
			i += 1
		fig.text(0.5,0.03,'Years',ha='center',fontweight='bold',fontsize=12)
		fig.text(0.07,0.5,'Ratio',fontweight='bold',fontsize=12,va='center',rotation='vertical')
		fig.text(0.5,0.9,'TP/TN Ratios, Lead Time = %s Years, Threshold = %s'%(int(Lead[1:]),threshold),fontweight='bold',fontsize=14,ha='center')
		fig.text(0.15,0.75,'A) True Positive \n     Ratios',fontweight='bold',fontsize=12)
		fig.text(0.72,0.75,'B) True Negative \n     Ratios',fontweight='bold',fontsize=12)
		plt.legend(handles=CLS_custom_legend,labels=CLS_method_names,title='Legend',loc='center left',bbox_to_anchor=(1.01,0.5))
		if save == True:
			plt.savefig('All_Figures/Classification_Time_vs_TPR_%s_threshold-%s_subplots_no-markers_%s.svg'%(Lead,threshold,date),bbox_inches='tight')
		plt.show()



### CLASSIFICATION: CORRELATION HEATMAP ###
def CLASSIFICATION__Correlation_heatmap(date,save=False):
	num_features = 10 # choose how many features to view
	import seaborn as sns 
	sns.set(style='white',font_scale=1.4)
	names_L00 = ['M07 30-yr AVG\nOroville Storage','M06 30-yr AVG\nShasta Storage','M08 30-yr AVG\nOroville Storage','M07 30-yr AVG\nShasta Storage','M11 30-yr AVG\nTotal Shortage',
				'M07 30-yr AVG\nTotal Shortage','ANN 30-yr AVG\nTotal Shortage','M10 30-yr AVG\nTotal Shortage','M11 30-yr SD\nTotal Shortage','M06 30-yr AVG\nOroville Storage']
	names_L01 = ['M07 30-yr AVG\nOroville Storage','M08 30-yr AVG\nOroville Storage','M07 30-yr AVG\nShasta Storage','M06 30-yr AVG\nShasta Storage','ANN 30-yr AVG\nTotal Shortage',
				'M07 30-yr AVG\nTotal Shortage','M11 30-yr AVG\nTotal Shortage','M11 30-yr SD\nTotal Shortage','M10 30-yr AVG\nTotal Shortage','M06 30-yr AVG\nOroville Storage']
	names_L05 = ['M07 30-yr AVG\nOroville Storage','M07 20-yr AVG\nOroville Storage','M07 30-yr AVG\nShasta Storage','M08 30-yr AVG\nOroville Storage','ANN 20-yr AVG\nFolsom Storage',
				'M07 20-yr AVG\nShasta Storage','M11 20-yr AVG\nTotal Shortage','M06 30-yr AVG\nShasta Storage','M09 30-yr AVG\nFolsom Storage','M08 20-yr AVG\nOroville Storage']
	names_L10 = ['M07 20-yr AVG\nOroville Storage','M08 20-yr AVG\nOroville Storage','M07 20-yr AVG\nShasta Storage','M06 20-yr AVG\nShasta Storage','M09 20-yr AVG\nFolsom Storage',
				'M09 20-yr AVG\nDelta Inflow','M08 20-yr AVG\nFolsom min Temp','M09 20-yr AVG\nTotal Pumped from Delta','M07 30-yr AVG\nFolsom max Temp','M10 20-yr AVG\nFolsom Storage']
	names_L20 = ['ANN 10-yr AVG\nFolsom max Temp','M08 20-yr AVG\nOroville Storage','ANN 10-yr AVG\nFolsom avg Temp','M07 20-yr AVG\nOroville Storage','ANN 20-yr AVG\nFolsom max Temp',
				'M07 40-yr AVG\nShasta Storage','M07 20-yr AVG\nFolsom max Temp','M07 10-yr AVG\nShasta Storage','M08 10-yr AVG\nOroville Storage','M11 10-yr AVG\nTotal Shortage']
	names = ['names_L00','names_L01','names_L05','names_L10','names_L20']
	names2 = [names_L00,names_L01,names_L05,names_L10,names_L20]
	for lead in Leads:
		for name in names:
			if lead in name:
				names_to_use = names2[names.index(name)]

		data = pd.read_csv('Data_Matrices_19-06-03/None-Data_L%s.csv'%(int(lead[1:])),index_col=0)
		features = pd.read_csv('Removed_Features_Lists_Second_Run/Feature_Importances__value__'+lead+'_R30.csv',index_col=0)
		important_features_list = features.iloc[:num_features].index
		data_return = data[important_features_list]
		corr = data_return.corr()
		mask = np.zeros_like(corr,dtype=np.bool)
		mask[np.triu_indices_from(mask)] = True
		f, ax = plt.subplots(figsize=(11,9))
		cmap = sns.diverging_palette(10, 220, as_cmap=True)
		sns.heatmap(corr,mask=mask,cmap=cmap,square=True,center=0,linewidths=.5,cbar_kws={'shrink':.5},vmin=-1,vmax=1,annot=True,xticklabels=names_to_use,yticklabels=names_to_use)
		plt.title('Top Features Correlations, Lead Time = %s Years'%(int(lead[1:])),fontweight='bold')
		if save == True:
			plt.savefig('All_Figures/Correlation_heatmap_with_labels_%s_%s.svg'%(lead,date),bbox_inches='tight')
		plt.show()



### CLASSIFICATION: OVERVIEW FIG ###
def CLASSIFICATION__Overview_Fig(date,save=False):
	f, (ax1,ax2) = plt.subplots(2,1,sharex=True,gridspec_kw={'hspace':0,'wspace':0.02})
	lead = 10

	## Either choose a scenario name or number for the plot 
	sc = 'gfdl-esm2m_rcp60_r1i1p1' # if using a scenario name, put it here
	for sc_test in scenarios:
		if sc == sc_test:
			scenario_num = scenarios.index(sc)
	# scenario_num = 28 # if you want to use the scenario number (0 through 96) instead of name, activate this line

	Year_list = list(range((2100-99+lead),2100))
	if lead == 5:
		data = pd.read_csv('Data_Matrices_19-06-03/None-Data_L5.csv',index_col=0)
		reliability = pd.read_csv('Data_Matrices_19-06-03/None-Solution_L5.csv',index_col=0)
		var_1 = data['ORO_storage_30_avg_m07']
		var_2 = data['ORO_storage_20_avg_m07']
		var_3 = data['SHA_storage_30_avg_m07']
		var_4 = data['ORO_storage_30_avg_m08']
		var_5 = data['FOL_storage_20_avg_ann']
	elif lead == 10:
		data = pd.read_csv('Data_Matrices_19-06-03/None-Data_L10.csv',index_col=0)
		reliability = pd.read_csv('Data_Matrices_19-06-03/None-Solution_L10.csv',index_col=0)
		var_1 = data['ORO_storage_20_avg_m07']
		var_2 = data['ORO_storage_20_avg_m08']
		var_3 = data['SHA_storage_20_avg_m07']
		var_4 = data['SHA_storage_20_avg_m06']
		var_5 = data['FOL_storage_20_avg_m09']
	for x in range(0,97):
		start = int(x*(len(var_1)/97))
		stop = int((x+1)*(len(var_1)/97))
		ax2.plot(Year_list,reliability.iloc[start:stop],color='lightgray')
	start = int(scenario_num*(len(var_1)/97))
	stop = int((scenario_num+1)*(len(var_1)/97))

	ax1.plot(Year_list,var_1.iloc[start:stop],color='lightgray',linestyle='--')
	ax1.plot(Year_list,var_2.iloc[start:stop],color='lightgray',linestyle='--')
	ax1.plot(Year_list,var_3.iloc[start:stop],color='lightgray',linestyle='--')
	ax1.plot(Year_list,var_4.iloc[start:stop],color='lightgray',linestyle='--')
	ax1.plot(Year_list,var_5.iloc[start:stop],color='lightgray',linestyle='--')

	for x in range(start,stop):
		if reliability.iloc[x,0] < 0.76:
			index = x - start
			threshold_year = Year_list[index]
			break

	new_stop = int(stop-(2100-threshold_year)-lead)
	year_stop = int(threshold_year-(2100-99+lead)-lead)

	ax1.plot(Year_list[0:year_stop],var_1.iloc[start:new_stop],color='r')
	ax1.plot(Year_list[0:year_stop],var_2.iloc[start:new_stop],color='cyan')
	ax1.plot(Year_list[0:year_stop],var_3.iloc[start:new_stop],color='limegreen')
	ax1.plot(Year_list[0:year_stop],var_4.iloc[start:new_stop],color='violet')
	ax1.plot(Year_list[0:year_stop],var_5.iloc[start:new_stop],color='cornflowerblue')
	ax2.plot(Year_list,reliability.iloc[start:stop],color='k')	

	ax2.add_line(Line2D([threshold_year,threshold_year],[0,1],color='k',linestyle='--'))
	ax1.add_line(Line2D([threshold_year,threshold_year],[0,4000],color='k',linestyle='--'))
	ax1.add_line(Line2D([threshold_year-lead,threshold_year-lead],[0,4000],color='b',linestyle='--'))
	ax2.add_line(Line2D([2000,2100],[0.76,0.76],color='k',linestyle='--'))

	ax1.set_ylabel('Reservoir Storage (TAF)',fontweight='bold')
	y_label = ax2.set_ylabel('30-yr Reliability',fontweight='bold')
	ax2.yaxis.set_label_coords(-.101,.5)
	ax2.text(threshold_year+1,0.49,'t',fontweight='bold')
	ax2.annotate('Threshold',xy=(2075,0.76),xytext=(2080,0.85),fontweight='bold',arrowprops=dict(facecolor='black',width=2,headwidth=9)) 
	ax1.annotate(' ',xy=(threshold_year,1270),xytext=(threshold_year-lead-1,1200),fontweight='bold',arrowprops=dict(facecolor='b',width=2,headwidth=6,headlength=6,color='b')) 
	ax1.text(threshold_year-25,1000,'  10-year\nlead time',fontweight='bold',color='b')
	plt.xlabel('Years',fontweight='bold')

	legend_lines = [Line2D([0],[0],color='r'),Line2D([0],[0],color='cyan'),Line2D([0],[0],color='limegreen'),Line2D([0],[0],color='violet'),Line2D([0],[0],color='cornflowerblue'),Line2D([0],[0],color='k'),Line2D([0],[0],color='lightgray'),Line2D([0],[0],color='lightgray',linestyle='--')]
	feature_labels = ['Oroville Storage, 30-yr avg of July','Oroville Storage, 20-yr avg of July','Shasta Storage, 30-yr avg of July','Oroville Storage, 30-yr avg of August','Folsom Storage, 20-yr annual avg','Example Scenario','Other Scenarios','Unavailable Information at time t - lead']
	plt.legend(handles=legend_lines,labels=feature_labels,title='Feature Name',loc='center left',bbox_to_anchor=(1.01,1))
	if save == True:
		plt.savefig('All_Figures/Methods_Fig_Scenario_%s_%s'%(scenario_num,date),bbox_inches='tight')
	plt.show()



### P-VALUE HEATMAP ###
def P_Value_Heatmap(date,save=False):
	num_features = 500

	import seaborn as sns
	for method in CLS_methods:
		print(method)
		fig,(ax1,ax2,axcb) = plt.subplots(1,3,figsize=(9,5),gridspec_kw={'width_ratios':[1,1,0.08],'hspace':0,'wspace':0.02})
		ax1.get_shared_y_axes().join(ax2)
		ax1.get_shared_x_axes().join(ax2)
		Pvalue_Results_TP = pd.DataFrame(columns=Leads, index=thresholds_all)
		Pvalue_Results_TN = pd.DataFrame(columns=Leads, index=thresholds_all)
		for threshold in thresholds_all:
			Positives_Benchmark = []
			Negatives_Benchmark = []
			TN = 0
			TP = 0
			FP = 0
			FN = 0
			for Lead in Leads:
				Scenario_TP_ratio = []
				Scenario_TN_ratio = []
				for sc in scenarios:
					Scenario_TP = 0
					Scenario_TN = 0
					Scenario_FP = 0
					Scenario_FN = 0
					results = pd.read_csv('Results_Raw_Outputs/%s-Predictions(%s)_%s_features-%s_%s_%s_%s_threshold-%s.csv'%(sc,'CLS',Lead,num_features,sol_type,'R30',method,threshold),index_col=0)
					results.columns = ['Solution','Predictions']
					for row in range(0,len(results['Solution'])):
						if results['Solution'][row] == True and results['Predictions'][row] == True: # both are above the threshold (not vulnerable)
							TN += 1
							Scenario_TN += 1
						elif results['Solution'][row] == True and results['Predictions'][row] == False: # solution is above the threshold but prediction is below the threshold (false positive)
							FP += 1
							Scenario_FP += 1
						elif results['Solution'][row] == False and results['Predictions'][row] == False: # both are below the threshold (vulnerable)
							TP += 1
							Scenario_TP += 1
						elif results['Solution'][row] == False and results['Predictions'][row] == True: # solution is below threshold but prediction is above the threshold
							FN += 1
							Scenario_FN += 1
						else:
							print('Unable to classify scenario %s, row %s into TP/TN/FP/FN.'%(sc,row))
					try:
						Scenario_TP_ratio.append(Scenario_TP/(Scenario_TP+Scenario_FN))
					except:
						print(' ')
					try:
						Scenario_TN_ratio.append(Scenario_TN/(Scenario_TN+Scenario_FP))
					except:
						print(' ')
				Pos_Bench = (((TP+FN)/(TP+FN+FP+TN))**2)
				Neg_Bench = (((TN+FP)/(TP+FN+FP+TN))**2)
				Positives_Benchmark.append(Pos_Bench)
				Negatives_Benchmark.append(Neg_Bench)

				# Calculating P-values
				step = .1
				for x in np.arange(0,100+step,step):
					TP_value = np.percentile(Scenario_TP_ratio,x)
					if TP_value >= Pos_Bench:
						Pvalue_Results_TP.loc[threshold,Lead] = x/100
						break
				for x in np.arange(0,100+step,step):
					TN_value = np.percentile(Scenario_TN_ratio,x)
					if TN_value >= Neg_Bench:
						Pvalue_Results_TN.loc[threshold,Lead] = x/100
						break
		Pvalue_Results_TP = Pvalue_Results_TP[Pvalue_Results_TP.columns].astype(float)
		H1 = sns.heatmap(Pvalue_Results_TP,cmap='seismic',vmin=0,vmax=1,annot=True,xticklabels=[0,1,5,10,20],yticklabels=thresholds_all,ax=ax1,cbar=False)
		H1.set_ylabel('Thresholds',fontweight='bold',fontsize=12)
		Pvalue_Results_TN = Pvalue_Results_TN[Pvalue_Results_TN.columns].astype(float)
		H2 = sns.heatmap(Pvalue_Results_TN,cmap='seismic',vmin=0,vmax=1,annot=True,xticklabels=[0,1,5,10,20],yticklabels=[],ax=ax2,cbar_ax=axcb)
		fig.text(0.5,0.03,'Leads (years)',ha='center',fontweight='bold',fontsize=12)
		fig.text(0.3,0.9,'True Positives',fontweight='bold',fontsize=14,ha='center')
		fig.text(0.69,0.9,'True Negatives',fontweight='bold',fontsize=14,ha='center')
		if save == True:
			plt.savefig('All_Figures/P-Values_TN_%s_%s.svg'%(method,date),bbox_inches='tight')
		plt.show()
	