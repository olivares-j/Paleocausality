import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import grangercausalitytests

def plot_ts(X,labels=None,file="Synthetic.png"):
	plt.figure()
	for name,values in X.iteritems():
		plt.plot(values,label=name)
	plt.xlabel("Time")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.tight_layout()
	plt.savefig(file)
	plt.close()

def plot_Pmatrix(X,file="P_values.png",alpha=0.01,figsize=(5,5)):
	plt.figure(figsize=figsize)
	ax = sns.heatmap(
					X, 
					vmin=0, vmax=alpha,
					cmap=sns.light_palette("seagreen", n_colors=30,reverse=True),
					square=True,
					annot=True, fmt="0.2f"
					)
	ax.set_xticklabels(
					ax.get_xticklabels(),
					rotation=45,
					horizontalalignment='right'
					)
	ax.set_yticklabels(
					ax.get_xticklabels(),
					rotation=0,
					horizontalalignment='right'
					)
	plt.tight_layout()
	plt.savefig(file)
	plt.close()

def GrangerTS(X,order=3,test_type="ssr_ftest",verbose=False):
	"""Perform the actual Granger test of each pair of TS
		Test types: ssr_ftest,lrtest,ssr_chi2test,params_ftest
	"""
	if isinstance(X,pd.DataFrame):
		names = X.columns.values
		X = X.to_numpy()
	else:
		names = [str(i) for i in range(X.shape[1])]

	N,D  = X.shape
	Pval = np.zeros((D,D))
	for i in range(D):
		for j in range(D):
			gc = grangercausalitytests(X[:,[i,j]],[order],verbose=verbose)
			Pval[i,j] = gc[order][0][test_type][1]
	df = pd.DataFrame(Pval,columns=names)
	return df