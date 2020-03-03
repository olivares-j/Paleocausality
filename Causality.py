import sys
import numpy as np
import pandas as pd
from Functions import plot_ts,plot_Pmatrix,GrangerTS

orders = [5,10,15,20,25,30,35,40,45,50]
dir_images = "./Images/"
file  = "./Data/Unified.csv"

#------- Read data -----
X = pd.read_csv(file)

delta_age = X["Age"][1]-X["Age"][0]


#------ Drop Age related columns ------
remove = np.where([ "Age" in col for col in X.columns.values])[0]
X.drop(columns=X.columns.values[remove],inplace=True)

#------- Drop NaNs ----------------
X.dropna(axis=0,how="any",inplace=True)

#----- Plot the TS------
names = X.columns.values
plot_ts(X,labels=names,file=dir_images+"Real.png")

for order in orders:
	#---- Compute the P_values -----
	P = GrangerTS(X,order=order,verbose=False)

	#-- Plot the matrix of P values ---
	file_png = dir_images+"Causality_{0:3.1f}ky.png".format(order*delta_age)
	plot_Pmatrix(P,file=file_png,figsize=(14,14))
