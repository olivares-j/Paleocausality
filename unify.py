import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir_data = "./Data/"
files    = ["First_set.csv","Second_set.csv","Third_set.csv"]#,"Fourth_set.csv"]
output   = dir_data + "Unified.csv"
variate  = "Age (LR04)"


#------------------ Read files into DataFrames ------------------
datasets = []
lengths  = []
columns  = []

for file in files:
	x = pd.read_csv(dir_data+file)
	x.drop(columns=["Depth (crmcd)"],inplace=True)
	lengths.append(x.shape[0])
	columns.append(x.shape[1])
	datasets.append(x)
#-------------------------------------------------------


#-------- Linearly spaced age ----------------------------
age = np.linspace(datasets[0][variate].min(),
					datasets[0][variate].max(),200)

df = pd.DataFrame(age,columns=["Age"])

#----------- Interpolate -----------------------------
for dataset in datasets:
	x = dataset[variate]
	columns = list(dataset.columns)
	columns.remove(variate)
	for column in columns:
		y = dataset[column]
		idx = np.where(np.isfinite(y))[0]
		z = np.interp(age,xp=x[idx],fp=y[idx])
		df[column] = z

		#-------- Plots ------------------------------------
		plt.figure(0)
		plt.scatter(x,y,label="original")
		plt.plot(df["Age"],df[column],label="Interpolation")
		plt.legend()
		plt.savefig("./Images/{0}.png".format(column))
		plt.close(0)
		#--------------------------------------------------
#------------------------------------------------------

#-------- Saves DataFrame -------------------
df.to_csv(output,index=False)