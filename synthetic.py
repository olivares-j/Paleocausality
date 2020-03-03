import sys
import numpy as np
import pandas as pd
from Functions import plot_ts,plot_Pmatrix,GrangerTS

def generate_syn(A,time_steps=1000):
	'''
	Generate the synthetic time series data
	'''
	tau_max,dim,_ = A.shape

	assert time_steps > tau_max,"Error: time steps must be larger than tau_max"

	Xs = np.random.normal(size=(time_steps,dim))
	for tau in range(tau_max):
		for t in range(1,time_steps):
			Xs[t]  += np.dot(A[tau],Xs[t-(1+tau)])

	return pd.DataFrame(Xs)


if __name__ == "__main__":

	A = np.array([[[0.0,0.0,0.0],[1.5,0.0,0.0],[2.5,0.0,0.0]],
			      [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]])
	print(A)
	X = generate_syn(A,100)

	#----- Plot the TS------
	plot_ts(X,file="Synthetic.png")

	#---- Compute the P_values -----
	P = GrangerTS(X,verbose=False)
	
	#-- Plot the matrix of P values ---
	plot_Pmatrix(P)
