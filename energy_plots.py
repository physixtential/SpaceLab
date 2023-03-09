import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
	# path = "/mnt/be2a0173-321f-4b9d-b05a-addba547276f/kolanzl/SpaceLab_stable/SpaceLab/jobs/test1/N_1/T_1/"
	# fileprefix = "2_R3e-05_v4e-01_cor0.63_mu0.1_rho2.25_k4e+00_Ha5e-12_dt5e-10_"
	path = "/mnt/be2a0173-321f-4b9d-b05a-addba547276f/kolanzl/SpaceLab_stable/SpaceLab/jobs/small_balls4/N_10/T_1/"
	fileprefix = "5_2_R3e-05_v4e-01_cor0.63_mu0.1_rho2.25_k4e+00_Ha5e-12_dt5e-10_"

	datafile = path + fileprefix + "energy.csv"

	header = np.genfromtxt(datafile,delimiter=',',dtype=str)[0]
	data = np.genfromtxt(datafile,delimiter=',',dtype=np.float64,skip_header=1)
	data = data.transpose()
	# print(data)
	# print(data[1])
	# # exit(0)

	fig, ax = plt.subplots(2,3)

	ax[0,0].plot(data[0],data[1],label="PE")
	ax[0,1].plot(data[0],data[2],label="KE")
	ax[0,2].plot(data[0],data[3],label="PE+KE")

	ax[1,0].plot(data[0],data[1],label="PE")
	ax[1,0].plot(data[0],data[2],label="KE")
	ax[1,0].plot(data[0],data[3],label="PE+KE")
	
	ax[1,1].plot(data[0],data[4],label="mom")

	ax[1,2].plot(data[0],data[5],label="ang mom")

	ax[0,0].legend()
	ax[0,1].legend()
	ax[0,2].legend()
	ax[1,0].legend()
	ax[1,1].legend()
	ax[1,2].legend()

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()