import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
	# path = "/mnt/be2a0173-321f-4b9d-b05a-addba547276f/kolanzl/SpaceLab_stable/SpaceLab/jobs/small_balls6/N_10/T_1/"
	# fileprefix = "1_2_R1e-05_v4e-01_cor0.63_mu0.1_rho2.25_k4e+00_Ha5e-12_dt5e-10_"
	path = []
	fileprefix = []
	path.append("/mnt/be2a0173-321f-4b9d-b05a-addba547276f/kolanzl/SpaceLab_stable/SpaceLab/jobs/small_balls13/N_10/T_1/")
	# path.append("/mnt/be2a0173-321f-4b9d-b05a-addba547276f/kolanzl/SpaceLab_stable/SpaceLab/jobs/small_balls6/N_10/T_1/")
	# path.append("/mnt/be2a0173-321f-4b9d-b05a-addba547276f/kolanzl/SpaceLab_stable/SpaceLab/jobs/small_balls6/N_10/T_1/")
	# path.append("/mnt/be2a0173-321f-4b9d-b05a-addba547276f/kolanzl/SpaceLab_stable/SpaceLab/jobs/small_balls6/N_10/T_1/")
	# fileprefix.append("2_R3e-05_v4e-01_cor0.63_mu0.1_rho2.25_k4e+00_Ha5e-12_dt5e-10_")
	fileprefix.append("1_2_R3e-05_v4e-01_cor0.63_mu0.1_rho2.25_k4e+00_Ha5e-12_dt5e-10_")
	# fileprefix.append("2_2_R3e-05_v4e-01_cor0.63_mu0.1_rho2.25_k4e+00_Ha5e-12_dt5e-10_")
	# fileprefix.append("9_2_R3e-05_v4e-01_cor0.63_mu0.1_rho2.25_k4e+00_Ha5e-12_dt5e-10_")

	
	# print(data)
	# print(data[1])
	# # exit(0)
	for i,p in enumerate(path):
		datafile = p + fileprefix[i] + "fricData.csv"

		header = np.genfromtxt(datafile,delimiter=',',dtype=str)[0]
		data = np.genfromtxt(datafile,delimiter=',',dtype=np.float64,skip_header=1)
		datashape0,datashape1 = data.shape
		# data = data.transpose()
		# print(data.shape)
		# print(np.transpose(data).shape)
		# print(data)
		# # print(data[:,0])
		# print(header)


		rolling = np.zeros((3,int(datashape1/6),datashape0))
		sliding = np.zeros((3,int(datashape1/6),datashape0))

		j = 0
		k = 0 
		for i in range(0,datashape1,3):
			if i % 2 == 0:
				rolling[:,j,:] = np.transpose(data[:,i:i+3])
				j += 1
			else:
				sliding[:,k,:] = np.transpose(data[:,i:i+3])
				k += 1
		data = []

		rolling_mag = np.zeros((int(datashape1/6),datashape0))
		sliding_mag = np.zeros((int(datashape1/6),datashape0))
		xdata = np.linspace(0,1,datashape0) #dummy x data. x is actually time
		for i in range(rolling.shape[1]):
			for j in range(rolling.shape[2]):
				rolling_mag[i,j] = np.linalg.norm(rolling[:,i,j])
				sliding_mag[i,j] = np.linalg.norm(sliding[:,i,j])

		rolling = []
		sliding = []
		
		# ax = plt.axes(projection='3d')
		for i in range(int(datashape1/6)):
			fig, ax = plt.subplots(2,1,figsize=(15,10))
			ax = ax.flatten()
			ax[0].plot(xdata, rolling_mag[i], label='ball {}'.format(i))
			ax[0].set_title('rolling friction magnitude')
			ax[0].set_xlabel('time (t/tmax)')
			ax[0].set_ylabel('friction')
			ax[0].legend()

			ax[1].plot(xdata, sliding_mag[i], label='ball {}'.format(i))
			ax[1].set_title('sliding friction magnitude')
			ax[1].set_xlabel('time (t/tmax)')
			ax[1].set_ylabel('friction')
			ax[1].legend()

			plt.savefig("figures/slidRollFric_{}.png".format(i))

			plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()