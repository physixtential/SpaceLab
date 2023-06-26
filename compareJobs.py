import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/lpkolanz/Desktop/SpaceLab_branch/SpaceLab")
import utils as u
import porosity_FD as p

def main():
	base = os.getcwd() + "/jobs/"
	folder1 = base + "accuracyTest5/"
	# folder2 = base + "accuracyTest2/"
	folder2 = "/home/lpkolanz/Desktop/SpaceLab_branch/SpaceLab/jobs/accuracyTest10/N_10/T_100/"

	max_ind = -1
	for file in os.listdir(folder1):
		if file[-4:] == ".csv":
			try:
				ind = int(file.split('_')[0])
				if ind > max_ind:
					max_ind = ind
					body = file.split('_')[1:-1]
			except ValueError:
				continue

	file1 = str(max_ind)+'_'+'_'.join(body)+"_simData.csv"

	max_ind = -1
	for file in os.listdir(folder2):
		if file[-4:] == ".csv":
			try:
				ind = int(file.split('_')[0])
				if ind > max_ind:
					max_ind = ind
					body = file.split('_')[1:-1]
			except ValueError:
				continue

	file2 = str(max_ind)+'_'+'_'.join(body)+"_simData.csv"
	file2 = "9_simData.csv"

	porositiesabc=[]
	porositiesKBM=[]
	contacts=[]
	FD_data=[]

	N = 10
	temp = 100
	show_FD_plots = False
	for data_folder in [folder1,folder2]:
		count = 0
		for root_dir, cur_dir, files in os.walk(data_folder):
		    count += len(files)
		if count/3 > N:
			porositiesabc.append(p.porosity_measure1(data_folder,N-1))
			porositiesKBM.append(p.porosity_measure2(data_folder,N-1))
			contacts.append(p.number_of_contacts(data_folder,N-1))
			if not np.isnan(porositiesabc[-1]):
				o3dv = u.o3doctree(data_folder,overwrite_data=False,index=N-1,Temp=temp)
				o3dv.make_tree()
				FD_data.append(o3dv.calc_fractal_dimension(show_graph=show_FD_plots))
	print("Porosity abc: [{},{}], difference: {}".format(porositiesabc[0],porositiesabc[1],np.diff(porositiesabc)))	
	print("Porosity KBM: [{},{}], difference: {}".format(porositiesKBM[0],porositiesKBM[1],np.diff(porositiesKBM)))	
	print("contacts    : [{},{}], difference: {}".format(contacts[0],contacts[1],np.diff(contacts)))	
	print("Fract dim   : [{},{}], difference: {}".format(FD_data[0],FD_data[1],np.diff(FD_data)))	

	data1 = np.loadtxt(folder1+file1,delimiter=",",dtype=np.float64,skiprows=1)[-1]
	data2 = np.loadtxt(folder2+file2,delimiter=",",dtype=np.float64,skiprows=1)[-1]
	fig, ax = plt.subplots(1,1,figsize=(15,7))
	ax.plot(np.arange(len(data1)),np.subtract(data2,data1))
	plt.show()

if __name__ == '__main__':
	main()