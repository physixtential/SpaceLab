import os
import json
import multiprocessing as mp
import subprocess
import numpy as np
import time


def run_job(location,num_balls):
	cmd = ["python3", "{}run_sim.py".format(location), location, str(num_balls)]
	subprocess.run(cmd)

if __name__ == '__main__':
	#make new output folders
	curr_folder = os.getcwd() + '/'
	job_set_name = "cpu_stress"
	folder_name_scheme = "cores_"

	job_set_folder = curr_folder + 'jobs/' + job_set_name + '/'
	if not os.path.exists(job_set_folder):
		os.makedirs(job_set_folder)
	else:
		print("Job set '{}' already exists.".format(job_set_folder))

	#Make an array of what you want to vary
	Temps = [1000]
	N = [5]
	num_cores = np.arange(1,11)

	#actually make the folders
	folds = []
	folder_values = num_cores
	folders = [job_set_folder + folder_name_scheme + str(val) + '/' for val in folder_values]
	for i,folder in enumerate(folders):
		for j in range(1,i+2):
			folds.append(folder+'job_{}/'.format(j))
	folders = folds

	for new_folder in folders:
		if not os.path.exists(new_folder):
			os.makedirs(new_folder)
		else:
			print("Job '{}' already exists.".format(new_folder))
	
	if len(Temps) != len(folders):
		Temps = [Temps[0] for i in range(len(folders))]

	#load default input file
	with open(curr_folder+"default_files/default_input.json",'r') as fp:
		input_json = json.load(fp)

	#loop through folders and generate input file and run_script.py for each
	#changing the necessary values in the input file
	for index,folder in enumerate(folders):
		input_json['temp'] = Temps[0]
		# input_json['Nballs'] = N[index]
		with open(folder + "input.json",'w') as fp:
			json.dump(input_json,fp,indent=4)

		#add run script and executable to folders
		os.system("cp default_files/run_sim.py {}run_sim.py".format(folders[index]))
		os.system("cp ColliderSingleCore/ColliderSingleCore.o {}ColliderSingleCore.o".format(folders[index]))

	if len(N) != len(folders):
		N = [str(N[0]) for i in range(len(folders))]

	inputs = list(zip(folders,N))
	times = []

	for i in num_cores:
		start_t = time.perf_counter()
		start = np.sum(num_cores[0:i-1])
		end = np.sum(num_cores[0:i])
		with mp.Pool(processes=i) as pool:
			pool.starmap(run_job,list(inputs)[start:end]) 
			pool.close()
			pool.join()
		end_t = time.perf_counter()
		times.append(end_t-start_t)

	print(times)


	
