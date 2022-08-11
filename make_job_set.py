import os
import json
import multiprocessing as mp
import subprocess

def run_job(location):
	cmd = ["python3", "{}run_sim.py".format(location), location]
	subprocess.run(cmd)

if __name__ == '__main__':
	#make new output folders
	curr_folder = os.getcwd() + '/'
	job_set_name = "tempVariance_attempt8"
	folder_name_scheme = "T_"

	job_set_folder = curr_folder + 'jobs/' + job_set_name + '/'
	if not os.path.exists(job_set_folder):
		os.makedirs(job_set_folder)
	else:
		print("Job set '{}' already exists.".format(job_set_folder))

	#Make an array of what you want to vary
	Temps = [3,10,30,100,300,1000]

	#actually make the folders
	folder_values = Temps
	folders = [job_set_folder + folder_name_scheme + str(val) + '/' for val in folder_values]
	for new_folder in folders:
		if not os.path.exists(new_folder):
			os.mkdir(new_folder)
		else:
			print("Job '{}' already exists.".format(new_folder))

	#load default input file
	with open(curr_folder+"default_files/default_input.json",'r') as fp:
		input_json = json.load(fp)

	#loop through folders and generate input file and run_script.py for each
	#changing the necessary values in the input file
	for index,value in enumerate(folder_values):
		input_json['temp'] = value
		with open(folders[index] + "input.json",'w') as fp:
			json.dump(input_json,fp,indent=4)

		#add run script and executable to folders
		os.system("cp default_files/run_sim.py {}run_sim.py".format(folders[index]))
		os.system("cp ColliderSingleCore/ColliderSingleCore.o {}ColliderSingleCore.o".format(folders[index]))


	with mp.Pool(processes=len(folders)) as pool:
		pool.map(run_job,folders) 
	# #loop folders again and run stuff
	# for job in folders:
	# 	os.system("cd {}".format(job))
	# 	os.system("pyth")

	# os.system("cd {}".format(curr_folder))

	
