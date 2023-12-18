import os
import json
import multiprocessing as mp
import subprocess

relative_path = "../"
relative_path = '/'.join(__file__.split('/')[:-1]) + '/' + relative_path
project_path = os.path.abspath(relative_path) + '/'

def run_job(location,num_balls):
	cmd = ["python3", "{}run_sim.py".format(location), location, str(num_balls)]
	# print(cmd)
	subprocess.run(cmd)

if __name__ == '__main__':
	#make new output folders
	# curr_folder = os.getcwd() + '/'


	try:
		# os.chdir("{}ColliderSingleCore".format(curr_folder))
		subprocess.run(["make","-C",project_path+"ColliderSingleCore"], check=True)
	except:
		print('compilation failed')
		exit(-1)
		
	job_set_name = "lognorm_radius_test"
	job_set_name = "restartTest"
	job_set_name = "test"

	# folder_name_scheme = "T_"

	runs_at_once = 7
	# attempts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] 
	attempts = [1] 
	N = [5]
	# Temps = [3,10,30,100,300,1000]
	Temps = [3]
	folders = []
	for attempt in attempts:
		for n in N:
			for Temp in Temps:
				# job = curr_folder + 'jobs/' + job_set_name + str(attempt) + '/'
				job = project_path + 'jobs/' + job_set_name + str(attempt) + '/'\
							+ 'N_' + str(n) + '/' + 'T_' + str(Temp) + '/'
				if not os.path.exists(job):
					os.makedirs(job)
				else:
					print("Job '{}' already exists.".format(job))


				#load default input file
				with open(project_path+"default_files/default_input.json",'r') as fp:
					input_json = json.load(fp)

				####################################
				######Change input values here######
				input_json['temp'] = Temp
				input_json['seed'] = 'default'
				input_json['radiiDistribution'] = 'logNormal'
				input_json['h_min'] = 0.5
				# input_json['u_s'] = 0.5
				# input_json['u_r'] = 0.5
				input_json['note'] = "Does this work at all?"
				####################################

				with open(job + "input.json",'w') as fp:
					json.dump(input_json,fp,indent=4)

				#add run script and executable to folders
				os.system(f"cp {project_path}default_files/run_sim.py {job}run_sim.py")
				os.system(f"cp {project_path}ColliderSingleCore/ColliderSingleCore.x {job}ColliderSingleCore.x")
				folders.append(job)
	# print(folders)
	if len(N) != len(folders):
		N = [str(N[0]) for i in range(len(folders))]

	inputs = list(zip(folders,N))
	print(inputs)


	# for i in range(0,len(folders),runs_at_once):
	# 	with mp.Pool(processes=runs_at_once) as pool:
	# 		pool.starmap(run_job,inputs[i:i+runs_at_once]) 
	with mp.Pool(processes=runs_at_once) as pool:
		for i in range(0,len(folders)):
			# input_data = inputs[i:i+runs_at_once]
			pool.apply_async(run_job,inputs[i]) 

		pool.close()
		pool.join()


	
