import os
import json
import multiprocessing as mp
import subprocess

def run_job(location,num_balls):
	cmd = ["python3", "{}run_sim.py".format(location), location, str(num_balls)]
	# print(cmd)
	subprocess.run(cmd)

if __name__ == '__main__':
	#make new output folders
	curr_folder = os.getcwd() + '/'

	try:
		# os.chdir("{}ColliderSingleCore".format(curr_folder))
		subprocess.run(["make","-C","ColliderSingleCore"], check=True)
	except:
		print('compilation failed')
		exit(-1)
		
	job_set_name = "restartTest"

	# folder_name_scheme = "T_"

	runs_at_once = 1
	# attempts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] 
	attempts = [1] 
	N = [50]
	# Temps = [3,10,30,100,300,1000]
	Temps = [3]
	folders = []
	for attempt in attempts:
		for n in N:
			for Temp in Temps:
				job = curr_folder + 'jobs/' + job_set_name + str(attempt) + '/'
				# job = curr_folder + 'jobs/' + job_set_name + str(attempt) + '/'\
							# + 'N_' + str(n) + '/' + 'T_' + str(Temp) + '/'
				if not os.path.exists(job):
					os.makedirs(job)
				else:
					print("Job '{}' already exists.".format(job))


				#load default input file
				with open(curr_folder+"default_files/default_input.json",'r') as fp:
					input_json = json.load(fp)

				####################################
				######Change input values here######
				input_json['temp'] = Temp
				input_json['seed'] = 101
				input_json['radiiDistribution'] = 'logNormal'
				input_json['h_min'] = 0.5
				# input_json['u_s'] = 0.5
				# input_json['u_r'] = 0.5
				input_json['note'] = "Restart test because it dont work no mo"
				####################################

				with open(job + "input.json",'w') as fp:
					json.dump(input_json,fp,indent=4)

				#add run script and executable to folders
				os.system("cp default_files/run_sim.py {}run_sim.py".format(job))
				os.system("cp ColliderSingleCore/ColliderSingleCore.x {}ColliderSingleCore.x".format(job))

				# os.system("cp /mnt/be2a0173-321f-4b9d-b05a-addba547276f/kolanzl/SpaceLab_stable/SpaceLab/jobs/lognorm0/N_100/T_3/3* {}".format(job))
				# os.system("cp /mnt/be2a0173-321f-4b9d-b05a-addba547276f/kolanzl/SpaceLab_stable/SpaceLab/jobs/lognorm0/N_100/T_3/4* {}".format(job))

				folders.append(job)
	# print(folders)
	if len(N) != len(folders):
		N = [str(N[0]) for i in range(len(folders))]

	inputs = list(zip(folders,N))
	print(inputs)


	with mp.Pool(processes=runs_at_once) as pool:
		for i in range(0,len(folders)):
			# input_data = inputs[i:i+runs_at_once]
			pool.apply_async(run_job,inputs[i]) 

		pool.close()
		pool.join()


	
