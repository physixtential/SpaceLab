import os
import json
import multiprocessing as mp
import subprocess
import numpy as np

def run_job(location,num_balls):
	cmd = ["python3", "{}run_multicore_sim.py".format(location), location, str(num_balls)]
	# print(cmd)
	subprocess.run(cmd)

if __name__ == '__main__':
	#make new output folders
	curr_folder = os.getcwd() + '/'

	try:
		# os.chdir("{}ColliderSingleCore".format(curr_folder))
		subprocess.run(["make","-C","ColliderMultiCore"], check=True)
	except:
		print('compilation failed')
		exit(-1)
		
	# job_set_name = "openMPallLoops"
	# job_set_name = "strongScaleCollide"
	# job_set_name = "weakScaleGrowth"
	job_set_name = "weakScaleCollide"
	# job_set_name = "pipeAndOpenmp"
	# job_set_name = "smallerDt"
	# job_set_name = "forceTest"
	# folder_name_scheme = "T_"

	runs_at_once = 1
	# attempts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] 
	attempts = [1]
	threads = [1,2,4,8,16,32,64]
	Nsquare = [2,4,8,16,32,64,128]
	N = [1,2,3,4,6,8,11]
	N = [i+299 for i in N]
	Temps = [100]
	folders = []
	for a,attempt in enumerate(attempts):
		# for n in N:
		for Temp in Temps:
			# job = curr_folder + 'jobs/' + job_set_name + str(attempt) + '/'
			job = curr_folder + 'jobs/' + job_set_name + str(attempt) + '/'\
							+ 'thread_' + str(thread) + '/'
			if not os.path.exists(job):
				os.makedirs(job)
			else:
				print("Job '{}' already exists.".format(job))

			# os.system("cp {}/jobs/collidable_aggregate/* {}".format(curr_folder,job))

			#load default input file
			with open(curr_folder+"default_files/default_input.json",'r') as fp:
				input_json = json.load(fp)

			####################################
			######Change input values here######
			input_json['temp'] = Temps[0]
			input_json['seed'] = 101
			input_json['radiiDistribution'] = 'constant'
			input_json['N'] = N[a]
			input_json['simType'] = "BPCA"
			input_json['OMPthreads'] = attempt
			# input_json['kConsts'] = 3e3
			input_json['h_min'] = 0.5
			# input_json['u_s'] = 0.5
			# input_json['u_r'] = 0.5
			# input_json['projectileName'] = "299_2_R4e-05_v4e-01_cor0.63_mu0.1_rho2.25_k4e+00_Ha5e-12_dt5e-10_"
			# input_json['targetName'] = "299_2_R4e-05_v4e-01_cor0.63_mu0.1_rho2.25_k4e+00_Ha5e-12_dt5e-10_"
			# input_json['note'] = "Uses openmp and loop unwinding to parallelize sim_one_step."
			input_json['note'] = "Weak scaling with {} OMP threads.".format(attempt)
			####################################

			with open(job + "input.json",'w') as fp:
				json.dump(input_json,fp,indent=4)


			time = np.ceil(N[a]/15)

			sbatchfile = ""
			sbatchfile += "#!/bin/bash\n"
			sbatchfile += "#SBATCH -A m4189\n"
			sbatchfile += "#SBATCH -C gpu\n"
			sbatchfile += "#SBATCH -q regular\n"
			sbatchfile += "#SBATCH -t {}:00:00\n".format(int(time))
			sbatchfile += "#SBATCH -n 1\n"
			# sbatchfile += "#SBATCH -c 64\n\n"
			sbatchfile += 'module load gpu\n'
			batchfile += 'export OMP_NUM_THREADS={}\n'.format(thread)
			sbatchfile += 'export SLURM_CPU_BIND="cores"\n'
			sbatchfile += "srun -c {} --cpu-bind=cores ./ColliderMultiCore.x {} 2>sim_err.log 1>sim_out.log".format(thread*2,job)
			
			with open(job+"sbatchMulti.bash",'w') as sfp:
				sfp.write(sbatchfile)

			#add run script and executable to folders
			os.system("cp default_files/run_multicore_sim.py {}run_multicore_sim.py".format(job))
			os.system("cp ColliderMultiCore/ColliderMultiCore.x {}ColliderMultiCore.x".format(job))
			os.system("cp ColliderMultiCore/ColliderMultiCore.cpp {}ColliderMultiCore.cpp".format(job))
			os.system("cp ColliderMultiCore/ball_group_multi_core.hpp {}ball_group_multi_core.hpp".format(job))
			folders.append(job)
	# print(folders)
	# if len(N) != len(folders):
		# N = [str(N[0]) for i in range(len(folders))]

	# inputs = list(zip(folders,N))
	cwd = os.getcwd()
	print(folders)
	for folder in folders:
		os.chdir(folder)
		os.system("sbatch sbatchMulti.bash")
	os.chdir(cwd)
	# for i in range(0,len(folders),runs_at_once):
	# 	with mp.Pool(processes=runs_at_once) as pool:
	# 		pool.starmap(run_job,folders[i:i+runs_at_once]) 


	