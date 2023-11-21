##restart jobs that have errored

#Error 1: restart failed, number of balls didn't carry over so there are an incorrect 
#			number of colums in at least 1 sim_data output
#Error 2: sim fails to write all the data (so there aren't the correct number of colums in simData)
#Error *: possible signed integer over flow in number of steps for a sim
#			Indicated by a specific sequence at the end of sim_errors.txt
#Error general: did we get "Simulation complete!" within the last 10 lines of sim_error.log

import os
import glob
import numpy as np
import subprocess
import check_for_errors as cfe


def restart_job(folder,test=True,move_folder=''):

	if len(move_folder) > 0 #move data to new folder specified in move_folder
		if os.path.exists(move_folder): #if move_folder already exists
			#if it already exists then we need to change the name of it so it doesn't overwrite
			move_folder += '_MOVE-'
			move_index = 0
			while os.path.exists(move_folder+str(move_index)): #check for lowest number transpher. 
				move_index += 1

			move_folder += str(move_index) + '/'
			os.makedirs(move_folder)
		
		else : #make it
			os.makedirs(move_folder)

		command = f"mv {folder}* {move_folder}."
		if test:
			print(command)
		else:
			os.system(command)
	else: #remove and restart
		cwd = os.getcwd()
		os.chdir(folder)
		if test:
			command = "ls"
		else:
			command = "rm"

		os.system(f"{command} *.csv")
		os.system(f"{command} *.txt")
		os.system(f"{command} *.x")
		os.system(f"{command} *.py")
		os.system(f"{command} *.cpp")
		os.system(f"{command} *.hpp")
		os.chdir(cwd)

		try:
			# os.chdir("{}ColliderSingleCore".format(curr_folder))
			subprocess.run(["make","-C","ColliderSingleCore"], check=True)
		except Exception as e:
			print('compilation failed')
			print(e)
			exit(-1)

		if not test:
			os.system("cp default_files/run_sim.py {}run_sim.py".format(folder))
			os.system("cp ColliderSingleCore/ColliderSingleCore.x {}ColliderSingleCore.x".format(folder))
			os.system("cp ColliderSingleCore/ColliderSingleCore.cpp {}ColliderSingleCore.cpp".format(folder))
			os.system("cp ball_group.hpp {}ball_group.hpp".format(folder))

		cwd = os.getcwd()
		os.chdir(folder)
		if not test:
			os.system('qsub qsub.bash')
		else:
			os.system('ls qsub.bash')
			os.system('ls input.json')
		os.chdir(cwd)

def main():

	curr_folder = os.getcwd() + '/'

	job = curr_folder + 'jobs/lognorm$a$/N_$n$/T_$t$/'
	move_folder = curr_folder + 'erroredJobs/lognorm$a$/N_$n$/T_$t$/'



	attempts = [i for i in range(30)]
	# attempts = [0]

	N = [30,100,300]
	# N=[30]

	Temps = [3,10,30,100,300,1000]
	# Temps = [3]

	# errorgen_folders = check_error(job,error_general,N,Temps,attempts)
	# error1_folders = check_error(job,error1,N,Temps,attempts)
	# print(error1_folders)

	error2_folders = cfe.check_error(job,cfe.error2,N,Temps,attempts)
	for folder in error2_folders:
		# print(folder)
		restart_job(folder,test=True,move_folder=move_folder)
		# exit(0)
	# print(error2_folders)


if __name__ == '__main__':
	main()