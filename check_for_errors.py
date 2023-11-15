##Check output of simulations for errors

#Error 1: restart failed, number of balls didn't carry over so there are an incorrect 
#			number of colums in at least 1 sim_data output
#Error 2: possible signed integer over flow in number of steps for a sim
#			Indicated by a specific sequence at the end of sim_errors.txt

import os

#If fullpath has error1 in it, return 1, if not return 0
def error1(fullpath,correct_N):
	directory = os.fsencode(fullpath)
	max_ind=-1
	test_file = ''
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.endswith("constants.csv"): 

			index = int(filename.split('_')[0])
			if (index > max_ind):
				max_ind = index
				test_file = filename
	# print(fullpath+test_file)

	if (not test_file): #if test_file is empty string the sim hasnt started yet
		return False

	with open(fullpath+test_file, 'r') as fp: #number of lines in this file is the number of balls in sim
	    for count, line in enumerate(fp):
	        pass
	balls = count+1 #IDK why you need to add one but this method doesn't count every line, it misses one at the beginning or end

	if balls == max_ind+3: ### THIS IS SPECIFIC TO BPCA GROWTH RUNS
		return False
	else:
		return True




def main():

	curr_folder = os.getcwd() + '/'


	job_set_name = "lognorm"

	attempts = [i for i in range(10)]

	# attempts = [1]
	attempts_300 = attempts


	N = [30,100,300]
	# N=[30]

	Temps = [3,10,30,100,300,1000]
	# Temps = [3]

	error_folders = []
	for n in N:
		for Temp in Temps:
			temp_attempt = attempts
			if n == 300:
				temp_attempt = attempts_300
			for attempt in temp_attempt:
				job = curr_folder + 'jobs/' + job_set_name + str(attempt) + '/'\
							+ 'N_' + str(n) + '/' + 'T_' + str(Temp) + '/'
				if os.path.exists(job):
					if error1(job,n):
						error_folders.append(job)
				else:
					print("Folder doesn't exist: {}".format(job))

	print(error_folders)


if __name__ == '__main__':
	main()