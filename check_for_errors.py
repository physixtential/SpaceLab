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
		# print("balls in sim                : {}".format(balls))
		# print("balls that should be in sim : {}".format(max_ind+3))
		# print("initially specified N value : {}".format(correct_N))
		return True

def where_did_error1_start(fullpath):
	directory = os.fsencode(fullpath)
	min_ind=9999999999
	min_balls = 9999999999
	test_file = ''
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.endswith("constants.csv") and filename[2] != "R": 
			index = int(filename.split('_')[0])
			with open(fullpath+filename, 'r') as fp: #number of lines in this file is the number of balls in sim
			    for count, line in enumerate(fp):
			        pass
			balls = count+1 #IDK why you need to add one but this method doesn't count every line, it misses one at the beginning or end
			if (balls != index+3):
				if (index < min_ind):
					min_balls = balls
					min_ind = index
					test_file = filename

	print(test_file)
	print(min_ind)
	print(min_balls)

	


def main():

	curr_folder = os.getcwd() + '/'


	job_set_name = "lognorm"

	attempts = [i for i in range(10)]

	attempts = [0]
	attempts_300 = attempts


	N = [30,100,300]
	N=[100]

	Temps = [3,10,30,100,300,1000]
	Temps = [3]

	error_folders = []
	for n in N:
		for Temp in Temps:
			temp_attempt = attempts
			if n == 300:
				temp_attempt = attempts_300
			for attempt in temp_attempt:
				job = curr_folder + 'jobs/' + job_set_name + str(attempt) + '/'\
							+ 'N_' + str(n) + '/' + 'T_' + str(Temp) + '/'
				where_did_error1_start(job)

				# if os.path.exists(job):
				# 	if error1(job,n):
				# 		error_folders.append(job)
				# else:
				# 	print("Folder doesn't exist: {}".format(job))

	print(error_folders)


if __name__ == '__main__':
	main()