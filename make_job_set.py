import os
import json


def main():
	#make new output folders
	curr_folder = os.getcwd() + '/'
	job_set_name = "TESTSET"
	folder_name_scheme = "T_"

	job_set_folder = curr_folder + 'jobs/' + job_set_name + '/'
	if not os.path.exists(job_set_folder):
		os.mkdir(job_set_folder)
	else:
		print("Job set '{}' already exists.".format(job_set_folder))

	#Make an array of what you want to vary
	Temps = [3]

	folder_values = Temps
	folders = [job_set_folder + folder_name_scheme + str(val) + '/' for val in folder_values]
	for new_folder in folders:
		if not os.path.exists(new_folder):
			os.mkdir(new_folder)
		else:
			print("Job '{}' already exists.".format(new_folder))

	with open(curr_folder+"default_files/default_input.json",'r') as fp:
		input_json = json.load(fp)

	#loop through folders and generate input file and run_script.py for each
	#changing the necessary values in the input file
	for index,value in enumerate(folder_values):
		input_json['Temp'] = value
		with open(folders[index] + "input.json",'w') as fp:
			json.dump(input_json,fp,indent=4)

		os.system("cp default_files/run_sim.py {}run_sim.py".format(folders[index]))
		os.system("cp ColliderSingleCore/ColliderSingleCore.o {}ColliderSingleCore.o".format(folders[index]))

	



	#add run script to folders

if __name__ == '__main__':
	main()