import os
import sys

def main():
	# os.system("")
	curr_folder = os.getcwd() + '/'
	output_file = "sim_output.txt"
	error_file = "sim_errors.txt"
	out = os.system("./ColliderSingleCore.o {}".format(curr_folder))
	# out = os.system("./ColliderSingleCore.o {} 1>> {} 2>> {}".format(curr_folder,output_file,error_file))
	print(out)

if __name__ == '__main__':
	main()