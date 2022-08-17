import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/kolanzl/Desktop/SpaceLab")
import utils as u

# def make_vox_rep(data_folder,vox_size):


# def voxelize(data_folder,min_vox_size):
# 	dmgr = u.datamgr(data_folder)
# 	dmgr.vox_me_bro(num_vox)

# 	exit(0)
	

if __name__ == '__main__':
	data_prefolder = '/home/kolanzl/Desktop/SpaceLab/jobs/tempVariance_attempt'
	temps = [3,10,30,100,300,1000]
	attempts = [2]

	min_vox_size = 0.0000001
	
	for attempt in attempts:
		for temp in temps:
			data_folder = data_prefolder + str(attempt) + '/' + 'T_' + str(temp) + '/'
			vox = u.voxelize(data_folder,min_vox_size)
			vox.write_book()
			exit(0)