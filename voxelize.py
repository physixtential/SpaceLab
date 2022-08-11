import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/kolanzl/Desktop/SpaceLab")
import utils as u

# def make_vox_rep(data_folder,vox_size):


def voxelize(data_folder,num_vox):
	dmgr = u.datamgr(data_folder)
	dmgr.vox_me_bro(num_vox)

	exit(0)
	

if __name__ == '__main__':
	data_prefolder = '/home/kolanzl/Desktop/SpaceLab/jobs/tempVariance_attempt'
	temps = [3,10,30,100,300,1000]
	attempts = [2]

	num_vox = 1000
	
	for attempt in attempts:
		for temp in temps:
			data_folder = data_prefolder + str(attempt) + '/' + 'T_' + str(temp) + '/'
			voxelize(data_folder,num_vox)