import numpy as np
import os,glob
import sys
# cwd = os.getcwd()
# os.system("cd /home/kolanzl/Open3D/build")
# sys.path.append("/home/kolanzl/Open3D/build/lib/python_package/open3d")
import open3d as o3d
# os.system("cd {}".format(cwd))

data_columns = 11

def get_data_file(data_folder):
	files = os.listdir(data_folder) 
	file_indicies = np.array([file.split("_")[0] for file in files\
							if file.endswith('simData.csv')],dtype=np.int64)
	max_index = np.max(file_indicies)

	data_file = [file for file in files \
				if file.endswith("simData.csv") and file.startswith(str(max_index))]
	
	if len(data_file) == 1:
		return data_file[0]
	else:
		print("data file in folder '{}' not found.".format(data_folder))
		print("Now exiting.")
		exit(-1)

def get_last_line_data(data_folder):
	# data_headers = np.loadtxt(data_folder + data_file,skiprows=0,dtype=str,delimiter=',')[0]
	data_file = get_data_file(data_folder)
	data = np.loadtxt(data_folder + data_file,skiprows=1,dtype=float,delimiter=',')[-1]
	return format_data(data)

def get_constants(data_folder):
	data_file = get_data_file(data_folder)
	data_file = data_file.replace('simData','constants')	
	data_constants = np.loadtxt(data_folder+data_file,skiprows=0,dtype=float,delimiter=',')[0]
	return data_constants[0],data_constants[1],data_constants[2]

def format_data(data):
	data = np.reshape(data,(int(data.size/data_columns),data_columns))
	data = data[:,:3]
	return data

def get_data(data_folder):
	data_file = get_data_file(data_folder)
	radius,mass,moi = get_constants(data_folder)
	data = get_last_line_data(data_folder)
	return data,radius,mass,moi

def get_data_range(data_folder):
	data = get_last_line_data(data_folder)
	radius,m,moi = get_constants(data_folder)

	max_x = np.max(data[:,0]) + radius
	min_x = np.min(data[:,0]) - radius
	max_y = np.max(data[:,1]) + radius
	min_y = np.min(data[:,1]) - radius
	max_z = np.max(data[:,2]) + radius
	min_z = np.min(data[:,2]) - radius
	
	return max_x,min_x,max_y,min_y,max_z,min_z




class datamgr(object):
	"""docstring for datamgr"""
	def __init__(self, data_folder,voxel_buffer=5,ppb=1000):
		super(datamgr, self).__init__()
		self.data_folder = data_folder
		#how many points in single ball pointcloud
		self.ppb = ppb
		self.data,self.radius,self.mass,self.moi = get_data(self.data_folder)
		self.nBalls = self.data.shape[0]
		self.buffer = voxel_buffer # how many extra voxels in each direction 

	def vox_init(self,num_vox):
		data_range = get_data_range(self.data_folder)
		data_abs_max = max(data_range,key=abs) 
		self.vox_size = (data_abs_max*2)/num_vox
		self.vox_rep = np.zeros((num_vox+self.buffer*2,num_vox+self.buffer*2,num_vox+self.buffer*2))

	#evenly spaced points on sphere code form:
	#http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
	def gen_pt_cloud(self,pt):
		goldenRatio = (1 + 5**0.5)/2
		i = np.arange(0, self.ppb)
		theta = 2 * np.pi * i / goldenRatio
		phi = np.arccos(1 - 2*(i+0.5)/self.ppb)

		return_array = np.zeros((self.ppb,3))
		return_array[:,0] = self.radius*(np.cos(theta) * np.sin(phi)) + pt[0]
		return_array[:,1] = self.radius*(np.sin(theta) * np.sin(phi)) + pt[1]
		return_array[:,2] = self.radius*np.cos(phi) + pt[2]
		
		return return_array
					

	def vox_me_bro(self,num_vox):
		self.point_cloud = np.zeros((self.nBalls*self.ppb,3))
		self.vox_init(num_vox)

		for ind,pt in enumerate(self.data):
			# self.gen_pt_cloud(pt)
			self.point_cloud[ind*self.ppb:(ind+1)*self.ppb] = self.gen_pt_cloud(pt)
		pcd = o3d.geometry.PointCloud()
		# Add the points, colors and normals as Vectors
		pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
		# print(self.data.shape)
		# colors = np.random.rand(self.data.shape[0],self.data.shape[1])	
		# exit(0)
		pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(self.data.shape[0], 3)))
		voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=self.vox_size)
		vis = o3d.visualization.Visualizer()
		# Create a window, name it and scale it
		vis.add_geometry(voxel_grid)
		print(vis.create_window(window_name='VoxVis', width=800, height=600))

		# # Add the voxel grid to the visualizer

		# # We run the visualizater
		# vis.run()
		# # Once the visualizer is closed destroy the window and clean up
		# vis.destroy_window()

				
		
	