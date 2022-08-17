import numpy as np
import os,glob
import sys
import matplotlib.pyplot as plt
from treelib import Node, Tree
# cwd = os.getcwd()
# os.system("cd /home/kolanzl/Open3D/build")
# sys.path.append("/home/kolanzl/Open3D/build/lib/python_package/open3d")
import open3d as o3d
##include <pybind11/stl.h>`? Or <pybind11/complex.h>,
# <pybind11/functional.h>, <pybind11/chrono.h>

data_columns = 11

def plot(verts,center,radius):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	# print('')
	# print(verts)
	# print(verts[:][0])
	# exit(0)
	for i in range(len(verts)):
		print(verts[i],i)
		ax.scatter(verts[i][0],verts[i][1],verts[i][2],marker='*',color='b')
	ax.scatter(center[0],center[1],center[2],marker='.',color='r')
	ax.set_xlabel('X (row)')
	ax.set_ylabel('Y (col)')
	ax.set_zlabel('Z (dep)')

	# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
	# x = np.cos(u)*np.sin(v) - center[0]
	# y = np.sin(u)*np.sin(v) - center[1]
	# z = np.cos(v) - center[2]
	# ax.plot_wireframe(x, y, z, color="r")
	plt.show()

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

#following functions taken from 
#http://www.open3d.org/docs/release/tutorial/geometry/voxelization.html#Voxel-carving
def xyz_spherical(xyz):
	x = xyz[0]
	y = xyz[1]
	z = xyz[2]
	r = np.sqrt(x * x + y * y + z * z)
	r_x = np.arccos(y / r)
	r_y = np.arctan2(z, x)
	return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
	rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
						[0, np.sin(r_x), np.cos(r_x)]])
	rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
						[-np.sin(r_y), 0, np.cos(r_y)]])
	return rot_y.dot(rot_x)


def get_extrinsic(xyz):
	rvec = xyz_spherical(xyz)
	r = get_rotation_matrix(rvec[1], rvec[2])
	t = np.asarray([0, 0, 2]).transpose()
	trans = np.eye(4)
	trans[:3, :3] = r
	trans[:3, 3] = t
	return trans


def preprocess(model):
	min_bound = model.get_min_bound()
	max_bound = model.get_max_bound()
	center = min_bound + (max_bound - min_bound) / 2.0
	scale = np.linalg.norm(max_bound - min_bound) / 2.0
	# scale = 1
	vertices = np.asarray(model.vertices)
	vertices -= center
	model.vertices = o3d.utility.Vector3dVector(vertices / scale)
	return model

def preprocess_pt(model):
	min_bound = model.get_min_bound()
	max_bound = model.get_max_bound()
	print("max: {}\tmin: {}".format(max_bound,min_bound))
	center = min_bound + (max_bound - min_bound) / 2.0
	scale = np.linalg.norm(max_bound - min_bound) / 2.0
	# scale = 1
	vertices = np.asarray(model.points)
	vertices -= center
	model.points = o3d.utility.Vector3dVector(vertices / scale)
	return model


def vox_carve(mesh,
				  cubic_size,
				  voxel_resolution,
				  w=300,
				  h=300,
				  use_depth=True,
				  surface_method='pointcloud'):
	# mesh.compute_vertex_normals()
	mesh.estimate_normals()
	camera_sphere = o3d.geometry.TriangleMesh.create_sphere()

	# setup dense voxel grid
	voxel_carving = o3d.geometry.VoxelGrid.create_dense(
		width=cubic_size,
		height=cubic_size,
		depth=cubic_size,
		voxel_size=cubic_size / voxel_resolution,
		origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
		color=[1.0, 0.7, 0.0])
	print("Vox size: {}".format(cubic_size / voxel_resolution))
	# rescale geometry
	camera_sphere = preprocess(camera_sphere)
	mesh = preprocess_pt(mesh)

	# setup visualizer to render depthmaps
	vis = o3d.visualization.Visualizer()
	vis.create_window(width=w, height=h, visible=False)
	vis.add_geometry(mesh)
	vis.get_render_option().mesh_show_back_face = True
	ctr = vis.get_view_control()
	param = ctr.convert_to_pinhole_camera_parameters()

	# carve voxel grid
	pcd_agg = o3d.geometry.PointCloud()
	centers_pts = np.zeros((len(camera_sphere.vertices), 3))
	for cid, xyz in enumerate(camera_sphere.vertices):
		# get new camera pose
		trans = get_extrinsic(xyz)
		param.extrinsic = trans
		c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
		centers_pts[cid, :] = c[:3]
		ctr.convert_from_pinhole_camera_parameters(param)

		# capture depth image and make a point cloud
		vis.poll_events()
		vis.update_renderer()
		depth = vis.capture_depth_float_buffer(False)
		pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(
			o3d.geometry.Image(depth),
			param.intrinsic,
			param.extrinsic,
			depth_scale=1)

		# depth map carving method
		if use_depth:
			voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
		else:
			voxel_carving.carve_silhouette(o3d.geometry.Image(depth), param)
		# print("Carve view %03d/%03d" % (cid + 1, len(camera_sphere.vertices)))
	vis.destroy_window()

	# add voxel grid survace
	print('Surface voxel grid from %s' % surface_method)
	if surface_method == 'pointcloud':
		voxel_surface = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
			pcd_agg,
			voxel_size=cubic_size / voxel_resolution,
			min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
			max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
	elif surface_method == 'mesh':
		voxel_surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
			mesh,
			voxel_size=cubic_size / voxel_resolution,
			min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
			max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
	else:
		raise Exception('invalid surface method')
	voxel_carving_surface = voxel_surface + voxel_carving

	return voxel_carving_surface, voxel_carving, voxel_surface

def dist(pt1,pt2):
	return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)


class datamgr(object):
	"""docstring for datamgr"""
	# def __init__(self, data_folder,voxel_buffer=5,ppb=3000):
	def __init__(self, data_folder,ppb=300000):
		super(datamgr, self).__init__()
		self.data_folder = data_folder
		#how many points in single ball pointcloud
		self.ppb = ppb
		self.data,self.radius,self.mass,self.moi = get_data(self.data_folder)
		self.nBalls = self.data.shape[0]
		# self.buffer = voxel_buffer # how many extra voxels in each direction 

	def shift_to_first_quad(self,data_range=None):
		if data_range is None:
			data_range = get_data_range(self.data_folder)

		self.data[:,0] -= data_range[1] 
		self.data[:,1] -= data_range[3] 
		self.data[:,2] -= data_range[5] 


	def vox_init(self,num_vox):
		self.data_range = get_data_range(self.data_folder)
		data_abs_max = max(self.data_range,key=abs) 
		self.vox_size = (data_abs_max*2)/num_vox
		# self.vox_size = 
		# print(self.vox_per_radius)
		# self.vox_rep = np.zeros((num_vox+self.buffer*2,num_vox+self.buffer*2,num_vox+self.buffer*2))

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

	def get_center_pt(self,ind):
		return [ind[0]*self.radius+self.radius/2,ind[1]*self.radius+self.radius/2,ind[2]*self.radius+self.radius/2]

	

class voxelize(object):
	"""docstring for voxelize"""
	def __init__(self, data_folder,min_vox_size):
		super(voxelize, self).__init__()
		self.data_folder = data_folder
		self.dm = datamgr(self.data_folder)
		self.dm.shift_to_first_quad()
		self.min_vox_size = min_vox_size

		self.volume_elements = []
		
		self.data_range = get_data_range(self.data_folder)

		axes_len = [self.data_range[0]-self.data_range[1],self.data_range[2]-self.data_range[3],self.data_range[4]-self.data_range[5],]
		self.init_vox_size = np.max(axes_len)/2




		self.book = Tree()
		self.num_nodes = 0
		self.book.create_node(self.num_nodes,self.num_nodes)
		self.deepest_nodes = [self.book.get_node(self.num_nodes)]
		self.max_depth = self.book.depth()
		'''
		#The book implimentation stores too much redundent data
		book[chapter][sentence][word][letters]

		chapter: chapter is defined as its n value when it was written
		sentence: sentence is defined as a list of three words corresponding
					to the row, col, dep of a particular chapter.
					Basically sentence is what contains info needed to 
					determine where a box is.
					A sentence defines a colored box.
					The chapter a sentence is in defines the box size.
		word: string of 1's and 0's that define a chapter's containing boxes.
					A word can either describe a row, col, or dep.  
		letter: the 1's and 0's making up a word
		'''

		self.data_range = get_data_range(self.data_folder)

	#TODO
	def pt_to_ind(self,pt,vox_size):
		x_ind = np.ceil(pt[0]/vox_size)-1
		y_ind = np.ceil(pt[1]/vox_size)-1
		z_ind = np.ceil(pt[2]/vox_size)-1
		return (int(x_ind),int(y_ind),int(z_ind))

	def ind_to_pt(self,sentence):
		vertex = [0,0,0]
		vox_size = 0
		n=0
		for point in list(zip(*[iter(sentence)]*3)):
			for coord,letter in enumerate(point):
				# print(coord,letter)
				vox_size = self.init_vox_size/(2**n)
				if letter == '1':
					vertex[coord] += vox_size
			n += 1
			# exit(0)

		vertices = []
		for i in range(8):
			vertices.append(vertex.copy())
			
		vertices[1][0] += vox_size
		vertices[2][1] += vox_size
		vertices[3][2] += vox_size

		vertices[4][0] += vox_size
		vertices[4][1] += vox_size
		vertices[5][1] += vox_size
		vertices[5][2] += vox_size
		vertices[6][0] += vox_size
		vertices[6][2] += vox_size
		
		vertices[7][0] += vox_size
		vertices[7][1] += vox_size
		vertices[7][2] += vox_size
		
		return vertices


	def box_contains(self,sentence,pt,radius):
		box_verts = self.ind_to_pt(sentence)
		# print(box_verts)
		# exit(0)
		# print(pt,radius)
		# plot(box_verts,pt,radius)
		# exit(0)
		# sphere_bounding_verts = []
		# sphere_bounding_verts.append(pt[0]-radius)
		# sphere_bounding_verts.append(pt[0]+radius)
		# sphere_bounding_verts.append(pt[1]-radius)
		# sphere_bounding_verts.append(pt[1]+radius)
		# sphere_bounding_verts.append(pt[2]-radius)
		# sphere_bounding_verts.append(pt[2]+radius)

		center_to_vert_dists = []
		for vert in box_verts:
			center_to_vert_dists.append(dist(vert,pt))

		#if 1, that vertex is outside (or at) radius
		#if 0, that vertex is inside radius
		compare_radius = np.where(np.array(center_to_vert_dists)>=radius,1,0)
		#if all vertices are inside circle
		#i.e. circle completely encloses voxel
		if np.sum(compare_radius) == 0:
			# self.volume_elements.append(sentence)
			#return -1 so we can mark this node as a volume element
			return -1 #dont add to sentence since this is a volume element, dont need to see it again
		#if all vertices are outside circle
		elif np.sum(compare_radius) == compare_radius.shape:
			vox_size = self.init_vox_size/(2**self.max_depth)
			compare_vox = np.where(np.array(center_to_vert_dists)>vox_size,1,0)
			#is the circle contained in the box?
			if np.sum(compare_vox) == 0:
				return 1
			else:
				return 0
		else:
			return 1
		
	def write_chapter(self,n):
		vox_size = self.init_vox_size/(2**n)
		chapter = []
		# if self.num_nodes == 0:
		# 	max_depth = self.book.depth()
		# 	print(max_depth)
		# 	exit(0)
		# 	# previous_chapter = #Nodes in previous level
		# else:
		# 	previous_chapter = ['','','']
		# self.book.show()
		#get all nodes at lowest level
		# max_depth_nodes = [self.book[node].tag for node in \
			# self.book.expand_tree(filter = lambda x: \
			# self.book.depth(x) == self.book.depth())]
		# max_depth_nodes = list(self.book.filter_nodes(lambda x: self.book.depth(x) == self.book.depth())) 
		# print(list(self.book.filter_nodes(lambda x: self.book.depth(x) == self.book.depth())))
		# print(self.book.depth())
		# print(self.book.depth(self.book.get_node('1')))
		# print(self.book.depth(self.book.get_node('0')))
		# print('max depth nodes: ',max_depth_nodes)
		temp_deepest_nodes = []
		for node in self.deepest_nodes:
			word = ''	
			t_node = node
			# exit(0)
			while(t_node.tag != 0):
				
				# print("node",node,type(node))
				# print("t_node",t_node,type(node))
				word = t_node.data + word
				t_node = self.book.parent(t_node.tag)
				# print("t_node", t_node)
			# for previous_word in previous_sentence:
			# print("node", node)

			for row in [0,1]:
				for col in [0,1]:
					for dep in [0,1]:
						new_word = str(row) + str(col) + str(dep)
						for pt in self.dm.data:
							test_ind = word + new_word

							# sentence = previous_sentence.copy()
							# sentence[0] += str(row)
							# sentence[1] += str(col)
							# sentence[2] += str(dep)
							# print(sentence)
							# # exit(0)
							result = self.box_contains(test_ind,pt,self.dm.radius)
							if result == 1:
								#if true, add this node
								self.num_nodes += 1
								self.book.create_node(str(self.num_nodes),str(self.num_nodes),data=new_word,parent=node)
								temp_deepest_nodes.append(self.book.get_node(str(self.num_nodes)))
								break
							elif result == -1:
								self.num_nodes += 1
								#if true, this node is a volume element
								self.book.create_node(str(self.num_nodes)+"VOL",str(self.num_nodes)+"VOL",data=new_word,parent=node)
								break

			# self.book.show()

		# self.book.append(chapter)
		self.deepest_nodes = temp_deepest_nodes
		self.max_depth += 1
		# print("depth")
		return vox_size

	def write_book(self):
		curr_vox_size = self.init_vox_size
		n = 0
		while curr_vox_size > self.min_vox_size:# and n != 3:
			curr_vox_size = self.write_chapter(n)
			print(curr_vox_size,n)
			# print("depth",self.book.depth())
			n += 1

		# print(self.book)

# class datamgr(object):
# 	"""docstring for datamgr"""
# 	# def __init__(self, data_folder,voxel_buffer=5,ppb=3000):
# 	def __init__(self, data_folder,ppb=300000):
# 		super(datamgr, self).__init__()
# 		self.data_folder = data_folder
# 		#how many points in single ball pointcloud
# 		self.ppb = ppb
# 		self.data,self.radius,self.mass,self.moi = get_data(self.data_folder)
# 		self.nBalls = self.data.shape[0]
# 		# self.buffer = voxel_buffer # how many extra voxels in each direction 

# 	def vox_init(self,num_vox):
# 		self.data_range = get_data_range(self.data_folder)
# 		data_abs_max = max(self.data_range,key=abs) 
# 		self.vox_size = (data_abs_max*2)/num_vox
# 		# self.vox_size = 
# 		# print(self.vox_per_radius)
# 		# self.vox_rep = np.zeros((num_vox+self.buffer*2,num_vox+self.buffer*2,num_vox+self.buffer*2))

# 	#evenly spaced points on sphere code form:
# 	#http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
# 	def gen_pt_cloud(self,pt):
# 		goldenRatio = (1 + 5**0.5)/2
# 		i = np.arange(0, self.ppb)
# 		theta = 2 * np.pi * i / goldenRatio
# 		phi = np.arccos(1 - 2*(i+0.5)/self.ppb)

# 		return_array = np.zeros((self.ppb,3))
# 		return_array[:,0] = self.radius*(np.cos(theta) * np.sin(phi)) + pt[0]
# 		return_array[:,1] = self.radius*(np.sin(theta) * np.sin(phi)) + pt[1]
# 		return_array[:,2] = self.radius*np.cos(phi) + pt[2]
		
# 		return return_array

# 	def get_center_pt(self,ind):
# 		return [ind[0]*self.radius+self.radius/2,ind[1]*self.radius+self.radius/2,ind[2]*self.radius+self.radius/2]

# 	#Not tested
# 	def pt_to_ind(self,pt):
# 		x_ind = np.ceil(pt[0]/self.vox_size)-1
# 		y_ind = np.ceil(pt[1]/self.vox_size)-1
# 		z_ind = np.ceil(pt[2]/self.vox_size)-1
# 		return (int(x_ind),int(y_ind),int(z_ind))

# 	#Not tested
# 	def ind_to_pt(self,ind):
# 		vertex1 = (self.vox_size*ind[0],self.vox_size*ind[1],self.vox_size*ind[2])
# 		vertex2 = (self.vox_size*(ind[0]+1),self.vox_size*ind[1],self.vox_size*ind[2])
# 		vertex3 = (self.vox_size*ind[0],self.vox_size*(ind[1]+1),self.vox_size*ind[2])
# 		vertex4 = (self.vox_size*ind[0],self.vox_size*ind[1],self.vox_size*(ind[2]+1))
# 		vertex5 = (self.vox_size*(ind[0]+1),self.vox_size*(ind[1]+1),self.vox_size*ind[2])
# 		vertex6 = (self.vox_size*ind[0],self.vox_size*(ind[1]+1),self.vox_size*(ind[2]+1))
# 		vertex7 = (self.vox_size*(ind[0]+1),self.vox_size*ind[1],self.vox_size*(ind[2]+1))
# 		vertex8 = (self.vox_size*(ind[0]+1),self.vox_size*(ind[1]+1),self.vox_size*(ind[2]+1))

# 		return [vertex1,vertex2,vertex3,vertex4,vertex5,vertex6,vertex7,vertex8]

# 	#gotta write my own voxelization algorithm
# 	#open3d uses too much memory and has a memory leak
# 	#num_vox is number of voxels across largest dimension //TODO
# 	# def vox_me_bro(self,num_vox=500):
# 	# 	self.vox_init(num_vox)
# 	# 	self.vox_array = np.zeros((num_vox,num_vox,num_vox),dtype=np.uint8)
# 	# 	# print(self.data_range)
		
	# 	self.data[:,0] -= self.data_range[1] 
	# 	self.data[:,1] -= self.data_range[3] 
	# 	self.data[:,2] -= self.data_range[5] 


	# 	vox_in_radius = np.ceil(self.radius/self.vox_size)
	# 	print(self.radius)
	# 	print(self.vox_size)
	# 	print(vox_in_radius)
	# 	for pt in self.data:
	# 		ind = self.pt_to_ind(pt)
	# 		# box_bounds = [[ind[0]-vox_in_radius,ind[0]+vox_in_radius],[ind[1]-vox_in_radius,ind[1]+vox_in_radius],[ind[2]-vox_in_radius,ind[2]+vox_in_radius]] 
	# 		buffer = 2
	# 		box_bounds = [[ind[0]-vox_in_radius-buffer,ind[0]+vox_in_radius+buffer],[ind[1]-vox_in_radius-buffer,ind[1]+vox_in_radius+buffer],[ind[2]-vox_in_radius-buffer,ind[2]+vox_in_radius+buffer]] 
	# 		box_bounds = np.array(box_bounds,dtype=int)
	# 		print(box_bounds)
	# 		for x in range(box_bounds[0][0],box_bounds[0][1]):
	# 			for y in range(box_bounds[1][0],box_bounds[1][1]):
	# 				for z in range(box_bounds[2][0],box_bounds[2][1]):
	# 					center_pt = self.get_center_pt(ind)
	# 					print(center_pt,dist(center_pt,pt),self.radius)
	# 					for vert in self.ind_to_pt(ind):
	# 						if dist(vert,pt) < self.radius and self.vox_array[x,y,z] == 0:
	# 							self.vox_array[x,y,z] = 1
	# 							break

	# 		print(np.sum(self.vox_array))
	# 		print(ind)
	# 		exit(0)
	# 		self.vox_array[self.pt_to_ind(pt)] += 1

		# np.fromiter((self))


		# print("HERE")
		# ax = plt.figure().add_subplot(projection='3d')
		# print("HERE1")
		# ax.voxels(self.vox_array, edgecolor='k')
		# print("HERE2")

		# plt.show()



	#adapted from
	#https://towardsdatascience.com/how-to-voxelize-meshes-and-point-clouds-in-python-ca94d403f81d
	# def vox_me_bro(self,num_vox=5000):
	# 	num_vox = 5000

	# 	# self.point_cloud = np.zeros((2*self.ppb,3))
	# 	self.point_cloud = np.zeros((self.nBalls*self.ppb,3))
	# 	# self.point_cloud = np.zeros((self.ppb,3))
	# 	self.vox_init(num_vox)

	# 	# colors = np.random.rand((self.ppb,3))	
	# 	pcd = o3d.geometry.PointCloud()
	# 	for ind,pt in enumerate(self.data):
	# 		# self.gen_pt_cloud(pt)
	# 		self.point_cloud[ind*self.ppb:(ind+1)*self.ppb] = self.gen_pt_cloud(pt)
	# 		# Add the points, colors and normals as Vectors
	# 	# pcd.points = o3d.utility.Vector3dVector(self.gen_pt_cloud(pt))
	# 	pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
	# 	print(pcd)
	# 	# exit(0)
	# 	pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(self.ppb*self.nBalls, 3)))
		
	# 	print("HERE1")

	# 	# o3d.visualization.draw_geometries([pcd])
	# 	# tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
	# 	print("HERE2")
	# 	alpha = 0.03
	# 	# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
	# 	# mesh.compute_vertex_normals()
	# 	# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
	# 	# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
	# 	# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha,tetra_mesh,pt_map)
	# 	print("HERE3")
	# 	# mesh.compute_vertex_normals()

	# 	# self.point_cloud[ind*self.ppb:(ind+1)*self.ppb] = self.gen_pt_cloud(pt)
	# 	voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=self.vox_size)
	# 	print(voxel_grid)
	# 	# voxel_grid = o3d.geometry.VoxelGrid.create_dense(voxel_grid)
	# 	# print(voxel_grid.get_voxels)
		

	# 	vis = o3d.visualization.Visualizer()
	# 	# # Create a window, name it and scale it
	# 	vis.create_window(window_name='VoxVis', width=800, height=600)
	# 	# vis.draw_geometries([mesh], mesh_show_back_face=True)

	# 	# # # Add the voxel grid to the visualizer
	# 	vis.add_geometry(voxel_grid)

	# 	# # # We run the visualizater
	# 	vis.run()
	# 	# # # Once the visualizer is closed destroy the window and clean up
	# 	vis.destroy_window()
	# 	exit(0)


	#adapted from
	#https://towardsdatascience.com/how-to-voxelize-meshes-and-point-clouds-in-python-ca94d403f81d
	# def vox_me_bro1(self,vox_res=50,cubic_size=1.0):
	# 	self.point_cloud = np.zeros((self.nBalls*self.ppb,3))
	# 	# self.point_cloud = np.zeros((self.ppb,3))
	# 	# self.vox_init(num_vox)

	# 	# colors = np.random.rand((self.ppb,3))	

	# 	#check if pointcloud exists
	# 	vox_file = self.data_folder + "voxFile.out"
	# 	# vox_file = self.data_folder + "voxFile.pts"
	# 	# if os.path.isfile(vox_file):
	# 	if False:
	# 		# pcd = o3d.io.read_point_cloud(vox_file)
	# 		loaded_arr = np.loadtxt(vox_file) 
	# 		voxel_array = loaded_arr.reshape(\
 #    			loaded_arr.shape[0], loaded_arr.shape[1] // vox_res,vox_res)
	# 	else:
	# 		# for ind,pt in enumerate([self.data[0]]):
	# 		for ind,pt in enumerate(self.data):
	# 			# self.gen_pt_cloud(pt)
	# 			self.point_cloud[ind*self.ppb:(1+ind)*self.ppb] = self.gen_pt_cloud(pt)
	# 			# Add the points, colors and normals as Vectors
	# 		pcd = o3d.geometry.PointCloud()
	# 		pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
	# 		# pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(self.ppb*self.nBalls, 3)))
	# 		visualization = True
	# 		cubic_size = cubic_size
	# 		# voxel_resolution = 50.0
	# 		voxel_resolution = vox_res

	# 		voxel_grid, voxel_carving, voxel_surface = vox_carve(pcd, cubic_size, voxel_resolution)

	# 		# # pcd.points = o3d.utility.Vector3dVector(self.gen_pt_cloud(pt))
	# 		# pcd.estimate_normals()
	# 		# # exit(0)
		
	# 		# print("HERE1")
	# 		# o3d.io.write_voxel_grid(vox_file, voxel_grid)
	# 		# print(voxel_grid.get_geometry_type())
	# 		print(len(voxel_grid.get_voxels()))

	# 		voxel_array = np.zeros((int(voxel_resolution),int(voxel_resolution),int(voxel_resolution)))
	# 		voxels = voxel_grid.get_voxels()

	# 		for vox in voxels:
	# 			vxpt = vox.grid_index
	# 			voxel_array[vxpt[0],vxpt[1],vxpt[2]] += 1

	# 		print(voxel_array)
	# 		print(np.sum(voxel_array))

	# 		# np.savetxt(vox_file,voxel_array.reshape(voxel_array.shape[0],-1))

	# 	# o3d.visualization.draw_geometries([pcd])
	# 	# tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
	# 	# print("HERE2")
	# 	# alpha = 0.03
	# 	# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
	# 	# mesh.compute_vertex_normals()
	# 	# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
	# 	# # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
	# 	# # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha,tetra_mesh,pt_map)
	# 	# print("HERE3")
	# 	# exit(0)
	# 	# mesh.compute_vertex_normals()

		

	# 	# self.point_cloud[ind*self.ppb:(ind+1)*self.ppb] = self.gen_pt_cloud(pt)
	# 	# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.000001)
	# 	# voxel_grid = o3d.geometry.VoxelGrid.create_dense(voxel_grid)
	# 	# print(voxel_grid.get_voxels)
		





	# 	o3d.visualization.draw_geometries([voxel_grid])
	# 	# vis = o3d.visualization.Visualizer()
	# 	# # # Create a window, name it and scale it
	# 	# vis.create_window(window_name='VoxVis', width=800, height=600)
	# 	# # vis.draw_geometries([mesh], mesh_show_back_face=True)
	# 	# o3d.visualization.draw_geometries([voxel_carving])
	# 	# # # # Add the voxel grid to the visualizer
	# 	# vis.add_geometry(voxel_grid)

	# 	# # # # We run the visualizater
	# 	# vis.run()
	# 	# # # # Once the visualizer is closed destroy the window and clean up
	# 	# vis.destroy_window()



	# 	# ax = plt.figure().add_subplot(projection='3d')
	# 	# ax.voxels(voxel_array, edgecolor='k')

	# 	plt.show()
		

	# 	exit(0)

				
		
	