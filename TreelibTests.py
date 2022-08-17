# from treelib import Node, Tree
# import uuid


# tree = Tree()

stri = '001'
strj = '101'

# tree.create_node(uuid.uuid1(),data=stri)


class node(object):
	"""docstring for node"""
	def __init__(self,parent,id,data):
		super(node, self).__init__()
		self.parent = parent
		self.data = data
		self.id = uuid.uuid1()
		self.kids = []


class tree(object):
	"""docstring for tree"""
	def __init__(self, root):
		super(tree, self).__init__()
		self.root = root
		self.nnodes = 0 #root is node 0
	
	# def add_node(parent,id,data):

		

result = 0

if result == 1:
	print('bad')
elif result == 0:
	print("bad1")
else:
	print('good')