Welcome to SpaceLab!

SpaceLab is a discrete element code for simulating cosmic collisions. These collisions can be at the scale of micrometers for simulating dust to the scale of kilometers for simulating asteroid collisions. 

No installation is needed for SpaceLab. All you need is a g++ compiler and you should be good to go.

To get started, all you need to do is run one of the "make_*.py" files and it will compile and run the specified jobs. These "make_*.py" files work by creating an input.json file for the simulation inputs, copying necessary files to the correct directores, and then compiling and running the individual jobs. Depending on your system, you may need to edit the Makefile provided.

To initialize the submodule in the git folder run the following two commands
	git submodule init
	git submodule update

File/Directory scheme:
When you make a new job set with a make_*.py file, there are several ways of organizing the resulting data. The first way to distinguish specific jobs is by setting the "job_set_name" variable in the make_*.py file. This will create a new folder in the form:
	/*SpaceLabDir*/jobs/job_set_name
This is useful both for small tests, where the output can go directly into the job_set_name folder, as well as large sets of jobs that can then have a further folder hierarchy to distinguish between specific jobs. In this case the make_*.py file may need to be edited to suit your job set's specific needs. For example, if I want a job set that has simulations for a set of temperatures for various aggregate final sizes, I might use the folder scheme:
	/*SpaceLabDir*/jobs/job_set_name/N_{}/T_{}/
Currently, if multiple runs of the same conditions (in this example, the conditions of N and T) then multiple job_set_name folders are made, distinguished by an index immediatly following job_set_name. For example:
	/*SpaceLabDir*/jobs/job_set_name0/N_{}/T_{}/
	/*SpaceLabDir*/jobs/job_set_name1/N_{}/T_{}/
There is no reason that this couldn't be changed to: 
	/*SpaceLabDir*/jobs/job_set_name/index/N_{}/T_{}/
Where index is the distinguishing feature.