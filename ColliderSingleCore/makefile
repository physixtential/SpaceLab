# CC=g++-10
# std=c++2a
# CFLAGS=-I.
CPP=CC
CPPflags=-O3 -std=c++2a -fopenmp

ColliderSingleCore.o: ColliderSingleCore.cpp ../dust_const.hpp ../ball_group.hpp ../timing/timing.hpp
	${CPP} ${CPPflags} -o ColliderSingleCore.x ColliderSingleCore.cpp

test: ColliderSingleCore.cpp 
	${CPP} ${CPPflags} -o testSC.x ColliderSingleCore.cpp

clean:
	rm -f *.o
