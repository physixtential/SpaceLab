# CC=g++-10
# std=c++2a
# CFLAGS=-I.
# CPP=CC
CPP=g++
# CPPflags=-O3 -std=c++2a -fopenmp
# CPPflags=-O3 -std=c++2a 
CPPflags=-std=c++2a -O2

ColliderSingleCore.x: ColliderSingleCore.cpp ../dust_const.hpp ../ball_group.hpp ../timing/timing.hpp
	${CPP} ${CPPflags} -o ColliderSingleCore.x ColliderSingleCore.cpp

ColliderSingleCore_COPY.x: ColliderSingleCore_COPY.cpp ../dust_const.hpp ../ball_group_COPY.hpp ../timing/timing.hpp
	${CPP} ${CPPflags} -o ColliderSingleCore_COPY.x ColliderSingleCore_COPY.cpp

clean:
	rm -f *.o
	rm -f *.x
