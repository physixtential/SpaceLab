CFLAGS = -std=c++2a -O2 -I/usr/include/hdf5/serial/
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5 -lhdf5_cpp

# Define sources and objects
SOURCES = testh5cpp.cpp unitTest.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: h5cpp unitTest

h5cpp: testh5cpp.o
	g++ -o $@ $^ $(LDFLAGS)

unitTest: unitTest.o
	g++ -o $@ $^ $(LDFLAGS)

%.o: %.cpp h5CPP.hpp
	g++ $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f *.hdf5
	rm -f *.h5
	rm -f h5cpp unitTest
