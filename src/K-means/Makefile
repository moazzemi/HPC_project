######
## Kmeans clustering algorithm sequntial.
###
##
####inspored by Antonoi Gulli's code at http://codingplayground.blogspot.com/2009/03/k-means-in-c.html
#####
CC=g++
CFLAGS=-c -Wall -I /usr/local/include/boost-1_37/ -g
LDFLAGS=-L /usr/local/lib 
SOURCES=cluster.cpp test.cpp visualize.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=k-means

all: $(SOURCES) $(EXECUTABLE)
	

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
run:
	@./k-means
clean:
	rm -fr *.o *~ $(EXECUTABLE)
