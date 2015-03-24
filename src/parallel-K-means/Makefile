######
## Kmeans clustering algorithm sequntial.
###
##
####inspored by Antonoi Gulli's code at http://codingplayground.blogspot.com/2009/03/k-means-in-c.html
#####
TOP := $(shell pwd)
CC=g++
CUDACC= nvcc
CUDAFLAGS = -I /data/apps/boost/1.57/include/ 
CFLAGS= -Wall   -I /data/apps/boost/1.57/include/ -g
LDFLAGS=-L /usr/local/lib
OMPFLAGS= -fopenmp 
SOURCES=cluster.cpp test.cpp visualize.cpp 
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=k-means
BIN = $(TOP)/bin/


all: cuda

#seq omp cuda
# $(SOURCES) $(EXECUTABLE)
	
#########sequential##########
SEQ_DIR 	= src/sequential/
SEQ_SRC 	= $(SEQ_DIR)cluster.cpp $(SEQ_DIR)test.cpp $(SEQ_DIR)visualize.cpp
SEQ_OBJ		= $(SEQ_SRC:%.c=%.o)

$(SEQ_OBJ): $(H_FILES)

seq: seq_main
seq_main:$(SEQ_OBJ) $(H_FILES)
	$(CC) $(CFLAGS)  $(SEQ_OBJ) $(LDFLAGS) -o bin/seq_main

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
#########OMP#########
OMP_SRC	=	omp_test.cpp \
		omp_cluster.cpp
		
%.o: %.cc
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ -c $<
 
OMP_OBJ = 	$(OMP_SRC:%.cpp=%.o)

omp_kmeans.o:$(OMP_SRC) 
	$(CC) $(LDFLAGS) $(OMPFLAGS) -c $(OMP_SRC)

omp: omp_main
omp_main: $(OMP_OBJ)
	$(CC) $(LDFLAGS) $(OMPFLAGS) -o omp_main $(OMP_OBJ) 
#########CUDA#########
CUDA_DIR 	= src/cuda/
%.o : %.cu
	$(CUDACC) $(CUDAFLAGS) -o $@ -c $<
CUDA_C_SRC = $(CUDA_DIR)/visualize.cu $(CUDA_DIR)/cuda_cluster.cu
CUDA_CU_SRC = $(CUDA_DIR)/cuda_kmeans.cu

CUDA_C_OBJ = $(CUDA_C_SRC:%.cu=%.o)
CUDA_CU_OBJ = $(CUDA_CU_SRC:%.cu=%.o)

cuda: cuda_main
cuda_main: $(CUDA_C_OBJ) $(CUDA_CU_OBJ)
	$(CUDACC) $(LDFLAGS) -o $(BIN)/$@ $(CUDA_C_OBJ) $(CUDA_CU_OBJ)

#########
runseq:
	@$(BIN)seq_main 1000 2 input/samples.txt 2
runomp:
	@./omp_main 1000 2 input/samples.txt 2
runcuda:
	@$(BIN)cuda_main 1000 2 input/samples.txt 2
clean:
	rm -fr *.o *~ $(EXECUTABLE) $(BIN)/* $(SEQ_DIR)/*.o $(CUDA_DIR)/*.o
