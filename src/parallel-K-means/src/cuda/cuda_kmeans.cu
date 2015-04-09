#include <iostream>
#include "cluster.hpp"
#include "timer.c"
#include "boost/multi_array.hpp"
#include "cuda_utils.h"
#include <cassert>
//#include "cuPrintf.cu"

#define EPSILON 0.0001
#define NODEBUG 
using namespace Clustering;

/**
*finds next power of two value. useful for selecting num threads or blocks
**/
static inline int nextPowerTwo(int n){
	n--;

	n = n >> 1 | n;
	n = n >> 2 | n;
	n = n >> 4 | n;
	n = n >> 8 | n;
	n = n >> 16 | n;	
	
	return ++n;
}

/*
* allocates a 2D matrix
*/
float** malloc2Dfloat(int arraySizeX, int arraySizeY) {
float** theArray;
theArray = (float**) malloc(arraySizeX*sizeof(float*));
for (int i = 0; i < arraySizeX; i++) 
   theArray[i] = (float*) malloc(arraySizeY*sizeof(float)); 
   return theArray;  
}

/*
*finds square of Euclid distance between multi dimensional points.
*WARNING: This function is obtained from an open source project. is not completely tested
*/
__host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans = 0.0;

    for (i = 0; i < numCoords; i++) {
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }
    
    return(ans);
}

/**
* find the cluster that current point will map to
**/
__global__ static
void find_best_cluster(	int num_pts,
			int dim,
			int k,
			float *space,
			float *dClusters,
			int *mapping)
{
  int objectId = blockDim.x * blockIdx.x + threadIdx.x;
  float *clusters = dClusters;
   if (objectId < num_pts) {
	int i;
	int index = 0;
	float dist, current_dist;
	dist = 0.0;
	current_dist = 0.0;
	current_dist = euclid_dist_2( dim,num_pts, k, space, clusters, objectId, 0);
	for (i=1; i< k; i++) {
            dist = euclid_dist_2( dim, num_pts, k,
                                 space, clusters, objectId, i);
            /* no need square root */
            if (dist < current_dist) { /* find the min and its array index */
                current_dist = dist;
                index    = i;
            }
        
	}
	mapping[objectId] =  index;
    }

}


/***
*GPU main part
*
*
***/
void
gpuKmeans(	int num_pts,	//number of points in space
		int dim,	//number of dimentions
		int k,		//number of clusters
		float **space,	//data points in space
		int *mapping,   //point to cluster mapping
		int *clusterSize,//number of clusters mapped to each previous cluster
		int *currentClusterSize,
		float epsilon)	//the limit to determine the convergence
{
 //for measuring the memory instantiation and transfer to GPU before actual computation
  stopwatch_init ();
  struct stopwatch_t* timer_data = stopwatch_create (); assert (timer_data);
  stopwatch_start (timer_data);


  int iteration = 0;
  float delta = EPSILON;
  
  //device memory
  float *dSpace;
  float *dPreviousClusters;
  int *dMapping;

  /* calculating dimensions of blocks for GPU*/
  const unsigned int numThreadsPerClusterBlock = 128;
  const unsigned int numClusterBlocks =
        (num_pts + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
   
   const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char);
  

 
    float** transSpace = malloc2Dfloat(num_pts, dim);
    float** transClusters = malloc2Dfloat(dim, k);
    float** currentClusters = malloc2Dfloat( dim, k);
    float** previousClusters = malloc2Dfloat(dim,k);

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < num_pts; j++) {
            transSpace[i][j] = space[j][i];
        }
    }
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < k; j++) {
            transClusters[i][j] = space[i][j];
        }
    }

    /*allocate GPU memory*/
    CUDA_CHECK_ERROR (cudaMalloc(&dSpace, num_pts*dim*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&dPreviousClusters, k*dim*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&dMapping, num_pts*sizeof(int)));
    
    /*move space and clusters to GPU*/
    CUDA_CHECK_ERROR(cudaMemcpy(dSpace, transSpace[0],
         num_pts*dim*sizeof(float), cudaMemcpyHostToDevice));
   
   /*intialization of clusters with first K nodes*/
    for(int i = 0; i < k; i++){
	for(int j =0; j < dim; j++){
	    transClusters[j][i] = space[i][j];
	//std::cout << space[i][j];
	}
    }
   long double t_data = stopwatch_stop (timer_data);
   std::cout <<  "time for moving data is :"<< t_data << std::endl; 
  
   //for measuring the computation part(some memory operations still included)
   stopwatch_init ();
   struct stopwatch_t* timer_compute = stopwatch_create (); assert (timer_compute);
   stopwatch_start (timer_compute);
 
    do {
	iteration ++;
	delta = 0.0;
	/*move the previous clusters to GPU as it got updated at last iteration*/
    	CUDA_CHECK_ERROR(cudaMemcpy(dPreviousClusters, transClusters[0], 	
		k * dim *sizeof(float), cudaMemcpyHostToDevice));
	//find best cluster
	//cudaPrintfInit();
	find_best_cluster<<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize>>>
	(num_pts, dim, k, dSpace, dPreviousClusters, dMapping);	
	//cudaPrintfDisplay(stdout, true);
  	//cudaPrintfEnd();
        cudaDeviceSynchronize();
	
	
	//make next centroids
	CUDA_CHECK_ERROR(cudaMemcpy(mapping, dMapping,
                  num_pts*sizeof(int), cudaMemcpyDeviceToHost));
		
	//TODO:maybe change for optimizatio. have the data structures to be done on GPU	
	for(int i = 0; i < num_pts; i++){
	    int mydex = mapping[i];
	    for(int j = 0; j < dim; j++){	
		currentClusters[j][mydex] +=space[i][j];
	    }
	    currentClusterSize[mydex]++;	
	}
	
	#ifdef _DEBUGKM
	for(int i = 0; i < k; i++){
            for(int j = 0; j < dim; j++){
                std::cout << "current "<<i << "["<<j<<"]"<< currentClusters[j][i]/currentClusterSize[i]<<", ";
            }
            std::cout << std::endl;  
        }
        for(int i = 0; i < k; i++){
            for(int j = 0; j < dim; j++){
                std::cout << "prev "<<i << "["<<j<<"]"<< transClusters[j][i] << std::endl; 
            }
        }
        for(int i = 0; i < k; i++){
	 	std::cout << "size of prev " << clusterSize[i] << " vs " << currentClusterSize[i]<< std::endl; 
	}
	#endif
	
	//compute delta
        for(int i = 0; i < k; i++){
            for(int j = 0; j < dim; j++){
		delta+=(transClusters[j][i] - (currentClusters[j][i]/currentClusterSize[i]))*
			(transClusters[j][i] - (currentClusters[j][i]/currentClusterSize[i]));
	    }
	}
	//move next cluster to previous ones.
	for(int i = 0; i < k; i++){
            for(int j = 0; j < dim; j++){
		transClusters[j][i] = currentClusters[j][i]/currentClusterSize[i];
            	currentClusters[j][i] = 0.0;
	    }
            clusterSize[i] = currentClusterSize[i];
	    currentClusterSize[i] = 0; 
        }
	#ifdef _DEBUGKM
	std::cout << "delta is:"<< delta << std::endl;
 	#endif
     } while ((iteration < 100)&&(delta > EPSILON * EPSILON));//TODO:add the threshhold
    
     long double t_compute = stopwatch_stop (timer_compute);
     std::cout <<  "time for computing data is :"<< t_compute << std::endl; 
    
    CUDA_CHECK_ERROR(cudaFree(dSpace));
    CUDA_CHECK_ERROR(cudaFree(dPreviousClusters));
    CUDA_CHECK_ERROR(cudaFree(dMapping));

}
int main(int argc, char* argv[])
{

  char* filename;
  int num_pts;
  int dim;
  int k;
  //Clock timer;

  if (argc != 5) {
    std::cout << " wrong number of arguments: # points, dim, sample file, #clusters"<< std::endl;
    return -1;
  }

  /** Command line input */
  num_pts = atoi (argv[1]);
  dim = atoi (argv[2]);
  filename = argv[3];
  k = atoi (argv[4]);
  
  ClusterId num_clusters = k;
  PointId num_points = num_pts;
  Dimensions num_dimensions = dim;
  

  
  PointsSpace ps(num_points, num_dimensions, filename);
  #ifdef VERBOSE
  	std::cout << ps;
  	std::cout << "###" << std::endl;
  #endif

  Clusters clusters(num_clusters, ps);
  /** data structures**/
  float** pSpace = malloc2Dfloat(num_pts, dim);
  float** kCentroids = malloc2Dfloat(k, dim);
  int mapping [num_pts];
  int clusterSize [num_pts];
  int nextClusterSize [num_pts];

  assert(pSpace != NULL);
  assert(kCentroids != NULL);
  
  for(int i = 0; i < num_pts; i++){
    for(int j = 0; j < dim; j++){
	pSpace[i][j] =  ps.getPoint(i)[j];
    }
	mapping[i] = -1;
	clusterSize[i] = 0;
	nextClusterSize[i] = 0;
  }
  //for total gpu time
  stopwatch_init ();
  struct stopwatch_t* timer1 = stopwatch_create (); assert (timer1);
  stopwatch_start (timer1);

  gpuKmeans(num_pts, dim, k, pSpace, mapping, clusterSize, nextClusterSize, EPSILON);

  long double t_seq1 = stopwatch_stop (timer1);
  std::cout <<  "total time gpu is :"<< t_seq1 << std::endl;
  
  stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create (); assert (timer);
  stopwatch_start (timer);
  

  clusters.k_means();
  
  long double t_seq = stopwatch_stop (timer);
  std::cout <<  "time sequential is :"<< t_seq << std::endl;
  
  #ifdef VERBOSE
  	std::cout << "clusters are:"<<std::endl<<clusters;
  #endif

  //free stuff up!
  for(int i = 0; i < num_pts; i++){
     free(pSpace[i]);
  }
  free(pSpace);
  for(int i = 0; i < k; i++){
     free(kCentroids[i]);
  }
  free(kCentroids);
}


