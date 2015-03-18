#include <iostream>
#include "cluster.hpp"
#include "timer.c"
#include "boost/multi_array.hpp"
#include "cuda_utils.h"
#include <cassert>

#define EPSILON 0.0001

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
*finds square of Euclid distance between multi dimensional points. Thanks to Liao
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
    float ans=0.0;

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

   if (objectId < num_pts) {
	int index = 0;
	float dist, current_dist;

	current_dist = euclid_dist_2(dim, num_pts, k, space, dClusters, objectId, 0);

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
		int *mapping,
		float epsilon)	//the limit to determine the convergence
{
 
  int iteration = 0;

//  float **currentClusters;
//  float **previousClusters;
  


  float *dSpace;
  float *dPreviousClusters;
  int *dMapping;
  //int *deviceIntermediates;


  const unsigned int numThreadsPerClusterBlock = 128;
  const unsigned int numClusterBlocks =
        (num_pts + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
   
   const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char);
  
   const unsigned int numReductionThreads =
        nextPowerTwo(numClusterBlocks);
    const unsigned int reductionBlockSharedDataSize =
        numReductionThreads * sizeof(unsigned int);

    float** currentClusters = malloc2Dfloat(k, dim);
    float** previousClusters = malloc2Dfloat(k, dim);

    CUDA_CHECK_ERROR (cudaMalloc(&dSpace, num_pts*dim*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&dPreviousClusters, k*dim*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&dMapping, num_pts*sizeof(int)));
    //CUDA_CHECK_ERROR(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));
    
    CUDA_CHECK_ERROR(cudaMemcpy(dSpace, space[0],
         num_pts*dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(dMapping, mapping,
         num_pts*sizeof(int), cudaMemcpyHostToDevice));
   
    for(int i = 0; i < k; i++){
	for(int j =0; j < dim; j++){
	    previousClusters[i][j] = space[i][j];
	}
	std::cout<<std::endl;
    } 
    do {
        iteration ++;
	CUDA_CHECK_ERROR(cudaMemcpy(dPreviousClusters, previousClusters[0], k * dim *sizeof(float), cudaMemcpyHostToDevice));

	//find best cluster
	find_best_cluster<<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize>>>
	(num_pts, dim, k, dSpace, dPreviousClusters, dMapping);	
	
        cudaDeviceSynchronize();
	
	//compute threshhold
	
	CUDA_CHECK_ERROR(cudaMemcpy(mapping, dMapping,
                  num_pts*sizeof(int), cudaMemcpyDeviceToHost));
	for(int i = 0; i < num_pts; i++){
		std::cout<<mapping[i];
	}
	//make next centroids

	//move next cluster to previous ones.
 	
     } while (iteration < 100);//TODO:add the threshhold
    
    CUDA_CHECK_ERROR(cudaFree(dSpace));
    CUDA_CHECK_ERROR(cudaFree(dPreviousClusters));
    CUDA_CHECK_ERROR(cudaFree(dMapping));
    //CUDA_CHECK_ERROR(cudaFree(deviceIntermediates));
   
    //free();

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

  assert(pSpace != NULL);
  assert(kCentroids != NULL);
  
  for(int i = 0; i < num_pts; i++){
    for(int j = 0; j < dim; j++){
	pSpace[i][j] =  ps.getPoint(i)[j];
    }
	mapping[i] = -1;
  }


  gpuKmeans(num_pts, dim, k, pSpace, mapping, EPSILON);

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


