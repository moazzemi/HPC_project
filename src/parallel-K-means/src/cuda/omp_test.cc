#include <iostream>
#include "cluster.hpp"
#include "timer.c"
#include <omp.h>

using namespace Clustering;

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
  omp_set_num_threads(16); 

  
  PointsSpace ps(num_points, num_dimensions, filename);

  #ifdef VERBOSE
  	std::cout << ps;
  	std::cout << "###" << std::endl;
  #endif

  Clusters clusters(num_clusters, ps);

  stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create (); assert (timer);
  stopwatch_start (timer);

  clusters.k_means();
  
  long double t_seq = stopwatch_stop (timer);
  std::cout <<  "time openMP is :"<< t_seq << std::endl;
  
  #ifdef VERBOSE
  	std::cout << "clusters are:"<<std::endl<<clusters;
  #endif
}
