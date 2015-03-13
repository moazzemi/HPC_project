#include <iostream>
#include "cluster.hpp"
#include "timer.c"

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
  std::cout <<  "time sequential is :"<< t_seq << std::endl;
  
  #ifdef VERBOSE
  	std::cout << "clusters are:"<<endl<<clusters;
  	std::cout << "point cluster mapping is :" <<endl<<clusters.pcMap();
  	std::cout << clusters.points_to_clusters__;
  #endif
}
