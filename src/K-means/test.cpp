#include <iostream>
#include "cluster.hpp"

using namespace Clustering;

int main(int argc, char* argv[])
{
  ClusterId num_clusters = 20;
  PointId num_points = 2000;
  Dimensions num_dimensions = 2;

  PointsSpace ps(num_points, num_dimensions);
  std::cout << ps;
  std::cout << "###" << std::endl;
  Clusters clusters(num_clusters, ps);

  clusters.k_means();
//  std::cout << clusters;
//  std::cout << clusters.pcMap();
//    std::cout <<clusters.points_to_clusters__;
}
