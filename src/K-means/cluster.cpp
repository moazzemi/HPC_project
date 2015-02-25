#include "cluster.hpp"
//#define VERBOSE

namespace Clustering{

  //
  // distance between two points
  //
  Distance distance(const Point& x, const Point& y)
  {
    Distance total = 0.0;
    Distance diff;
    
    Point::const_iterator cpx=x.begin(); 
    Point::const_iterator cpy=y.begin();
    Point::const_iterator cpx_end=x.end();
    for(;cpx!=cpx_end;++cpx,++cpy){
      diff = *cpx - *cpy;
      total += (diff * diff); 
    }
    return total;  // no need to take sqrt, which is monotonic
  }

  //
  // Init collection of points
  //  
  void PointsSpace::init_points()
  {     
    for (PointId i=0; i < num_points__; i++)
    {
      Point p;
      for (Dimensions d=0 ; d < num_dimensions__; d++)
    	{ 
        p.push_back( rand() % 100 ); 
      }     
      points__.push_back(p);
     // std::cout << "my_pid[" << i << "]= (" << p << ")" <<std::endl;; 
    }
  }  

  //
  // Zero centroids
  //
  void Clusters::zero_centroids()
  {
    BOOST_FOREACH(Centroids::value_type& centroid, centroids__)
    {
      BOOST_FOREACH(Point::value_type& d, centroid)
      {
	      d = 0.0;
      }
    }
  }

  //
  // Compute Centroids
  //
  void Clusters::compute_centroids() {
    
    Dimensions i;
    ClusterId cid = 0;
    PointId num_points_in_cluster;
    // For each centroid
    BOOST_FOREACH(Centroids::value_type& centroid, centroids__){
       num_points_in_cluster = 0;
      // For earch PointId in this set
      BOOST_FOREACH(SetPoints::value_type pid, clusters_to_points__[cid])
      {
      	Point p = ps__.getPoint(pid);
      	//std::cout << "(" << p << ")";
      	for (i=0; i<num_dimensions__; i++)
	         centroid[i] += p[i];	
      	num_points_in_cluster++;
      }
      //
      // if no point in the clusters, this goes to inf (correct!)
      //
      for (i=0; i<num_dimensions__; i++)
      	centroid[i] /= num_points_in_cluster;	
      cid++;
    }
  }

  //
  // Initial partition points among available clusters
  //
  void Clusters::initial_partition_points(){
    
    ClusterId cid;
    
    for (PointId pid = 0; pid < ps__.getNumPoints(); pid++){
      
      cid = pid % num_clusters__;

      points_to_clusters__[pid] = cid;
      clusters_to_points__[cid].insert(pid);
    }    
   // uncomment if you want to see the initial mapping of Clustors to points and vice versa 
   // std::cout << "Points_to_clusters " << std::endl;
   // std::cout << points_to_clusters__;
   // std::cout << "Clusters_to_points " << std::endl;
   // std::cout << clusters_to_points__;  
  };

  //
  // k-means
  //
  void Clusters::k_means(void){
    
    bool move;
    bool some_point_is_moving = true;
    unsigned int num_iterations = 0;
    PointId pid;
    ClusterId cid, to_cluster;
    Distance d, min;
    

    //
    // Initial partition of points
    //
    initial_partition_points();

    //
    // Until not converge
    //
    while (some_point_is_moving){
    #ifdef VERBOSE
      std::cout << std::endl << "*** Num Iterations " 
		<< num_iterations  << std::endl << std::endl ;;
    #endif
      some_point_is_moving = false;

      compute_centroids();
      // shows the current centroids     
      // std::cout << "Centroids" << std::endl << centroids__;      

      //
      // for each point
      //
      for (pid=0; pid<num_points__; pid++)
      {
	    // distance from current cluster
	      min = Clustering::distance(centroids__[points_to_clusters__[pid]], ps__.getPoint(pid));
     #ifdef VERBOSE
     std::cout << "pid[" << pid << "] in cluster=" << points_to_clusters__[pid] 
     << " with distance=" << min << std::endl;
     #endif
	//
	// foreach centroid
	//
	cid = 0; 
	move = false;
	BOOST_FOREACH(Centroids::value_type c, centroids__){
	  

	  d = Clustering::distance(c, ps__.getPoint(pid));
	  if (d < min){
	    min = d;
	    move = true;
	    to_cluster = cid;

	    // remove from current cluster
	    clusters_to_points__[points_to_clusters__[pid]].erase(pid);

	    some_point_is_moving = true;
	    #ifdef VERBOSE
       std::cout << "\tcluster=" << cid 
		      << " closer, dist=" << d << std::endl;	    
      #endif
    }
	  cid++;
	}
	
	//
	// move towards a closer centroid 
	//
	if (move){
	  
	  // insert
	  points_to_clusters__[pid] = to_cluster;
	  clusters_to_points__[to_cluster].insert(pid);
	  #ifdef VERBOSE
    std::cout << "\t\tmove to cluster=" << to_cluster << std::endl;
    #endif
  }
      }      

      num_iterations++;
    } // end while (some_point_is_moving)
//    #ifdef VERBOSE
    //shows mapping of each point to corresponding to cluster 
    std::cout << points_to_clusters__;
//    #endif
//    for (PointId i=0; i < num_points__; i++)
//    {
//      std::cout << "my_pid[" << i << "]= (" << p << ")" <<std::endl; 
//    }
//    std::cout << points__;
    //std::cout << std::endl << "Final clusters" << std::endl;
    #ifdef VERBOSE
    //comment this if dont want Clusters to Points
    // std::cout << clusters_to_points__;
    #endif
}
};
