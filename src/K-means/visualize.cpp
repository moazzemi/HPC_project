#include "cluster.hpp"
namespace Clustering{


  //dumps a point
  std::ostream& operator << (std::ostream& os, Point& p)
  {
   BOOST_FOREACH(Point::value_type d, p)
     { os << d << ",, "; } 
   return os;
  }

  //dump a collection of points
  std::ostream& operator << (std::ostream& os, Points& cps)
  {
    BOOST_FOREACH(Points::value_type p, cps)
    {
      os<< p << std::endl;
    }
    return os;
  }

  //  dumps a set of points
  std::ostream& operator << (std::ostream& os, SetPoints & sp)
  {
    BOOST_FOREACH(SetPoints::value_type pid, sp)
    {
      os << "MY_pid=" << pid << " ";
    }
    return os;
  }

  //dump ClustersTOPoints
  std::ostream& operator << (std::ostream& os, ClustersToPoints & cp)
  {
    ClusterId cid = 0;
    BOOST_FOREACH( ClustersToPoints::value_type set, cp)
    {
      os << "MY_clusterid[" << cid << "]" << "=(" << set << ")" << std::endl;
      cid++;
    }
    return os;
  }

  // dump PointsToClusters
  std::ostream& operator << (std::ostream& os, PointsToClusters & pc)
  {
    PointId pid = 0;
    BOOST_FOREACH( PointsToClusters::value_type cid, pc)
    {
      std::cout << "pid[" << pid << "]==" << cid << std::endl;
      pid ++;
    }
    return os;
  }
}
//#eof
