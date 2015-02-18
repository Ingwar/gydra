#include <boost/format.hpp>
#include <cuda_runtime.h>

#include "utils.h"


std::ostream& operator<<(std::ostream& os, const uint3& point) {
  boost::format representation("{x = %1%, y = %2%, z = %3%}");
  representation % point.x % point.y % point.z;
  os << representation;
  return os;
}


bool operator==(const uint3& a_point, const uint3& another_point) {
  return (a_point.x == another_point.x) && (a_point.y == another_point.y) && (a_point.z == another_point.z);
}
