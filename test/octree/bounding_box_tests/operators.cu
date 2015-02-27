#include <boost/format.hpp>

#include <gydra.h>

#include "operators.h"


std::ostream& operator<<(std::ostream& os, const gydra::real3& point) {
  boost::format representation("{x = %1%, y = %2%, z = %3%}");
  representation % point.x % point.y % point.z;
  os << representation;
  return os;
}
