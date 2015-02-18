#include <ctime>
#include <cassert>
#include <limits>

#include "morton_order.h"

namespace gydra {

namespace octree {

namespace morton {

namespace test {

namespace {

const unsigned int max_20_bits_unsigned_int = 1 << 20;
const unsigned int max_unsigned_int = std::numeric_limits<unsigned int>::max();

} //  namespace


PropertyTestForAllPointsInside::PropertyTestForAllPointsInside(const uint3 left_bottom_rear_corner, const uint3 rigth_top_front_corner) {
  assert(left_bottom_rear_corner.x <= rigth_top_front_corner.x);
  assert(left_bottom_rear_corner.y <= rigth_top_front_corner.y);
  assert(left_bottom_rear_corner.z <= rigth_top_front_corner.z);

  rng = thrust::default_random_engine(std::time(NULL));

  x_dist = thrust::uniform_int_distribution< unsigned int >(left_bottom_rear_corner.x, rigth_top_front_corner.x);
  y_dist = thrust::uniform_int_distribution< unsigned int >(left_bottom_rear_corner.y, rigth_top_front_corner.y);
  z_dist = thrust::uniform_int_distribution< unsigned int >(left_bottom_rear_corner.z, rigth_top_front_corner.z);
}

uint3 PropertyTestForAllPointsInside::GenerateCase() {
  const unsigned int x = x_dist(rng);
  const unsigned int y = y_dist(rng);
  const unsigned int z = z_dist(rng);
  return make_uint3(x, y, z);
}


PropertyTestForAllPointsWithCoordinatesHavingLessThan20Bits::PropertyTestForAllPointsWithCoordinatesHavingLessThan20Bits():
  PropertyTestForAllPointsInside(make_uint3(0, 0, 0), make_uint3(max_20_bits_unsigned_int, max_20_bits_unsigned_int, max_20_bits_unsigned_int)) {
}


PropertyTestForAllPoints::PropertyTestForAllPoints():
 PropertyTestForAllPointsInside(make_uint3(0, 0, 0), make_uint3(max_unsigned_int, max_unsigned_int, max_unsigned_int)) {
}


} //  namespace test

} //  namespace morton

} //  namespace octree

} //  namespace gydra
