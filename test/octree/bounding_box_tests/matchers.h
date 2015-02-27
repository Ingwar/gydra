#ifndef GYDRA_TEST_OCTREE_BOUNDING_BOX_MATCHERS_H_
#define GYDRA_TEST_OCTREE_BOUNDING_BOX_MATCHERS_H_

#include <gmock/gmock.h>

#include <gydra.h>


namespace gydra {

namespace octree {

namespace bounding_box {

namespace test {


MATCHER_P(IsEqualToThePointWithFloatCoordinates, point, "") {
  const ::testing::internal::FloatingPoint<real> arg_x(arg.x);
  const ::testing::internal::FloatingPoint<real> arg_y(arg.y);
  const ::testing::internal::FloatingPoint<real> arg_z(arg.z);

  const ::testing::internal::FloatingPoint<real> point_x(point.x);
  const ::testing::internal::FloatingPoint<real> point_y(point.y);
  const ::testing::internal::FloatingPoint<real> point_z(point.z);

  if (!arg_x.AlmostEquals(point_x)) {
    return false;
  }

  if (!arg_y.AlmostEquals(point_y)) {
    return false;
  }

  if (!arg_z.AlmostEquals(point_z)) {
    return false;
  }

  return true;
}


}  //namespace test

}  //namespace bounding_box

}  // namespace octree

}  // namespace gydra

#endif //  GYDRA_TEST_OCTREE_BOUNDING_BOX_MATCHERS_H_
