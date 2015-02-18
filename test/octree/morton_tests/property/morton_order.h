#ifndef GYDRA_TEST_OCTREE_MORTON_ORDER_PROPERTY_H_
#define GYDRA_TEST_OCTREE_MORTON_ORDER_PROPERTY_H_

#include <thrust/random.h>

#include <test/property/property.h>

namespace gydra {

namespace octree {

namespace morton {

namespace test {


class PropertyTestForAllPointsInside : public gydra::testing::property::PropertyTest<uint3> {

 public:
  PropertyTestForAllPointsInside(const uint3 left_bottom_rear_corner, const uint3 rigth_top_front_corner);

 protected:
  uint3 GenerateCase();

 private:
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution< unsigned int > x_dist;
  thrust::uniform_int_distribution< unsigned int > y_dist;
  thrust::uniform_int_distribution< unsigned int > z_dist;

};


class PropertyTestForAllPointsWithCoordinatesHavingLessThan20Bits : public PropertyTestForAllPointsInside {

 public:
  PropertyTestForAllPointsWithCoordinatesHavingLessThan20Bits();

};


class PropertyTestForAllPoints : public PropertyTestForAllPointsInside {

 public:
  PropertyTestForAllPoints();

};


} //  namespace test

} //  namespace morton

} //  namespace octree

} //  namespace gydra

#endif //  GYDRA_TEST_OCTREE_MORTON_ORDER_PROPERTY_H_
