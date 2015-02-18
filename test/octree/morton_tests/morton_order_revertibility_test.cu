#include <gtest/gtest.h>

#include <src/octree/morton.h>

#include <test/utils/matchers/bitwise.h>
#include "property/morton_order.h"
#include "utils.h"

namespace gydra {

namespace octree {

namespace morton {

namespace test {

using gydra::testing::matchers::HasNoMoreThanNSignificantBits;
using gydra::testing::matchers::HasBitsEqualToTheBitsOf;


class MortonOrderRevertibilityTest : public PropertyTestForAllPointsWithCoordinatesHavingLessThan20Bits {
};


class MortonOrderPartialRevertibilityTest : public PropertyTestForAllPoints {
};


PROPERTY_TEST(MortonOrderRevertibilityTest, get_morton_key_should_be_revertible_for_all_points_with_coordinates_having_less_than_20_significant_bits, point) {
  ASSERT_THAT(point.x, HasNoMoreThanNSignificantBits(20)) << "Precondition for test is not hold";
  ASSERT_THAT(point.y, HasNoMoreThanNSignificantBits(20)) << "Precondition for test is not hold";
  ASSERT_THAT(point.z, HasNoMoreThanNSignificantBits(20)) << "Precondition for test is not hold";

  const MortonKey morton_order = get_morton_key(point);
  const uint3 reverted_coordinates = get_coordinates_for_key(morton_order);

  ASSERT_EQ(point, reverted_coordinates) << "Calculation of Morton order for the point " << point << " was not revertible";
}


PROPERTY_TEST(
  MortonOrderPartialRevertibilityTest, result_of_applying_get_coordinates_for_key_to_the_get_morton_key_should_have_first_20_bits_of_each_coordinate_equal_to_the_first_20_bits_of_each_coordinate_of_original_point, point) {
  const MortonKey morton_order = get_morton_key(point);
  const uint3 reverted_coordinates = get_coordinates_for_key(morton_order);

  ASSERT_THAT(reverted_coordinates.x, HasBitsEqualToTheBitsOf(point.x, 0, 20u)) << "Calculation of Morton order for the point " << point << " was not partially revertible";
  ASSERT_THAT(reverted_coordinates.y, HasBitsEqualToTheBitsOf(point.y, 0, 20u)) << "Calculation of Morton order for the point " << point << " was not partially revertible";
  ASSERT_THAT(reverted_coordinates.z, HasBitsEqualToTheBitsOf(point.z, 0, 20u)) << "Calculation of Morton order for the point " << point << " was not partially revertible";
}


} //  namespace test

} //  namespace morton

} //  namespace octree

} //  namespace gydra
