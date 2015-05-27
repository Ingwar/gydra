#include <gtest/gtest.h>

#include <src/octree/morton.h>

#include <test/utils/matchers/bitwise.h>
#include <test/utils/operators.h>
#include "property/morton_code.h"


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


PROPERTY_TEST(MortonOrderRevertibilityTest, compute_morton_code_should_be_revertible_for_all_points_with_coordinates_having_less_than_20_significant_bits, point) {
  ASSERT_THAT(point.x, HasNoMoreThanNSignificantBits(20)) << "Precondition for test is not hold";
  ASSERT_THAT(point.y, HasNoMoreThanNSignificantBits(20)) << "Precondition for test is not hold";
  ASSERT_THAT(point.z, HasNoMoreThanNSignificantBits(20)) << "Precondition for test is not hold";

  const MortonCode morton_code = compute_morton_code(point);
  const uint3 reverted_coordinates = get_coordinates_for_code(morton_code);

  ASSERT_EQ(point, reverted_coordinates) << "Calculation of Morton order for the point " << point << " was not revertible";
}


PROPERTY_TEST(
  MortonOrderPartialRevertibilityTest, result_of_applying_get_coordinates_for_code_to_the_compute_morton_code_should_have_first_20_bits_of_each_coordinate_equal_to_the_first_20_bits_of_each_coordinate_of_original_point, point) {
  const MortonCode morton_code = compute_morton_code(point);
  const uint3 reverted_coordinates = get_coordinates_for_code(morton_code);

  ASSERT_THAT(reverted_coordinates.x, HasBitsEqualToTheBitsOf(point.x, 0, 20u)) << "Calculation of Morton order for the point " << point << " was not partially revertible";
  ASSERT_THAT(reverted_coordinates.y, HasBitsEqualToTheBitsOf(point.y, 0, 20u)) << "Calculation of Morton order for the point " << point << " was not partially revertible";
  ASSERT_THAT(reverted_coordinates.z, HasBitsEqualToTheBitsOf(point.z, 0, 20u)) << "Calculation of Morton order for the point " << point << " was not partially revertible";
}


} //  namespace test

} //  namespace morton

} //  namespace octree

} //  namespace gydra
