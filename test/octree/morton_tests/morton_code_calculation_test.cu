#include <gtest/gtest.h>

#include <src/octree/morton.h>

#include <test/utils/matchers/bitwise.h>
#include "matchers/morton_code.h"
#include "property/morton_code.h"
#include "utils.h"

namespace gydra {

namespace octree {

namespace morton {

namespace test {

using gydra::testing::matchers::HasNoMoreThanNSignificantBits;


class MortonCodeTest : public PropertyTestForAllPointsWithCoordinatesHavingLessThan20Bits {
};


class MortonCodeLengthTest: public PropertyTestForAllPoints {
};


PROPERTY_TEST(MortonCodeTest, compute_morton_code_should_correctly_calculate_morton_code, point) {
  const MortonCode result = compute_morton_code(point);
  ASSERT_THAT(result, IsCorrectMortonCodeForPoint(point));
}

PROPERTY_TEST(MortonCodeTest, compute_morton_code_should_return_same_result_for_points_which_have_same_first_20_bits_of_coordidates, point) {

  const unsigned int shift = 1 << 22;

  const size_t number_of_cases = 7;

  const MortonCode expected_morton_code = compute_morton_code(point);

  const uint3 shifted_point_array[number_of_cases] = {
    make_uint3(point.x + shift, point.y, point.z),
    make_uint3(point.x, point.y + shift, point.z),
    make_uint3(point.x, point.y, point.z + shift),
    make_uint3(point.x + shift, point.y + shift, point.z),
    make_uint3(point.x + shift, point.y, point.z + shift),
    make_uint3(point.x, point.y + shift, point.z + shift),
    make_uint3(point.x + shift, point.y + shift, point.z + shift)
  };

  for (size_t i = 0; i < number_of_cases; i++) {
    const uint3 shifted_point = shifted_point_array[i];
    const MortonCode morton_code_for_shifted_point = compute_morton_code(shifted_point);
    ASSERT_EQ(expected_morton_code, morton_code_for_shifted_point) << "Morton code differs for points " << point << " and " << shifted_point;
  }

}


PROPERTY_TEST(MortonCodeLengthTest, morton_code_should_be_no_longer_than_60_bits, point) {
  const MortonCode result = compute_morton_code(point);
  ASSERT_THAT(result, HasNoMoreThanNSignificantBits(60));
}

} //  namespace test

} //  namespace morton

} //  namespace octree

} //  namespace gydra
