#include <gtest/gtest.h>

#include <src/octree/morton.h>

#include <test/utils/matchers/bitwise.h>
#include <test/property/numeric.h>
#include "matchers/morton_code.h"


namespace gydra {

namespace octree {

namespace morton {

namespace test {

using gydra::testing::matchers::HasNoMoreThanNSignificantBits;


template<typename T>
class PropertyTestForAllIntegersHavingLessThan20SignificantBits : public gydra::testing::property::PropertyTestForAllIntegersBetween<T> {

 public:
  PropertyTestForAllIntegersHavingLessThan20SignificantBits(): gydra::testing::property::PropertyTestForAllIntegersBetween<T>(1, 1<<20) {
}

};


class MortonCodeReversionTest : public PropertyTestForAllIntegersHavingLessThan20SignificantBits<MortonCode> {
};


PROPERTY_TEST(MortonCodeReversionTest, coordinates_returned_by_get_coordinates_for_code_should_be_correct_reversion_of_morton_code, morton_code) {
  const uint3 coordinates = get_coordinates_for_code(morton_code);

  ASSERT_THAT(coordinates, IsCorrectReversionOfMortonCode(morton_code));
}

PROPERTY_TEST(MortonCodeReversionTest, coordinates_returned_by_get_coordinates_for_code_should_have_no_more_than_20_bits, morton_code) {
  const uint3 coordinates = get_coordinates_for_code(morton_code);

  ASSERT_THAT(coordinates.x, HasNoMoreThanNSignificantBits(20));
  ASSERT_THAT(coordinates.y, HasNoMoreThanNSignificantBits(20));
  ASSERT_THAT(coordinates.z, HasNoMoreThanNSignificantBits(20));
}


} //  namespace test

} //  namespace morton

} //  namespace octree

} //  namespace gydra
