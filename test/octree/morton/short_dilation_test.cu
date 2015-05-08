#include <gtest/gtest.h>

#include <src/octree/morton.h>

#include <test/property/property.h>
#include <test/property/numeric.h>
#include <test/utils/matchers/bitwise.h>

#include "matchers/dilation.h"

namespace gydra {

namespace octree {

namespace morton {

namespace helpers {

namespace test {

using gydra::testing::matchers::HasNoMoreThanNSignificantBits;

class ShortDilationTest : public gydra::testing::property::PropertyTestForAbsolutelyAllTenBitsUnsignedIntegers {
};

PROPERTY_TEST(ShortDilationTest, dilated_number_should_be_no_longer_than_30_bits, number) {
  const unsigned int dilated_number = dilate_short(number);

  ASSERT_THAT(dilated_number, HasNoMoreThanNSignificantBits(3 * DILATION_SIZE));
}

PROPERTY_TEST(ShortDilationTest, short_dilation_should_correctly_dilate_first_10_bits_of_number, number) {
  const unsigned int dilated_number = dilate_short(number);

  ASSERT_THAT(dilated_number, IsShortDilationOf(number));
}

} //  namespace test

} //  namespace helpers

} //  namespace morton

} //  namespace octree

} //  namespace gydra
