#include <gtest/gtest.h>

#include <src/octree/morton.h>

#include <test/property/property.h>
#include <test/property/numeric.h>
#include <test/utils/matchers/bitwise.h>

namespace gydra {

namespace octree {

namespace morton {

namespace helpers {

namespace test {

using gydra::testing::matchers::HasNoMoreThanNSignificantBits;
using gydra::testing::matchers::HasBitsEqualToTheBitsOf;

class ShortDilationAndUndilationTest : public gydra::testing::property::PropertyTestForAbsolutelyAllTenBitsUnsignedIntegers {
};

PROPERTY_TEST(ShortDilationAndUndilationTest, short_dilation_and_undilation_should_be_revertible_for_integers_with_no_more_than_10_significant_bits, number) {
  ASSERT_THAT(number, HasNoMoreThanNSignificantBits(DILATION_SIZE)) << "Precondition for test is not hold"; 

  const unsigned int dilated_number = dilate_short(number);
  const unsigned int undilated_number = undilate_short(dilated_number);

  ASSERT_EQ(number, undilated_number) << "Result of undilation of dilated number is not equal to the original number";
}

} //  namespace test

} //  namespace helpers

} //  namespace morton

} //  namespace octree

} //  namespace gydra
