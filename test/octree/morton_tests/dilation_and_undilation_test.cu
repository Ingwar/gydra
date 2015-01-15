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

class DilationAndUndilationTest : public gydra::testing::property::PropertyTestForAllIntegersBetween<uint64> {
 public:
  DilationAndUndilationTest(): PropertyTestForAllIntegersBetween<uint64>(0ull, 1ull << 20) {
  }
};

PROPERTY_TEST(DilationAndUndilationTest, dilation_and_undilation_should_be_revertible_for_integers_with_no_more_than_20_significant_bits, number) {
  ASSERT_THAT(number, HasNoMoreThanNSignificantBits(2 * DILATION_SIZE)) << "Precondition for test is not hold";

  const uint64 dilated_number = dilate(number);
  const uint64 undilated_number = undilate(dilated_number);

  ASSERT_EQ(number, undilated_number) << "Result of undilation of dilated number is not equal to the original number";
}

PROPERTY_TEST(DilationAndUndilationTest, undilation_of_dilated_number_should_have_first_twenty_bits_equal_to_the_first_twenty_bits_of_original_number, number) {
  const uint64 dilated_number = dilate(number);
  const uint64 undilated_number = undilate(dilated_number);

  ASSERT_THAT(undilated_number, HasBitsEqualToTheBitsOf(number, 0, 2 * DILATION_SIZE));
}

} //  namespace test

} //  namespace helpers

} //  namespace morton

} //  namespace octree

} //  namespace gydra
