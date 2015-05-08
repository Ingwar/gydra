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
using gydra::testing::matchers::HasNBitsEqualToTheBitsOf;

class BitsGettersTest : public gydra::testing::property::PropertyTestForAllIntegersOfType<unsigned int> {
};

class BitsGettersForDilatedNumberTest : public gydra::testing::property::PropertyTestForAllIntegersOfType<uint64> {
};

TEST(DilationSizeTest, DILATION_SIZE_should_be_equal_to_10) {
  ASSERT_EQ(10u, DILATION_SIZE);
}

PROPERTY_TEST(BitsGettersTest, result_of_get_first_10_bits_of_number_should_be_no_longer_than_10_bits, number) {
  const unsigned int first_10_bits = get_first_10_bits_of_number(number);

  ASSERT_THAT(first_10_bits, HasBitsEqualToTheBitsOf(number, 0, DILATION_SIZE));
}

PROPERTY_TEST(BitsGettersTest, result_of_get_first_10_bits_of_number_should_be_equal_to_the_first_10_bits_of_original_number, number) {
  const unsigned int first_10_bits = get_first_10_bits_of_number(number);

  ASSERT_THAT(first_10_bits, HasNoMoreThanNSignificantBits(DILATION_SIZE));
}

PROPERTY_TEST(BitsGettersTest, result_of_get_second_10_bits_of_number_should_be_no_longer_than_10_bits, number) {
  const unsigned int second_10_bits = get_second_10_bits_of_number(number);

  ASSERT_THAT(second_10_bits, HasNoMoreThanNSignificantBits(DILATION_SIZE));
}

PROPERTY_TEST(BitsGettersTest, result_of_get_second_10_bits_of_number_should_be_equal_to_the_second_10_bits_of_original_number, number) {
  const unsigned int second_10_bits = get_second_10_bits_of_number(number);

  ASSERT_THAT(second_10_bits, HasNBitsEqualToTheBitsOf(DILATION_SIZE, number, DILATION_SIZE));
}

PROPERTY_TEST(BitsGettersForDilatedNumberTest, result_of_get_first_bits_of_dilated_number_should_be_no_longer_than_30_bits, number) {
  const unsigned int first_bits = get_first_bits_of_dilated_number(number);

  ASSERT_THAT(first_bits, HasBitsEqualToTheBitsOf(number, 0, 3 * DILATION_SIZE));
}

PROPERTY_TEST(BitsGettersForDilatedNumberTest, result_of_get_first_bits_of_dilated_number_should_be_equal_to_the_first_30_bits_of_original_number, number) {
  const unsigned int first_10_bits = get_first_bits_of_dilated_number(number);

  ASSERT_THAT(first_10_bits, HasNoMoreThanNSignificantBits(3 * DILATION_SIZE));
}

PROPERTY_TEST(BitsGettersForDilatedNumberTest, result_of_get_second_bits_of_dilated_number_should_be_no_longer_than_30_bits, number) {
  const unsigned int second_bits = get_second_bits_of_dilated_number(number);

  ASSERT_THAT(second_bits, HasNoMoreThanNSignificantBits(3 * DILATION_SIZE));
}

PROPERTY_TEST(BitsGettersForDilatedNumberTest, result_of_get_second_bits_of_dilated_number_should_be_equal_to_the_second_30_bits_of_original_number, number) {
  const uint64 second_bits = get_second_bits_of_dilated_number(number);

  ASSERT_THAT(second_bits, HasNBitsEqualToTheBitsOf(3 * DILATION_SIZE, number, 3 * DILATION_SIZE));
}

} //  namespace test

} //  namespace helpers

} //  namespace morton

} //  namespace octree

} //  namespace gydra
