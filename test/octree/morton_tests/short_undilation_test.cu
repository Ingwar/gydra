#include <gtest/gtest.h>

#include <src/octree/morton.h>

#include <test/utils/matchers/bitwise.h>

#include "matchers/dilation.h"
#include "property/dilation.h"

namespace gydra {

namespace octree {

namespace morton {

namespace helpers {

namespace test {

using gydra::testing::matchers::HasNoMoreThanNSignificantBits;

class ShortUndilationTest : public property::DilatedIntegerPropertyTest<unsigned int> {

 public:
  ShortUndilationTest(): DilatedIntegerPropertyTest<unsigned int>(3 * DILATION_SIZE) {
  }

};

PROPERTY_TEST(ShortUndilationTest, undilated_number_should_be_no_longer_than_10_bits, number) {
  const unsigned int undilated_number = undilate_short(number);

  ASSERT_THAT(undilated_number, HasNoMoreThanNSignificantBits(10));
}

PROPERTY_TEST(ShortUndilationTest, short_dilation_should_correctly_dilate_first_10_bits_of_number, number) {
  const unsigned int undilated_number = undilate_short(number);

  ASSERT_THAT(undilated_number, IsShortUndilationOf(number));
}

} //  namespace test

} //  namespace helpers

} //  namespace morton

} //  namespace octree

} //  namespace gydra
