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

class UndilationTest : public property::DilatedIntegerPropertyTest<uint64> {

 public:
  UndilationTest(): DilatedIntegerPropertyTest<uint64>(6 * DILATION_SIZE) {
  }

};

PROPERTY_TEST(UndilationTest, undilated_number_should_be_no_longer_than_20_bits, number) {
  const unsigned int undilated_number = undilate(number);

  ASSERT_THAT(undilated_number, HasNoMoreThanNSignificantBits(20));
}

PROPERTY_TEST(UndilationTest, short_dilation_should_correctly_dilate_first_10_bits_of_number, number) {
  const uint64 undilated_number = undilate(number);

  ASSERT_THAT(undilated_number, IsUndilationOf(number));
}

} //  namespace test

} //  namespace helpers

} //  namespace morton

} //  namespace octree

} //  namespace gydra
