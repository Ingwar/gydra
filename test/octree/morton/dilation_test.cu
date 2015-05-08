#include <gtest/gtest.h>

#include <src/octree/morton.h>

#include <test/property/numeric.h>
#include <test/utils/matchers/bitwise.h>

#include "matchers/dilation.h"

namespace gydra {

namespace octree {

namespace morton {

namespace helpers {

namespace test {

using gydra::testing::matchers::HasNoMoreThanNSignificantBits;

class DilationTest : public gydra::testing::property::PropertyTestForAllIntegersOfType<unsigned int> {
};

PROPERTY_TEST(DilationTest, dilated_number_should_be_no_longer_than_60_bits, number) {
  const uint64 dilated_number = dilate(number);

  ASSERT_THAT(dilated_number, HasNoMoreThanNSignificantBits(60)) << " for number " << number;
}

PROPERTY_TEST(DilationTest, dilation_should_correctly_dilate_first_20_bits_of_number, number) {
  const uint64 dilated_number = dilate(number);

  ASSERT_THAT(dilated_number, IsDilationOf(static_cast<uint64>(number)));
}

} //  namespace test

} //  namespace helpers

} //  namespace morton

} //  namespace octree

} //  namespace gydra
