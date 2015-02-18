#ifndef GYDRA_TEST_OCTREE_MORTON_PROPERTY_H_
#define GYDRA_TEST_OCTREE_MORTON_PROPERTY_H_

#include <cstdlib>
#include <ctime>
#include <bitset>

#include <test/property/property.h>

namespace gydra {

namespace octree {

namespace morton {

namespace helpers {

namespace test {

namespace property {

using gydra::testing::matchers::SizeInfo;

template<typename T>
class DilatedIntegerPropertyTest : public gydra::testing::property::PropertyTest<T> {

 private:
  const unsigned int dilation_rank;
  const size_t dilated_integer_size;

 protected:
  T GenerateCase() {
    std::bitset< SizeInfo<T>::size_in_bits > bits;

    //initialize random number generator
    srand (time(NULL));

    for (size_t i = 0; i < dilated_integer_size; i += 3) {
      const bool current_bit = rand() % 2 == 1;
      bits.set(i, current_bit);
    }

    return bits.to_ulong();
  }

 public:
  DilatedIntegerPropertyTest(const size_t dilation_size): dilated_integer_size(dilation_size), dilation_rank(3) {
  }

};

} //  namespace property

} //  namespace test

} //  namespace helpers

} //  namespace morton

} //  namespace octree

} //  namespace gydra

#endif //  GYDRA_TEST_OCTREE_MORTON_PROPERTY_H_

