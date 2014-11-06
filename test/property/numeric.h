#ifndef GYDRA_TEST_PROPERTY_NUMERIC_H_
#define GYDRA_TEST_PROPERTY_NUMERIC_H_

#include <thrust/random.h>

#include "property.h"

namespace gydra {

namespace testing {

namespace property {

template<typename integer_type>
class PropertyTestForAllIntegersOfType : public gydra::testing::property::PropertyTest<integer_type> {

 private:
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<integer_type> dist;

 protected:
  integer_type GenerateCase() {
    return dist(rng);
  };

};

}  // namespace property

}  // namespace test

}  // namespace gydra

#endif // GYDRA_TEST_PROPERTY_NUMERIC_H_
