#ifndef GYDRA_TEST_NUMERIC_PROPERTY_IMPL_H_
#define GYDRA_TEST_NUMERIC_PROPERTY_IMPL_H_

#include<limits>
#include "../property.h"

namespace gydra {

namespace testing {

namespace property {

template<typename integer_type>
PropertyTestForAllIntegersBetween<integer_type>::PropertyTestForAllIntegersBetween(const integer_type a, const integer_type b): dist(a, b) {
  assert(a <= b);
}

template<typename integer_type>
integer_type PropertyTestForAllIntegersBetween<integer_type>::GenerateCase() {
  return dist(rng);
}

template<typename integer_type>
PropertyTestForAllIntegersOfType<integer_type>::PropertyTestForAllIntegersOfType():
  PropertyTestForAllIntegersBetween<integer_type>(std::numeric_limits<integer_type>::min(), std::numeric_limits<integer_type>::max()) {
}

template<typename integer_type>
PropertyTestForAbsolutlyAllIntegersBeetwen<integer_type>::PropertyTestForAbsolutlyAllIntegersBeetwen(const integer_type a, const integer_type b): 
    minimum(a), maximum(b), current_number(a), PropertyTest<integer_type>(b - a + 1) {

  assert(a <= b);

}

template<typename integer_type>
integer_type PropertyTestForAbsolutlyAllIntegersBeetwen<integer_type>::GenerateCase() {
  const integer_type result = current_number;
  assert(minimum <= result);
  assert(result <= maximum);
  current_number++;
  return result;
}

}  // namespace property

}  // namespace test

}  // namespace gydra

#endif // GYDRA_TEST_NUMERIC_PROPERTY_IMPL_H_
