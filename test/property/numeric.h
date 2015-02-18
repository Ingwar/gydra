#ifndef GYDRA_TEST_PROPERTY_NUMERIC_H_
#define GYDRA_TEST_PROPERTY_NUMERIC_H_

#include <thrust/random.h>

#include "property.h"

namespace gydra {

namespace testing {

namespace property {

template<typename integer_type>
class PropertyTestForAllIntegersBetween : public PropertyTest<integer_type>  {

 private:
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<integer_type> dist;

 protected:
  integer_type GenerateCase();

 public:
    PropertyTestForAllIntegersBetween(const integer_type a, const integer_type b);

};

template<typename integer_type>
class PropertyTestForAllIntegersOfType : public PropertyTestForAllIntegersBetween<integer_type> {

 public:
  PropertyTestForAllIntegersOfType();

};

template<typename integer_type>
class PropertyTestForAbsolutelyAllIntegersBetween : public gydra::testing::property::PropertyTest<integer_type> {

 private:
  const integer_type minimum;
  const integer_type maximum;
  integer_type current_number;

 protected:
  integer_type GenerateCase();

 public:
  PropertyTestForAbsolutelyAllIntegersBetween(const integer_type a, const integer_type b);

};

class PropertyTestForAbsolutelyAllTenBitsUnsignedIntegers: public PropertyTestForAbsolutelyAllIntegersBetween<unsigned int> {

 public:
  PropertyTestForAbsolutelyAllTenBitsUnsignedIntegers(): PropertyTestForAbsolutelyAllIntegersBetween<unsigned int>(0, 1023) {}

};

}  // namespace property

}  // namespace test

}  // namespace gydra


#include "impl/numeric_property_test_impl.inl"

#endif // GYDRA_TEST_PROPERTY_NUMERIC_H_
