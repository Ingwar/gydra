#ifndef GYDRA_TEST_PROPERTY_IMPL_H_
#define GYDRA_TEST_PROPERTY_IMPL_H_

#include "property.h"

namespace gydra {

namespace testing {

namespace property {

template<typename fixture_type>
void PropertyTest<fixture_type>::SetUp() {
  for (size_t i = 0; i < data.size(); i++) {
    data.at(i) = GenerateCase();
  }
}

template<typename fixture_type>
PropertyTest<fixture_type>::PropertyTest(size_t intended_number_of_cases): number_of_cases(intended_number_of_cases), data(intended_number_of_cases) {
}

}  // namespace property

}  // namespace test

}  // namespace gydra

#endif // GYDRA_TEST_PROPERTY_IMPL_H_
