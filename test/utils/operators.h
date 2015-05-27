#ifndef GYDRA_TEST_UTILS_OPERATORS_H_
#define GYDRA_TEST_UTILS_OPERATORS_H_

#include <ostream>

#include <gydra.h>


class uint3;


std::ostream& operator<<(std::ostream& os, const uint3& point);


bool operator==(const uint3& a_point, const uint3& another_point);


std::ostream& operator<<(std::ostream& os, const gydra::real3& point);


bool operator==(const gydra::real3& a_point, const gydra::real3& another_point);


#endif  // GYDRA_TEST_UTILS_OPERATORS_H_
