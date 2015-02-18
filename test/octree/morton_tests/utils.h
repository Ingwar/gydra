#ifndef GYDRA_TEST_OCTREE_MORTON_ORDER_UTILS_H_
#define GYDRA_TEST_OCTREE_MORTON_ORDER_UTILS_H_

#include <ostream>

class uint3;


std::ostream& operator<<(std::ostream& os, const uint3& point);


bool operator==(const uint3& a_point, const uint3& another_point);


#endif //  GYDRA_TEST_OCTREE_MORTON_ORDER_UTILS_H_
