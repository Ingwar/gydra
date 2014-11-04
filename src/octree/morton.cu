#include <cuda_runtime.h>

#include <gydra.h>

#include "morton.h"

namespace gydra {

namespace octree {

namespace morton {

using namespace helpers;

__host__ __device__ MortonKey get_morton_key(const uint3& coordinates) {
  const uint64 dilated_x = dilate(coordinates.x);
  const uint64 dilated_y = dilate(coordinates.y);
  const uint64 dilated_z = dilate(coordinates.z);
  return dilated_x | dilated_y << 1 | dilated_z << 2;
}

__host__ __device__ uint3 get_coordinates_for_key(const MortonKey key) {
  const unsigned int x = undilate(key);
  const unsigned int y = undilate(key >> 1);
  const unsigned int z = undilate(key >> 2);
  return make_uint3(x, y, z);
}

namespace helpers {

const uint64 FIRST_DILATED_BITS_MASK = (1ll << (DILATED_INTEGER_LENGTH / 2)) - 1;

const uint64 SECOND_DILATED_BITS_MASK = ((1ll << DILATED_INTEGER_LENGTH) - 1) ^ FIRST_DILATED_BITS_MASK;

__host__ __device__ uint64 dilate(const unsigned int number) {
  //Dilate first 10 bits
  const unsigned int first_10_bits_dilated = dilate_short(number & FIRST_TEN_BITS_MASK);
  //Dilate second 10 bits
  const unsigned int second_10_bits_dilated = dilate_short(number & SECOND_TEN_BITS_MASK);
  //Store result in 64-bit integer
  const uint64 dilated_number = first_10_bits_dilated | (second_10_bits_dilated << (3 * DILATION_SIZE));
  return dilated_number;
}

__host__ __device__ unsigned int undilate(const uint64 dilated_number) {
  //undilate first 10 bits
  const unsigned int first_bits = get_first_bits_of_dilated_number(dilated_number);
  const unsigned int first_10_bits = undilate_short(first_bits);
  //undilate second 10 bits
  const unsigned int second_bits = get_second_bits_of_dilated_number(dilated_number);
  const unsigned int second_10_bits = undilate_short(second_bits);
  //combine bits
  const unsigned int result = first_10_bits | (second_10_bits << DILATION_SIZE);
  return result;
}

__host__ __device__ unsigned int dilate_short(const unsigned int number) {
  unsigned int result = number;
  result = (result * 0x10001) & 0xFF0000FF;
  result = (result * 0x00101) & 0x0F00F00F;
  result = (result * 0x00011) & 0xC30C30C3;
  result = (result * 0x00005) & 0x49249249;
  return result;
}

__host__ __device__ unsigned int undilate_short(const unsigned int number) {
  unsigned int result = (number * 0x00015) & 0x0E070381;
  result = (result * 0x01041) & 0x0FF80001;
  result = (result * 0x40001) & 0x0FFC0000;
  return result >> 18;
}

__host__ __device__ unsigned int get_first_bits_of_dilated_number(const uint64 number) {
  return number & FIRST_DILATED_BITS_MASK;
}

__host__ __device__ unsigned int get_second_bits_of_dilated_number(const uint64 number) {
  return (number & SECOND_DILATED_BITS_MASK) >> (DILATED_INTEGER_LENGTH / 2);
}

}  // namespace helpers

} //  namespace morton

} //  namespace octree

} //  namespace gydra

