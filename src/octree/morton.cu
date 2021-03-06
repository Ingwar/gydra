#include <cuda_runtime.h>

#include <gydra.h>

#include "morton.h"

namespace gydra {

namespace octree {

namespace morton {

using namespace helpers;

__host__ __device__ MortonCode compute_morton_code(const uint3& coordinates) {
  const uint64 dilated_x = dilate(coordinates.x);
  const uint64 dilated_y = dilate(coordinates.y);
  const uint64 dilated_z = dilate(coordinates.z);
  return dilated_x | dilated_y << 1 | dilated_z << 2;
}

__host__ __device__ uint3 get_coordinates_for_code(const MortonCode code) {
  //Mask for casting Morton code to the form that could be used by `undilate`.
  //It should have ones as 0th, 3rd, 6th ... 60th bits and zeros as others.
  const MortonCode mask = 164703072086692425;

  const MortonCode x_bits = code & mask;
  const MortonCode y_bits = (code >> 1) & mask;
  const MortonCode z_bits = (code >> 2) & mask;

  const unsigned int x = undilate(x_bits);
  const unsigned int y = undilate(y_bits);
  const unsigned int z = undilate(z_bits);

  return make_uint3(x, y, z);
}

namespace helpers{

namespace {

/** Mask for retrieving first 10 bits of integer.
 */
const unsigned int FIRST_TEN_BITS_MASK = (1 << DILATION_SIZE) - 1;

/** Mask for retrieving second 10 bits of integer.
 */
const unsigned int SECOND_TEN_BITS_MASK = ((1 << (2 * DILATION_SIZE)) - 1) ^ FIRST_TEN_BITS_MASK;

const uint64 FIRST_DILATED_BITS_MASK = (1ull << (DILATED_INTEGER_LENGTH / 2)) - 1;

const uint64 SECOND_DILATED_BITS_MASK = ((1ull << DILATED_INTEGER_LENGTH) - 1) ^ FIRST_DILATED_BITS_MASK;

} //  namespace

__host__ __device__ uint64 dilate(const unsigned int number) {
  //Dilate first 10 bits
  const unsigned int first_10_bits = get_first_10_bits_of_number(number);
  const uint64 first_10_bits_dilated = dilate_short(first_10_bits);
  //Dilate second 10 bits
  const unsigned int second_10_bits = get_second_10_bits_of_number(number);
  const uint64 second_10_bits_dilated = dilate_short(second_10_bits);
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

__host__ __device__ unsigned int get_first_10_bits_of_number(const unsigned int number) {
  return number & FIRST_TEN_BITS_MASK;
}

__host__ __device__ unsigned int get_second_10_bits_of_number(const unsigned int number) {
  return (number & SECOND_TEN_BITS_MASK) >> DILATION_SIZE;
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

