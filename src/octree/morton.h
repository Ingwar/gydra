/** @file */

#ifndef GYDRA_OCTREE_MORTON_H_
#define GYDRA_OCTREE_MORTON_H_

#include <boost/static_assert.hpp>
#include <limits>

//Forward declaration of CUDA uint3
struct uint3;

namespace gydra {

namespace octree {

/** Functions for Morton order calculation.
 */
namespace morton {

//Type alias for unsigned 64-bit integer. CUDA doesn't currently support
// C++11, therefore we need to use this alias
typedef unsigned long long uint64;

//Check (at compile time) that uint64 is really at least 64 bit long
BOOST_STATIC_ASSERT_MSG(std::numeric_limits<uint64>::digits >= 64, "64-bit unsigned integer is not available");

typedef uint64 MortonKey; 

/** Compute Morton Z-order for point with given coordinates.
 *
 * @warning Only first 20 bits of each coordinate will be used
 * for calculation. Therefore, different points with coordinates that 
 * have same first 20 bits (e.g., {0, 0, 0} and {1048576, 0, 0}) 
 * will have same Morton code.
 *
 * @see get_coordinates_for_key for finding coordinates by known Morton code 
 *
 * @param coordinates integer coordinates of point
 * @returns Morton code for given coordinates
 */
__host__ __device__ MortonKey get_morton_key(const uint3& coordinates);

/** Compute coordinates for given Morton code.
 *
 * @warning Because only first 20 bits of coordinates are used for
 * Morton code calculation, statement 
 * `get_coordinates_for_key(get_morton_key(point)) == point`
 * will be true only for points with every coordinate less then \f$2^{20}\f$
 *
 * @see get_morton_key for calculating Morton code for given coordinates
 *
 * @param key Morton code  
 * @returns coordinates corresponding to give Morton code
 */
__host__ __device__ uint3 get_coordinates_for_key(const MortonKey key);

/** Miscellaneous functions, constants and typedefs used in computation of Morton order.
 *  These helpers are not intended for public use and they are put in the header
 *  only for testing purposes. 
 */
namespace helpers {

const unsigned int DILATION_SIZE = 10;

const unsigned int DILATED_INTEGER_LENGTH = 60;

/** Extract first 10 bits of the unsigned integer.
 *
 * @param number
 * @return unsigned integer with first 10 bits equal to the first 10 bits of arguments and others set to zero
 */
__host__ __device__ unsigned int get_first_10_bits_of_number(const unsigned int number);

/** Extract second 10 bits of unsigned integer.
 *
 * This function extracts 10 bits of argument from 10th to 19th (counting from zero)
 * and returns it as a unsigned number so that 0th bit of result is equal to the 10th bit of
 * argument, 1st bit of result is equal to the 11st bit of argument and so on till the 9th bit
 * of result and 19th bit of argument. All other bits of result should be equal to zero.
 *
 * **Example**:
 *
 * @code
 *
 * const unsigned int number = 5102; // binary representation is "1010000000000"
 * const unsigned int result = get_first_10_bits_of_number(number);
 * cout << result; // prints "5", binary representation is "101"
 *
 * @endcode

 * @param number
 * @return unsigned integer with first 10 bits equal to the second 10 bits of argument and others set to zero
 */
__host__ __device__ unsigned int get_second_10_bits_of_number(const unsigned int number);

/** Extract first 30 bits of the 64-bit unsigned integer.
 *
 * @param number   
 * @return unsigned integer with first 30 bits equals to first 30 bits of argument and others set to zero
 */
__host__ __device__ unsigned int get_first_bits_of_dilated_number(const uint64 number);

/** Extract second 30 bits of the 64-bit unsigned integer.
 *
 * This function extracts 30 bits of argument from 30th to 59th (counting from zero)
 * and returns it as a unsigned number so that 0th bit of result is equal to the 30th bit of
 * argument, 1st bit of result is equal to the 31st bit of argument and so on till the 29th bit
 * of result and 59th bit of argument. All other bits of result should be equal to zero.
 *
 * **Example**:
 *
 * @code
 *
 * const uint64 number = 3221225472; // binary representation is "11000000000000000000000000000000"
 * const unsigned int result = get_first_bits_of_dilated_number(number);
 * cout << result; // prints "3", binary representation is "11"
 *
 * @endcode
 *
 * @param number
 * @return unsigned integer with first 30 bits equal to the second 30 bits of argument and others set to zero
 *
 */
__host__ __device__ unsigned int get_second_bits_of_dilated_number(const uint64 number);

/** Interleaves first 10 bits of integer with zeros.
 *
 * This function takes first 10 bytes of the integer, 
 * insert two zeros between them and returns result as unsigned
 * integer.
 *
 * **Example**:
 *
 * @code
 *
 * const unsigned int number = 11; // binary representation is "1011"
 * const unsigned int result = dilate_short(number);
 * cout << result; // prints "521", binary representation is "1000001001"
 *
 * @endcode
 *
 * @warning Values of result bits from 30th to end are not specified!
 *
 * @see undilate_short --- reverse operation
 *
 * @param number
 * @returns 
 */
__host__ __device__ unsigned int dilate_short(const unsigned int number);

/** Reverse dilation (if orifinal number was less than \f$2^{10}\f$).
 *
 * This function takes each third bit of unsigned integer from 0th to 27th
 * (i.e., 0th, 3rd, 6th, ... , 27th) and returns them as unsigned integer. All other
 * bits of input number are ignored.
 *
 * **Example**:
 *
 * @code
 *
 * const unsigned int number = 521; // binary representation is "1000001001"
 * const uint64 result = undilate_short(number);
 * cout << result; // prints "11", binary representation is "1011"
 *
 * @endcode

 *
 * @see dilate_short --- reverse operation
 *
 * @param number
 * @returns dilated integer 
 */
__host__ __device__ unsigned int undilate_short(const unsigned int number);

/** Interleaves first 20 butes of integer with zeros.
 *
 *  This function takes first 20 bytes of the integer, 
 *  insert two zeros between them and returns result as 64-bit unsigned
 *  integer.
 *
 * **Example**:
 *
 * @code
 *
 * const unsigned int number = 7; // binary representation is "111"
 * const uint64 result = dilate(number);
 * cout << result; // prints "73", binary representation is "1001001"
 *
 * @endcode
 *
 *  @see undilate --- reverse operation
 *
 * @param number
 * @returns dilated integer 
 */
__host__ __device__ uint64 dilate(const unsigned int number);

/** Reverse dilation (if orifinal number was less than \f$2^{20}\f$).
 *
 * This function takes each third bit of 64-bit unsigned integer from 0th to 57th
 * (i.e., 0th, 3rd, 6th, ... , 57th) and returns them as unsigned integer. All other
 * bits of input number are ignored.
 *
 * **Example**:
 *
 * @code
 *
 * const unsigned int number = 73; // binary representation is "1001001"
 * const uint64 result = undilate(number);
 * cout << result; // prints "7", binary representation is "111"
 *
 * @endcode
 *
 * @see dilate --- reverse operation
 *
 * @param number
 */
__host__ __device__ unsigned int undilate(const uint64 dilated_number);

}  // namespace helpers

}  // namespace morton

}  // namespace octree

}  // namespace gydra

#endif  // GYDRA_OCTREE_MORTON_H_
