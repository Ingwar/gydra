#include<bitset>

#include <boost/format.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>

#include <test/utils/printing.h>

#include "../utils.h"

#include "morton_order.h"


namespace gydra {

namespace octree {

namespace morton {

namespace test {

using ::testing::MatchResultListener;

using gydra::testing::matchers::SizeInfo;

using gydra::testing::utils::as_ordinal;


MortonCodeMatcher::MortonCodeMatcher(const uint3& p): point(p), x_bits(p.x), y_bits(p.y), z_bits(p.z), dilation_rank(3), dilated_integer_length(60) {
}

bool MortonCodeMatcher::MatchAndExplain(MortonKey arg, MatchResultListener* result_listener) const {
  const std::bitset< SizeInfo<MortonKey>::size_in_bits > arg_bits(arg);
  for (size_t i = 0; i < dilated_integer_length / dilation_rank; i++) {

    const bool x_bit = x_bits.test(i);
    const size_t index_of_x_bit_in_the_morton_key = dilation_rank * i;
    const bool x_bit_in_the_morton_key = arg_bits.test(index_of_x_bit_in_the_morton_key);
    if (x_bit != x_bit_in_the_morton_key) {
      reportErrorInTheXBit(arg_bits, index_of_x_bit_in_the_morton_key, i, result_listener);
      return false;
    }

    const bool y_bit = y_bits.test(i);
    const size_t index_of_y_bit_in_the_morton_key = dilation_rank * i + 1;
    const bool y_bit_in_the_morton_key = arg_bits.test(index_of_y_bit_in_the_morton_key);
    if (y_bit != y_bit_in_the_morton_key) {
      reportErrorInTheYBit(arg_bits, index_of_x_bit_in_the_morton_key, i, result_listener);
      return false;
    }

    const bool z_bit = z_bits.test(i);
    const size_t index_of_z_bit_in_the_morton_key = dilation_rank * i + 2;
    const bool z_bit_in_the_morton_key = arg_bits.test(index_of_z_bit_in_the_morton_key);
    if (z_bit != z_bit_in_the_morton_key) {
      reportErrorInTheZBit(arg_bits, index_of_z_bit_in_the_morton_key, i, result_listener);
      return false;
    }

  }
  return true;
}

void MortonCodeMatcher::DescribeTo(::std::ostream* os) const {
  boost::format message("is correct Morton code for point %1%");
  *os << (message % point);
}

void MortonCodeMatcher::DescribeNegationTo(::std::ostream* os) const {
  boost::format message("isn't correct Morton code for point %1%");
  *os << (message % point);
}

template<size_t number_of_bits>
void MortonCodeMatcher::reportErrorInTheXBit(
    const std::bitset<number_of_bits>& arg_bits,
    const size_t morton_code_bit_index,
    const size_t coordinate_bit_index,
    MatchResultListener* result_listener
) const {
  reportError(arg_bits, morton_code_bit_index, coordinate_bit_index, "x", point.x, x_bits, result_listener);
}

template<size_t number_of_bits>
void MortonCodeMatcher::reportErrorInTheYBit(
    const std::bitset<number_of_bits>& arg_bits,
    const size_t morton_code_bit_index,
    const size_t coordinate_bit_index,
    MatchResultListener* result_listener
) const {
  reportError(arg_bits, morton_code_bit_index, coordinate_bit_index, "y", point.y, y_bits, result_listener);
}

template<size_t number_of_bits>
void MortonCodeMatcher::reportErrorInTheZBit(
    const std::bitset<number_of_bits>& arg_bits,
    const size_t morton_code_bit_index,
    const size_t coordinate_bit_index,
    MatchResultListener* result_listener
) const {
  reportError(arg_bits, morton_code_bit_index, coordinate_bit_index, "z", point.z, z_bits, result_listener);
}

template<size_t number_of_bits, size_t number_of_bits_in_coordinate>
void MortonCodeMatcher::reportError(
    const std::bitset<number_of_bits>& arg_bits,
    const size_t morton_code_bit_index,
    const size_t coordinate_bit_index,
    const std::string& coordinate_name,
    const unsigned int coordinate,
    const std::bitset<number_of_bits_in_coordinate>& coordinate_bits,
    MatchResultListener* result_listener
) const {
  boost::format error_message("(binary representation is %1%) - %2% bit isn't equal to the %3% bit of the '%4%' coordinate of original point - "
                              "%5% (binary representation is %6%)");
  error_message % arg_bits % as_ordinal(morton_code_bit_index) % coordinate_bit_index % coordinate_name % coordinate % coordinate_bits;
  *result_listener << error_message;
}


MortonCodeReversionMatcher::MortonCodeReversionMatcher(const MortonKey code):
  morton_code(code), morton_code_bits(code), dilation_rank(3), dilated_integer_length(60) {
}

bool MortonCodeReversionMatcher::MatchAndExplain(const uint3& arg, MatchResultListener* result_listener) const {

  if (!matchReversionOfXCoordinate(arg.x, result_listener)) {
    return false;
  }

  if (!matchReversionOfYCoordinate(arg.y, result_listener)) {
    return false;
  }

  if (!matchReversionOfZCoordinate(arg.z, result_listener)) {
    return false;
  }

  return true;
}

void MortonCodeReversionMatcher::DescribeTo(::std::ostream* os) const {
  boost::format message("is correct coordinates for Morton code %1% (binary representation is %2%)");
  *os << (message % morton_code % morton_code_bits);
}

void MortonCodeReversionMatcher::DescribeNegationTo(::std::ostream* os) const {
  boost::format message("isn't correct coordinates for Morton code %1% (binary representation is %2%)");
  *os << (message % morton_code % morton_code_bits);
}

bool MortonCodeReversionMatcher::matchReversionOfXCoordinate(const unsigned int x, MatchResultListener* result_listener) const {
  errorReporter reporter = boost::bind(&MortonCodeReversionMatcher::reportErrorOfTheXCoordinateReversion, this, _1, _2, _3, result_listener);
  return matchReversionOfTheCoordinate(x, 0, reporter);
}

bool MortonCodeReversionMatcher::matchReversionOfYCoordinate(const unsigned int y, MatchResultListener* result_listener)  const {
  errorReporter reporter = boost::bind(&MortonCodeReversionMatcher::reportErrorOfTheYCoordinateReversion, this, _1, _2, _3, result_listener);
  return matchReversionOfTheCoordinate(y, 1, reporter);
}

bool MortonCodeReversionMatcher::matchReversionOfZCoordinate(const unsigned int z, MatchResultListener* result_listener) const {
  errorReporter reporter = boost::bind(&MortonCodeReversionMatcher::reportErrorOfTheZCoordinateReversion, this, _1, _2, _3, result_listener);
  return matchReversionOfTheCoordinate(z, 2, reporter);
}

bool MortonCodeReversionMatcher::matchReversionOfTheCoordinate(const unsigned int coordinate, const size_t starting_offset, const errorReporter& reporter) const {
  std::bitset< SizeInfo<unsigned int>::size_in_bits > coordinate_bits(coordinate);

  for (size_t morton_code_bit_index = starting_offset; morton_code_bit_index < dilated_integer_length; morton_code_bit_index += dilation_rank) {
    const size_t coordinate_bit_index = morton_code_bit_index / dilation_rank;

    const bool morton_code_bit = morton_code_bits.test(morton_code_bit_index);
    const bool coordinate_bit = coordinate_bits.test(coordinate_bit_index);

    if (coordinate_bit != morton_code_bit) {
      reporter(coordinate, coordinate_bit_index, morton_code_bit_index);
      return false;
    }
  }
  return true;
}

void MortonCodeReversionMatcher::reportErrorOfTheXCoordinateReversion(
    const unsigned int x,
    const size_t coordinate_bit_index,
    const size_t morton_code_bit_index,
    MatchResultListener* result_listener
  ) const {
  reportErrorOfTheCoordinateReversion("x", x, coordinate_bit_index, morton_code_bit_index, result_listener);
}

void MortonCodeReversionMatcher::reportErrorOfTheYCoordinateReversion(
    const unsigned int y,
    const size_t coordinate_bit_index,
    const size_t morton_code_bit_index,
    MatchResultListener* result_listener
  ) const {
  reportErrorOfTheCoordinateReversion("y", y, coordinate_bit_index, morton_code_bit_index, result_listener);
}

void MortonCodeReversionMatcher::reportErrorOfTheZCoordinateReversion(
    const unsigned int z,
    const size_t coordinate_bit_index,
    const size_t morton_code_bit_index,
    MatchResultListener* result_listener
  ) const {
  reportErrorOfTheCoordinateReversion("z", z, coordinate_bit_index, morton_code_bit_index, result_listener);
}

void MortonCodeReversionMatcher::reportErrorOfTheCoordinateReversion(
    const std::string& coordinate_name,
    const unsigned int coordinate,
    const size_t coordinate_bit_index,
    const size_t morton_code_bit_index,
    MatchResultListener* result_listener
  ) const {
  std::bitset< SizeInfo<unsigned int>::size_in_bits > coordinate_bits(coordinate);
  boost::format message("'%1%' coordinate (value - %2%, binary representation - %3%) is reverted incorrectly - "
                        "%4% bit doesn't equal to the %5% bit of the original Morton code");
  *result_listener << (message % coordinate_name % coordinate % coordinate_bits % as_ordinal(coordinate_bit_index) % as_ordinal(morton_code_bit_index));
}


} //  namespace test

} //  namespace morton

} //  namespace octree

} //  namespace gydra
