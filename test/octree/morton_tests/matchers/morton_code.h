#ifndef GYDRA_TEST_OCTREE_MORTON_ORDER_MATCHERS_H_
#define GYDRA_TEST_OCTREE_MORTON_ORDER_MATCHERS_H_

#include <gmock/gmock.h>
#include <boost/function.hpp>

#include <src/octree/morton.h>

#include <test/utils/matchers/bitwise.h>

namespace gydra {

namespace octree {

namespace morton {

namespace test {


class MortonCodeMatcher : public ::testing::MatcherInterface<MortonCode> {

 public:

  MortonCodeMatcher(const uint3& p);

  bool MatchAndExplain(MortonCode arg, ::testing::MatchResultListener* result_listener) const;

  void DescribeTo(::std::ostream* os) const;

  void DescribeNegationTo(::std::ostream* os) const;

 private:
  const uint3 point;
  const std::bitset< gydra::testing::matchers::SizeInfo< unsigned int>::size_in_bits > x_bits;
  const std::bitset< gydra::testing::matchers::SizeInfo< unsigned int>::size_in_bits > y_bits;
  const std::bitset< gydra::testing::matchers::SizeInfo< unsigned int>::size_in_bits > z_bits;

  const unsigned int dilation_rank;
  const unsigned int dilated_integer_length;

  template<size_t number_of_bits, size_t number_of_bits_in_coordinate>
  void reportError(
      const std::bitset<number_of_bits>& arg_bits,
      const size_t morton_code_bit_index,
      const size_t coordinate_bit_index,
      const std::string& coordinate_name,
      const unsigned int coordinate,
      const std::bitset<number_of_bits_in_coordinate>& coordinate_bits,
      ::testing::MatchResultListener* result_listener
  ) const;

  template<size_t number_of_bits>
  void reportErrorInTheXBit(
      const std::bitset<number_of_bits>& arg_bits,
      const size_t morton_code_bit_index,
      const size_t coordinate_bit_index,
      ::testing::MatchResultListener* result_listener
  ) const;

  template<size_t number_of_bits>
  void reportErrorInTheYBit(
      const std::bitset<number_of_bits>& arg_bits,
      const size_t morton_code_bit_index,
      const size_t coordinate_bit_index,
      ::testing::MatchResultListener* result_listener
  ) const;

  template<size_t number_of_bits>
  void reportErrorInTheZBit(
      const std::bitset<number_of_bits>& arg_bits,
      const size_t morton_code_bit_index,
      const size_t coordinate_bit_index,
      ::testing::MatchResultListener* result_listener
  ) const;

};


class MortonCodeReversionMatcher: public ::testing::MatcherInterface<const uint3&> {

 public:
  MortonCodeReversionMatcher(const MortonCode code);

  bool MatchAndExplain(const uint3& arg, ::testing::MatchResultListener* result_listener) const;

  void DescribeTo(::std::ostream* os) const;

  void DescribeNegationTo(::std::ostream* os) const;

 private:

  typedef boost::function<void (const unsigned int, const size_t, const size_t)> errorReporter;

  MortonCode morton_code;
  const std::bitset< gydra::testing::matchers::SizeInfo< MortonCode >::size_in_bits > morton_code_bits;

  const unsigned int dilation_rank;
  const unsigned int dilated_integer_length;

  bool matchReversionOfXCoordinate(const unsigned int x, ::testing::MatchResultListener* result_listener) const;
  bool matchReversionOfYCoordinate(const unsigned int y, ::testing::MatchResultListener* result_listener) const;
  bool matchReversionOfZCoordinate(const unsigned int z, ::testing::MatchResultListener* result_listener) const;
  bool matchReversionOfTheCoordinate(
    const unsigned int coordinate,
    const size_t starting_offset,
    const errorReporter& reporter
  ) const;

  void reportErrorOfTheXCoordinateReversion(
    const unsigned int x,
    const size_t coordinate_bit_index,
    const size_t morton_code_bit_index,
    ::testing::MatchResultListener* result_listener
  ) const;

  void reportErrorOfTheYCoordinateReversion(
    const unsigned int y,
    const size_t coordinate_bit_index,
    const size_t morton_code_bit_index,
    ::testing::MatchResultListener* result_listener
  ) const;

  void reportErrorOfTheZCoordinateReversion(
    const unsigned int z,
    const size_t coordinate_bit_index,
    const size_t morton_code_bit_index,
    ::testing::MatchResultListener* result_listener
  ) const;

  void reportErrorOfTheCoordinateReversion(
    const std::string& coordinate_name,
    const unsigned int coordinate,
    const size_t coordinate_bit_index,
    const size_t morton_code_bit_index,
    ::testing::MatchResultListener* result_listener
  ) const;

};


inline ::testing::Matcher<MortonCode> IsCorrectMortonCodeForPoint(const uint3& point) {
  return ::testing::MakeMatcher(new MortonCodeMatcher(point));
}


inline ::testing::Matcher<const uint3&> IsCorrectReversionOfMortonCode(const MortonCode morton_code) {
  return ::testing::MakeMatcher(new MortonCodeReversionMatcher(morton_code));
}


} //  namespace test

} //  namespace morton

} //  namespace octree

} //  namespace gydra

#endif //  GYDRA_TEST_OCTREE_MORTON_ORDER_MATCHERS_H_
