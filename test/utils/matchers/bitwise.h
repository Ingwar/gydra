#ifndef GYDRA_TEST_MATCHERS_BITWISE_H_
#define GYDRA_TEST_MATCHERS_BITWISE_H_

#include <bitset>
#include <climits>

#include <gmock/gmock.h>

#include <test/utils/printing.h>

namespace gydra {

namespace testing {

namespace matchers {

using ::testing::PrintToString;
using gydra::testing::utils::as_ordinal;

template<typename T>
class SizeInfo {

 public:
  static const size_t size_in_bits = sizeof(T) * CHAR_BIT;
};

MATCHER_P(HasNoMoreThanNSignificantBits, n, "has no more than " + PrintToString(n) + " significant bits") {
  const std::bitset< SizeInfo<arg_type>::size_in_bits > arg_bits(arg);

  for (size_t i = n; i < arg_bits.size(); i++) {
    const bool current_bit_is_non_zero = arg_bits.test(i);

    if (current_bit_is_non_zero) {
      *result_listener << "binary representation is " << arg_bits;
      *result_listener << " (the " + as_ordinal(i) + " bit is 1, but should be 0)";
      return false;
    }

  }

  return true;
}

MATCHER_P(HasAtLeastNTrailingZeroBits, n, "has at least " + PrintToString(n) + " trailing zero bits") {
  const std::bitset< SizeInfo<arg_type>::size_in_bits > arg_bits(arg);

  for (size_t i = 0; i < n; i++) {
    const bool current_bit_is_non_zero = arg_bits.test(i);

    if (current_bit_is_non_zero) {
      *result_listener << "binary representation is " << arg_bits;
      *result_listener << " (the " + as_ordinal(i) + " bit is 1, but should be 0)";
      return false;
    }

  }

  return true;
}

MATCHER_P3(HasBitsEqualToTheBitsOf, target, from, to, "has bits from " + PrintToString(from) + " to " + PrintToString(to) + " equal to the bits of " + PrintToString(target)) {
  const std::bitset< SizeInfo<arg_type>::size_in_bits > arg_bits(arg);
  const std::bitset< SizeInfo<arg_type>::size_in_bits > target_bits(target);

  for (size_t i = from; i < to; i++) {
    const bool arg_bit = arg_bits.test(i);
    const bool target_bit = target_bits.test(i);

    if (arg_bit != target_bit) {
      *result_listener << "binary representation is " << arg_bits;
      *result_listener << " (";
      *result_listener << "target is " + PrintToString(target) + " with binary representation " << target_bits;
      *result_listener << ", the " + as_ordinal(i) + " bits are different";
      *result_listener << ")";
      return false;
    }
  }

  return true;
}

}  // namespace matchers

}  // namespace testing

}  // namespace gydra

#endif  // GYDRA_TEST_MATCHERS_BITWISE_H_
