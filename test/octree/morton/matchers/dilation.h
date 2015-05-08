#ifndef GYDRA_TEST_OCTREE_MORTON_MATCHERS_DILATION_H_
#define GYDRA_TEST_OCTREE_MORTON_MATCHERS_DILATION_H_

#include <bitset>
#include <limits>

#include <gmock/gmock.h>
#include <boost/format.hpp>

#include <src/octree/morton.h>

namespace gydra {

namespace octree {

namespace morton {

namespace helpers {

namespace test {

namespace impl {

using gydra::testing::utils::as_ordinal;
using gydra::testing::matchers::SizeInfo;

template<typename T>
class BaseDilationAndUndilationMatcher {

 protected:
  const unsigned int dilation_rank;
  const T original_number;
  const size_t dilated_number_size;
  const std::bitset< SizeInfo<T>::size_in_bits > original_number_bits;

 public:

  BaseDilationAndUndilationMatcher(const T original_n, const size_t dilation_length):
    dilation_rank(3), original_number(original_n), dilated_number_size(dilation_length), original_number_bits(original_n) {
  }

  virtual ~BaseDilationAndUndilationMatcher() {}

  virtual bool MatchAndExplain(const T arg, ::testing::MatchResultListener* result_listener) const = 0;

  // Describes this matcher to an ostream.
  virtual void DescribeTo(::std::ostream* os) const = 0;

  // Describes the negation of this matcher to an ostream.
  virtual void DescribeNegationTo(::std::ostream* os) const = 0;

};

template<typename T>
class BaseDilationMatcher : public BaseDilationAndUndilationMatcher<T> {

 private:
  template<size_t number_of_bits>
  std::string format_common_error_message(const std::bitset<number_of_bits>& arg_bits) const {
    const size_t size_of_number_part_to_dilate = this->dilated_number_size / this->dilation_rank;
    boost::format message("(binary representation is %1%) isn't a correct dilation of first %2% bits of %3% (binary representation is %4%)");
    message % arg_bits % size_of_number_part_to_dilate % this->original_number % this-> original_number_bits;
    return message.str();
  }

  template<size_t number_of_bits>
  void reportErrorOfUnequalBits(
      const std::bitset<number_of_bits>& arg_bits,
      const size_t arg_bit_index,
      const size_t original_number_bit_index,
      ::testing::MatchResultListener* result_listener
    ) const {
    const bool arg_bit = arg_bits.test(arg_bit_index);
    const bool original_number_bit = this->original_number_bits.test(original_number_bit_index);

    boost::format message("%1% - %2% bit of dilated number [%3%] isn't equal to the %4% bit of original number [%5%]");
    message % format_common_error_message(arg_bits) % as_ordinal(arg_bit_index) % arg_bit % original_number_bit_index % original_number_bit;
    *result_listener << message;
  }

  template<size_t number_of_bits>
  void reportErrorOfNonZeroBit(const std::bitset<number_of_bits>& arg_bits, const size_t nonzero_bit_index, ::testing::MatchResultListener* result_listener) const {
    boost::format message("%1% - %2% bit of dilated number should be zero");
    message % format_common_error_message(arg_bits) % as_ordinal(nonzero_bit_index);
    *result_listener << message;
  }

 public:

  BaseDilationMatcher(const T original_n, const size_t dilation_length): BaseDilationAndUndilationMatcher<T>(original_n, dilation_length) {
  }

  bool MatchAndExplain(const T arg, ::testing::MatchResultListener* result_listener) const {
    const std::bitset< SizeInfo<T>::size_in_bits > arg_bits(arg);

    for (size_t i = 0; i < this->dilated_number_size; i ++) {
      const size_t original_number_bit_index = i / this->dilation_rank;

      const bool arg_bit = arg_bits.test(i);
      const bool original_number_bit = this->original_number_bits.test(original_number_bit_index);

      if ((i % this->dilation_rank) == 0) {
        //every third bit (3ith) should be equal to the ith bit of original number
        if (arg_bit != original_number_bit) {
          reportErrorOfUnequalBits(arg_bits, i, original_number_bit_index, result_listener);
          return false;
        }
      } else {
        //all other bits should be equal to 0
        if (arg_bit) {
          reportErrorOfNonZeroBit(arg_bits, i, result_listener);
          return false;
        }
      }
    }

    return true;
  }

};

template <typename T>
class ShortDilationMatcher: public BaseDilationMatcher<T> {

public:

  ShortDilationMatcher(const T original_n): BaseDilationMatcher<T>(original_n, 3 * DILATION_SIZE) {
  }

  virtual void DescribeTo(::std::ostream* os) const {
    *os << "is short dilation of " << this->original_number;
  }

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "isn't short dilation of " << this->original_number;
  }

};

template <typename T>
class DilationMatcher: public BaseDilationMatcher<T> {

public:

  DilationMatcher(const T original_n): BaseDilationMatcher<T>(original_n, 6 * DILATION_SIZE) {
  }

  virtual void DescribeTo(::std::ostream* os) const {
    *os << "is dilation of " << this->original_number;
  }

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "isn't dilation of " << this->original_number;
  }

};

template<typename T>
class BaseUndilationMatcher: public BaseDilationAndUndilationMatcher<T> {

 private:
  template<size_t number_of_bits>
  void reportError(
      const std::bitset<number_of_bits>& arg_bits,
      const size_t arg_bit_index,
      const size_t original_number_bit_index,
      ::testing::MatchResultListener* result_listener
    ) const {
    const bool arg_bit = arg_bits.test(arg_bit_index);
    const bool original_number_bit = this->original_number_bits.test(original_number_bit_index);

    boost::format message("(binary representation is %1%) isn't correct undilation of %2% (binary representation is %3%)"
                          " - %4% bit of actual number [%5%] isn't equal to the %6% bit of original number [%7%]");
    message % arg_bits % this->original_number % this->original_number_bits %
            as_ordinal(arg_bit_index) % arg_bit % as_ordinal(original_number_bit_index) % original_number_bit;
    *result_listener << message;
  }

 public:
  BaseUndilationMatcher(const T original_n, size_t dilation_length): BaseDilationAndUndilationMatcher<T>(original_n, dilation_length) {
  }

  bool MatchAndExplain(const T arg, ::testing::MatchResultListener* result_listener) const {
    const std::bitset< SizeInfo<T>::size_in_bits > arg_bits(arg);

    for (size_t i = 0; i < this->dilated_number_size; i++) {
      const size_t original_number_bit_index = i * this->dilation_rank;

      const bool arg_bit = arg_bits.test(i);
      const bool original_number_bit = this->original_number_bits.test(original_number_bit_index);

      if (arg_bit != original_number_bit) {
        reportError(arg_bits, i, original_number_bit_index, result_listener);
        return false;
      }
    }

    return true;
  }

};

template <typename T>
class ShortUndilationMatcher : public BaseUndilationMatcher<T> {

 public:
  ShortUndilationMatcher(const T original_n): BaseUndilationMatcher<T>(original_n, 10) {
  }

  void DescribeTo(std::ostream* os) const {
    *os << "is short undilation of " << this->original_number;
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "is not short undilation of " << this->original_number;
  }

};

template <typename T>
class UndilationMatcher : public BaseUndilationMatcher<T> {

 public:
  UndilationMatcher(const T original_n): BaseUndilationMatcher<T>(original_n, 20) {
  }

  void DescribeTo(std::ostream* os) const {
    *os << "is undilation of " << this->original_number;
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "is not undilation of " << this->original_number;
  }

};


}  // namespace impl

template<typename T>
inline ::testing::PolymorphicMatcher< impl::ShortDilationMatcher<T> > IsShortDilationOf(const T original_number) {
  return ::testing::MakePolymorphicMatcher(impl::ShortDilationMatcher<T>(original_number));
}

template<typename T>
inline ::testing::PolymorphicMatcher< impl::DilationMatcher<T> > IsDilationOf(const T original_number) {
  return ::testing::MakePolymorphicMatcher(impl::DilationMatcher<T>(original_number));
}

template<typename T>
inline ::testing::PolymorphicMatcher< impl::ShortUndilationMatcher<T> > IsShortUndilationOf(const T original_number) {
  return ::testing::MakePolymorphicMatcher(impl::ShortUndilationMatcher<T>(original_number));
}

template<typename T>
inline ::testing::PolymorphicMatcher< impl::UndilationMatcher<T> > IsUndilationOf(const T original_number) {
  return ::testing::MakePolymorphicMatcher(impl::UndilationMatcher<T>(original_number));
}

} //  namespace test

} //  namespace helpers

} //  namespace morton

} //  namespace octree

} //  namespace gydra

#endif  // GYDRA_TEST_OCTREE_MORTON_MATCHERS_DILATION_H_
