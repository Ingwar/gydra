#include <sstream>
#include <cassert>

#include "printing.h"

namespace gydra {

namespace testing {

namespace utils {

namespace {

/** Returns suffix of ordinal corresponding to given digit.
 *
 * @param digit unsigned integer between 0 and 9 (inclusively)
 * @returns suffix of ordinal corresponding to given digit ("st", "snd")
 */
std::string get_ordinal_suffix_for_digit(const size_t digit) {
  assert((0 <= digit) && (digit <= 9));

  switch(digit) {
    case 1: return "st";
    case 2: return "nd";
    case 3: return "rd";
    default: return "th";
  }
}

std::string size_to_string(const size_t number) {
  std::ostringstream result;
  result << number;
  return result.str();
}

} // namespace

/** Returns suffix of ordinal corresponding to given number. 
 */
std::string get_ordinal_suffix(size_t number) {
  const size_t tens = (number / 10) % 10;
  const size_t ones = number % 10;

  if (tens == 1) {
    return "th";
  } else {
    return get_ordinal_suffix_for_digit(ones);
  }
}

/** Returns ordinal corresponding to given number.
 *
 * @param number
 * @returns string containing ordinal as number with suffix (i.e., "1st", "7th")
 */
std::string as_ordinal(const size_t number) {
  return size_to_string(number) + get_ordinal_suffix(number);
}

}  // namespace utils

}  // namespace testing

}  // namespace gydra
