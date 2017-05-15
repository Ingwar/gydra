/** @file.
 *
 * Basic tools needed to define property tests.
 *
 * Property testing is a style of testing when we define a data domain
 * (e.g., even numbers, nonempty strings, nonempty stacks etc),
 * statement that holds for all values from that domain
 * (e.g., remainder of division of any even integer by 2 is 0,
 * any string concatenated with empty string is equal to the original string,
 * size of any non-empty stack is greater than 0 etc), generate data
 * from that domain and check whether statement is indeed true for all that data.
 *
 * Ideally, we should use the whole domain for testing, but it often impossible to do
 * in practice, so we usually generate a subsample from the domain for actual testing.
 *
 * In our implementation, domain definition and test data generation are handled by
 * the subclasses of gydra::testing::property::PropertyTest, while predicate statement
 * definition and actual checks are handled by the usage of #PROPERTY_TEST macro.
 *
 */
#ifndef GYDRA_TEST_PROPERTY_H_
#define GYDRA_TEST_PROPERTY_H_

#include<vector>

#include<gtest/gtest.h>

/** Defines property test.
 *
 * This macro is used to define a statement that we want to check
 * on all data from the some domain (defined externally as a subclass of gydra::testing::property::PropertyTest).
 * It generates all support code needed to perform checks on the generated data.
 *
 * This macro is intended to be very similar in use to the Google Test TEST_F,
 * but it also accept as parameter name under which example of a test data would
 * be accessible inside test body (as a read-only value).
 *
 * @param test_case_name Name of the test case. It should be a name of existing class
 *        derived from gydra::testing::property::PropertyTest.
 * @param test_name Name of the test, describing the statement tht we want to
 *        check on the domain, defined by test_case_name class.
 * @param test_data_example name under which example of test data would be available
 *        in test body. **Note**: inside test body it would be a const value.
 *
 * Usage example:
 *
 * @code
 *
 * class PropertyOfAllOddIntegers: public PropertyTest<unsigned int> {
 *
 *  protected:
 *   // some property setup, see docs for PropertyTest for details
 *    unsigned int GenerateCase() override {
 *    ...
 *  }
 *
 * }
 *
 * PROPERTY_TEST(PropertyOfAllOddIntegers, all_odd_integers_should_have_remainder_of_1_when_divided_by_2, i) {
 *   const unsigned int remainder = i % 2;
 *   ASSERT_EQ(remainder, 1);
 * }
 *
 * @endcode
 *
 * @see gydra::testing::property::PropertyTest for details on how to create a new property test
 * @see property.h for the general info about property testing.
 * @see <a href="https://github.com/google/googletest/blob/master/googletest/docs/Primer.md">Google Test manual</a>
 *      for the info about test definitions and assertions.
 *
 * @note
 * This macros is based on the TEST_F macros from Google Test.
 */
#define PROPERTY_TEST(test_case_name, test_name, test_data_example)\
class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public test_case_name {\
 public:\
  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}\
 private:\
  virtual void TestBody();\
  /*That method is added by our implementation */ \
  template<typename test_data_example##_type>\
  void PropertyBody(const test_data_example##_type property);\
  static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;\
  GTEST_DISALLOW_COPY_AND_ASSIGN_(\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
};\
\
::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)\
  ::test_info_ =\
    ::testing::internal::MakeAndRegisterTestInfo(\
        #test_case_name, #test_name, NULL, NULL, \
        ::testing::internal::CodeLocation(__FILE__, __LINE__), \
        (::testing::internal::GetTypeId<test_case_name>()), \
        test_case_name::SetUpTestCase, \
        test_case_name::TearDownTestCase, \
        new ::testing::internal::TestFactoryImpl<\
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>);\
/*Original macro ends here, because the body*/\
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody() {\
  for (size_t i = 0; i < data.size(); i ++) {\
    PropertyBody(data[i]);\
    if (HasFatalFailure()) {\
      return;\
    }\
  }\
}\
/*That method body is provided by user, similar to the TestBody of regular test classes */\
template<typename test_data_example##_type>\
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::PropertyBody(const test_data_example##_type test_data_example)

namespace gydra {

namespace testing {

/** Namespace containing base classes needed to define property tests. */
namespace property {

/** Abstract base class for test fixtures used in property testing.
 *
 * It's subclasses are used to define testing domain and generate
 * testing data examples from it. The tests themselves are defined
 * with #PROPERTY_TEST macro.
 *
 * @tparam fixture_type type of the single example
 *
 * Usage example:
 *
 * @code
 *
 * class PropertyOfAllOddIntegers: public PropertyTest<unsigned int> {
 *  protected:
 *   unsigned int GenerateCase() override {
 *     unsigned int random_integer = static_cast<unsigned int>(rand());
 *
 *     if (random_integer % 2 == 0) {
 *       random_integer += 1;
 *     }
 *
 *     return random_integer;
 *   }
 * }
 *
 * @endcode
 *
 * @see PROPERTY_TEST for info about how to use this class to actually
 * define property tests.
 *
 * @see property.h for the general info about property testing.
 *
 */
template<typename fixture_type>
class PropertyTest : public ::testing::Test {

 protected:
  /** Number of test data examples that should be generated. */
  const size_t number_of_cases;

  /** Vector containing generated test data. */
  std::vector<fixture_type> data;

  virtual void SetUp();

  /** Generate single example of test data.
   *
   * Override this method to define generation process.
   *
   * @return generated example.
   */
  virtual fixture_type GenerateCase() = 0;

 public:
  /** Initialize base fixture and storage for the generated data.
   *
   * @param intended_number_of_cases number of test data examples that we planning to generate.
   */
  PropertyTest(const size_t intended_number_of_cases = 200);
  virtual ~PropertyTest() {}

};


}  // namespace property

}  // namespace test

}  // namespace gydra

#include "impl/property_test_impl.inl"

#endif  // GYDRA_TEST_PROPERTY_H_

