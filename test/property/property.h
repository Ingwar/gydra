#ifndef GYDRA_TEST_PROPERTY_H_
#define GYDRA_TEST_PROPERTY_H_

#include<vector>

#include<gtest/gtest.h>

#define PROPERTY_TEST(test_case_name, test_name, property)\
class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public test_case_name {\
 public:\
  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}\
 private:\
  virtual void TestBody();\
  template<typename property##_type>\
  void PropertyBody(const property##_type property);\
  static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;\
  GTEST_DISALLOW_COPY_AND_ASSIGN_(\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
};\
\
::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)\
  ::test_info_ =\
    ::testing::internal::MakeAndRegisterTestInfo(\
        #test_case_name, #test_name, NULL, NULL, \
        (::testing::internal::GetTypeId<test_case_name>()), \
        test_case_name::SetUpTestCase, \
        test_case_name::TearDownTestCase, \
        new ::testing::internal::TestFactoryImpl<\
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>);\
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody() {\
  for (size_t i = 0; i < data.size(); i ++) {\
    PropertyBody(data[i]);\
    if (HasFatalFailure()) {\
      return;\
    }\
  }\
}\
template<typename property##_type>\
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::PropertyBody(const property##_type property)

namespace gydra {

namespace testing {

namespace property {

template<typename fixture_type>
class PropertyTest : public ::testing::Test {

 protected:
  const size_t number_of_cases;
  std::vector<fixture_type> data;

  virtual void SetUp();
  virtual fixture_type GenerateCase() = 0;

 public:
  PropertyTest(size_t intended_number_of_cases = 200);

};


}  // namespace property

}  // namespace test

}  // namespace gydra

#include "property_test_impl.inl"

#endif  // GYDRA_TEST_PROPERTY_H_

