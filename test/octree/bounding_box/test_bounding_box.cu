#include <gmock/gmock.h>

#include <gydra.h>

#include <src/octree/bounding_box.h>

#include "operators.h"
#include "matchers.h"


namespace gydra {

namespace octree {

namespace bounding_box {

namespace test {


TEST(BoundingBoxTest, default_constructor_should_create_bounding_box_with_corners_at_0_0_0) {
  const BoundingBox bounding_box;

  const real3 left_bottom_rear_corner = bounding_box.get_left_bottom_rear();
  const real3 right_top_front_corner = bounding_box.get_right_top_front();

  ASSERT_THAT(left_bottom_rear_corner, IsEqualToThePointWithFloatCoordinates(make_real3(0, 0, 0)));
  ASSERT_THAT(right_top_front_corner, IsEqualToThePointWithFloatCoordinates(make_real3(0, 0, 0)));
}


TEST(BoundingBoxTest, bounding_box_constructor_should_create_bounding_box_with_corners_at_given_points) {
  const real3 left_bottom_rear_corner_coordinates = make_real3(-15, -1e6, -1e12);
  const real3 right_top_front_corner_coordinates = make_real3(1e8, 37.5, 3.2e15);

  const BoundingBox bounding_box(left_bottom_rear_corner_coordinates, right_top_front_corner_coordinates);

  const real3 left_bottom_rear_corner = bounding_box.get_left_bottom_rear();
  const real3 right_top_front_corner = bounding_box.get_right_top_front();

  ASSERT_THAT(left_bottom_rear_corner, IsEqualToThePointWithFloatCoordinates(left_bottom_rear_corner_coordinates));
  ASSERT_THAT(right_top_front_corner, IsEqualToThePointWithFloatCoordinates(right_top_front_corner_coordinates));
}


TEST(BoundingBoxTest, get_width_should_return_width_of_the_bounding_box) {
  const real width = 13.0;
  const BoundingBox bounding_box(make_real3(0, 0, 0), make_real3(width, 0, 0));

  ASSERT_FLOAT_EQ(width, bounding_box.get_width());
}


TEST(BoundingBoxTest, get_width_should_work_correctly_in_the_presence_of_negative_coordinates) {
  const real width = 17.0;
  const BoundingBox bounding_box(make_real3(-width, 0, 0), make_real3(0, 0, 0));

  ASSERT_FLOAT_EQ(width, bounding_box.get_width());
}


TEST(BoundingBoxTest, get_length_should_return_length_of_the_bounding_box) {
  const real length = 1e7;
  const BoundingBox bounding_box(make_real3(0, 0, 0), make_real3(0, length, 0));

  ASSERT_FLOAT_EQ(length, bounding_box.get_length());
}


TEST(BoundingBoxTest, get_length_should_work_correctly_in_the_presence_of_negative_coordinates) {
  const real length = 3.14e15;
  const BoundingBox bounding_box(make_real3(0, -length, 0), make_real3(0, 0, 0));

  ASSERT_FLOAT_EQ(length, bounding_box.get_length());
}


TEST(BoundingBoxTest, get_height_should_return_height_of_the_bounding_box) {
  const real height = 127.679;
  const BoundingBox bounding_box(make_real3(0, 0, 0), make_real3(0, 0, height));

  ASSERT_FLOAT_EQ(height, bounding_box.get_height());
}


TEST(BoundingBoxTest, get_height_should_work_correctly_in_the_presence_of_negative_coordinates) {
  const real height = 58.45;
  const BoundingBox bounding_box(make_real3(0, 0, -height), make_real3(0, 0, 0));

  ASSERT_FLOAT_EQ(height, bounding_box.get_height());
}


TEST(BoundingBoxTest, get_longest_edge_should_return_the_longest_age_of_the_bounding_box) {
  const BoundingBox bounding_box(make_real3(-1.567e13, -45.9, -1e8), make_real3(0, 0, 0));
  ASSERT_FLOAT_EQ(1.567e13, bounding_box.get_longest_edge());
}


TEST(BoundingBoxTest, get_shortest_edge_should_return_the_shortest_age_of_the_bounding_box) {
  const BoundingBox bounding_box(make_real3(-1.567e13, 0, -1e8), make_real3(0, 22, 0));
  ASSERT_FLOAT_EQ(22, bounding_box.get_shortest_edge());
}


}  //namespace test

}  //namespace bounding_box

}  // namespace octree

}  // namespace gydra
