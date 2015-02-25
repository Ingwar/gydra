#include <limits>
#include <ctime>

#include <gtest/gtest.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include <gydra.h>

#include <src/octree/bounding_box.h>


namespace gydra {

namespace octree {

namespace bounding_box {

namespace test {


class Point {

  real3 position;

 public:
 __host__ __device__ Point () {
    position = make_real3(0, 0, 0);
  }

  __host__ __device__ Point(real x, real y, real z) {
    position = make_real3(x, y, z);
  }

  __host__ __device__ real3 getPosition() const {
    return position;
  }

};


class FindBoundingBoxTest : public testing::Test {

 public:
  FindBoundingBoxTest() : N(10000) {
    rng = thrust::default_random_engine(std::time(NULL));

    const real low_coordinate_boundary = std::numeric_limits<real>::min();
    const real high_coordinate_boundary = std::numeric_limits<real>::max();

    distribution = thrust::uniform_real_distribution<real>(low_coordinate_boundary, high_coordinate_boundary);

    xs = thrust::host_vector<real>(N);
    ys = thrust::host_vector<real>(N);
    zs = thrust::host_vector<real>(N);

    data = thrust::host_vector<Point>(N);
  }

 protected:
  real min_x;
  real min_y;
  real min_z;

  real max_x;
  real max_y;
  real max_z;

  thrust::host_vector<Point> data;

  virtual void SetUp() {
    generatePoints();
    findMinimumCoordinates();
    findMaximumCoordinates();
  }

 private:
  const unsigned int N;

  thrust::default_random_engine rng;

  thrust::uniform_real_distribution<real> distribution;

  thrust::host_vector<real> xs;
  thrust::host_vector<real> ys;
  thrust::host_vector<real> zs;

  void generatePoints() {
    for (size_t i = 0; i < N; i++) {
      real x = xs[i] = distribution(rng);
      real y = ys[i] = distribution(rng);
      real z = zs[i] = distribution(rng);
      Point point(x, y, z);

      data[i] = point;
    }
  }

  void findMinimumCoordinates() {
    min_x = *thrust::min_element(xs.begin(), xs.end());
    min_y = *thrust::min_element(ys.begin(), ys.end());
    min_z = *thrust::min_element(zs.begin(), zs.end());
  }

  void findMaximumCoordinates() {
    max_x = *thrust::max_element(xs.begin(), xs.end());
    max_y = *thrust::max_element(ys.begin(), ys.end());
    max_z = *thrust::max_element(zs.begin(), zs.end());
  }

};


TEST_F(FindBoundingBoxTest, should_work_on_GPU) {

    thrust::device_vector<Point> device_data = data;

    BoundingBox bounding_box = find_bounding_box(device_data.begin(), device_data.end());

    real3 left_bottom_rear = bounding_box.getLeftBottomRear();

    real3 rigth_top_front = bounding_box.getRightTopFront();

    ASSERT_FLOAT_EQ(min_x, left_bottom_rear.x);
    ASSERT_FLOAT_EQ(min_y, left_bottom_rear.y);
    ASSERT_FLOAT_EQ(min_z, left_bottom_rear.z);

    ASSERT_FLOAT_EQ(max_x, rigth_top_front.x);
    ASSERT_FLOAT_EQ(max_y, rigth_top_front.y);
    ASSERT_FLOAT_EQ(max_z, rigth_top_front.z);
}


TEST_F(FindBoundingBoxTest, should_work_on_CPU) {

    BoundingBox bounding_box = find_bounding_box(data.begin(), data.end());

    real3 left_bottom_rear = bounding_box.getLeftBottomRear();

    real3 rigth_top_front = bounding_box.getRightTopFront();

    ASSERT_FLOAT_EQ(min_x, left_bottom_rear.x);
    ASSERT_FLOAT_EQ(min_y, left_bottom_rear.y);
    ASSERT_FLOAT_EQ(min_z, left_bottom_rear.z);

    ASSERT_FLOAT_EQ(max_x, rigth_top_front.x);
    ASSERT_FLOAT_EQ(max_y, rigth_top_front.y);
    ASSERT_FLOAT_EQ(max_z, rigth_top_front.z);
}


}  //namespace test

}  //namespace bounding_box

}  // namespace octree

}  // namespace gydra
