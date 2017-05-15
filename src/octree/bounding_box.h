/** @file */
#ifndef GYDRA_OCTREE_BOUNDING_BOX_H_
#define GYDRA_OCTREE_BOUNDING_BOX_H_

#include <cassert>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

#include <gydra.h>


namespace gydra {

/** Namespace containing classes and functions for creating octrees on GPU.
 */
namespace octree {

/** Utils for finding bounding boxes for sets of points.
 */
namespace bounding_box {


/** Class representing rectangular region of space.
 *
 *  It's intended to be used as a bounding box representation.
 */
class BoundingBox {

 public:
  /** Creates new instance of `BoundingBox` with both corners at x=0, y=0, z=0.
   */
  __host__ __device__ BoundingBox() {
    left_bottom_rear = make_real3(0, 0, 0);
    right_top_front = make_real3(0, 0, 0);
  }

  /** Creates new instance of `BoundingBox` from points representing corners.
   *
   * @param left left-bottom-rear corner of the new region
   * @param right right-top-front corner of the new region
   */
  __host__ __device__ BoundingBox(const real3& left, const real3& right): left_bottom_rear(left), right_top_front(right) {
    assert(left.x <= right.x);
    assert(left.y <= right.y);
    assert(left.z <= right.z);
  }

  /** Returns the minimum point of the bounding box.
   *
   * @returns left-bottom-rear corner of the bounding box
   */
  __host__ __device__ real3 get_left_bottom_rear() const {
    return left_bottom_rear;
  }

  /** Returns the maximum point of the bounding box.
   *
   *  @returns right-top-front corner of the BoundingBox
   */
  __host__ __device__ real3 get_right_top_front() const {
    return right_top_front;
  }

  /** Returns width of the bounding box.
   *
   * Width is a size of bounding box along the `x` dimension.
   *
   * @returns the distance between the leftmost and the rightmost
   *  points of the bounding box along the `x` axis
   */
  __host__ __device__ real get_width() const {
    return right_top_front.x - left_bottom_rear.x;
  }

  /** Returns length of the bounding box.
   *
   * Length is a size of bounding box along the `y` dimension.
   *
   * @returns the distance between the rearmost and the front
   *  points of the bounding box along the `y` axis
   */
  __host__ __device__ real get_length() const {
    return right_top_front.y - left_bottom_rear.y;
  }

  /** Returns height of the bounding box.
   *
   * Height is a size of bounding box along the `z` dimension.
   *
   * @returns the distance between the bottommost and the topmost
   *  points of the bounding box along the `z` axis
   */
  __host__ __device__ real get_height() const {
    return right_top_front.z - left_bottom_rear.z;
  }

  /** Returns the length of the longest edge of the bounding box.
   */
  __host__ __device__ real get_longest_edge() const {
    return thrust::max(thrust::max(get_width(), get_height()), get_length());
  }

  /** Returns the length of the shortest edge of the bounding box.
   */
  __host__ __device__ real get_shortest_edge() const {
    return thrust::min(thrust::max(get_width(), get_height()), get_length());
  }

 private:
  real3 left_bottom_rear;

  real3 right_top_front;

};


/**Function object for finding bounding box for given set of points.
 *
 * @tparam InputIterator **Thrust** iterator for the sequence of objects
 *   with `real3 get_position() const` method
 */
template <typename InputIterator>
class BoundingBoxFinder {

 public:

  /** Finds bounding box for given set of points.
   *
   * This method could work with data both in host and device memory.
   *
   * @param first the beginning of the sequence
   * @param last the end of the sequence
   * @returns `BoundingBox` that describes bounding box for given set of points
   */
  virtual BoundingBox operator() (const InputIterator& first, const InputIterator& last) const = 0;

  virtual ~BoundingBoxFinder() {}

};


/** Internal namespace for helper classes.
 *
 * It's not intended for public use an subject to changes.
 */
namespace helpers {


/** Function object for creating degenerate `BoundingBox` from other object position.
 *
 * Term "degenerate" means that both minimal and maximal positions of the `BoundingBox`
 * instance will be equal to the position of the given object.
 *
 * @tparam T class with method `real3 get_position() const`
 */
template <typename T>
class ToBoundingBox : public thrust::unary_function< T, BoundingBox > {

 public:

  ToBoundingBox() {}

  /** Creates `BoundingBox` instance from the given object's position.
   *
   * @param x object whose position will be used for `BoundingBox` instance construction
   * @returns instance of `BoundingBox` with both fields equal to the `x` position
   */
  __host__ __device__ BoundingBox operator()(const T& x) const {
    const BoundingBox result(x.get_position(), x.get_position());
    return result;
  }

};


/** Function object for merging two bounding boxes.
 *
 * "Merging" means finding minimum bounding box that includes both given boxes.
 */
class MergeBoundingBoxes: public thrust::binary_function< BoundingBox, BoundingBox, BoundingBox > {

 public:

  MergeBoundingBoxes() {}

  /** Finds the minimum rectangular circumscribed volume for given regions.
   *
   * @param one rectangular region of space
   * @param another rectangular region of space
   * @returns minimum rectangle that could hold both given regions
   */
  __host__ __device__ BoundingBox operator()(const BoundingBox& one, const BoundingBox& another) const {
    const real3 left_bottom_rear = find_minimum_point(one, another);
    const real3 right_top_front = find_maximum_point(one, another);
    return BoundingBox(left_bottom_rear, right_top_front);
  }

  private:
    /** Finds the minimum point of the given regions.
     *
     * @param one rectangular region of space
     * @param another rectangular region of space
     * @returns minimum point of the given regions
     */
    __host__ __device__ real3 find_minimum_point(const BoundingBox& one, const BoundingBox& another) const {
    const real min_x = thrust::min(one.get_left_bottom_rear().x, another.get_left_bottom_rear().x);
    const real min_y = thrust::min(one.get_left_bottom_rear().y, another.get_left_bottom_rear().y);
    const real min_z = thrust::min(one.get_left_bottom_rear().z, another.get_left_bottom_rear().z);
    return make_real3(min_x, min_y, min_z);
  }

    /** Finds the maximum point of the given regions.
     *
     * @param one rectangular region of space
     * @param another rectangular region of space
     * @returns maximum point of the given regions
     */
  __host__ __device__ real3 find_maximum_point(const BoundingBox& one, const BoundingBox& another) const {
    const real max_x = thrust::max(one.get_right_top_front().x, another.get_right_top_front().x);
    const real max_y = thrust::max(one.get_right_top_front().y, another.get_right_top_front().y);
    const real max_z = thrust::max(one.get_right_top_front().z, another.get_right_top_front().z);
    return make_real3(max_x, max_y, max_z);
  }

};


}  // namespace helpers


/**Function object for finding bounding box for given set of points.
 *
 * @tparam InputIterator **Thrust** iterator for the sequence of objects
 *   with `real3 get_position() const` method
 */
template <typename InputIterator>
class BoundingBoxFinderImplementation: public BoundingBoxFinder<InputIterator> {

 public:
  /** Finds bounding box for given set of points.
   *
   * This method could work with data both in host and device memory.
   *
   * @param first the beginning of the sequence
   * @param last the end of the sequence
   * @returns `BoundingBox` that describes bounding box for given set of points
   */
  BoundingBox operator()(const InputIterator& first, const InputIterator& last) const {
    const helpers::ToBoundingBox<Point> to_bounding_box;
    const helpers::MergeBoundingBoxes merge_boxes;

    const BoundingBox initial_bounding_box = to_bounding_box(*first);

    const BoundingBox box = transform_reduce(first, last, to_bounding_box, initial_bounding_box, merge_boxes);
    return box;
  }

 private:
  typedef typename thrust::iterator_value<InputIterator>::type Point;

};


/** Factory function for standard implementation of `BoundingBoxFinderInterface`.
 *
 * @tparam InputIterator **Thrust** iterator for the sequence of objects
 *   with `real3 get_position() const` method
 */
template <typename InputIterator>
BoundingBoxFinder<InputIterator>* make_bounding_box_finder() {
  return new BoundingBoxFinderImplementation<InputIterator>();
}


}  // namespace bounding_box

}  // namespace octree

}  // namespace gydra

#endif // GYDRA_OCTREE_BOUNDING_BOX_H_
