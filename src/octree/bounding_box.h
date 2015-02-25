/** @file */
#ifndef GYDRA_OCTREE_ENCLOSING_VOLUME_H_
#define GYDRA_OCTREE_ENCLOSING_VOLUME_H_

#include <cassert>

#include <cuda_runtime.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

#include <gydra.h>

namespace gydra {

/** Namespace containing classes and functions for creating octrees on GPU.
 */
namespace octree {

namespace bounding_box {


/** Class representing rectangular region of space.
 *
 *  It's intended to be used as a bounding box representation.
 */
class BoundingBox {

 private:
  real3 left_bottom_rear;

  real3 rigth_top_front;

 public:
  /** Creates new instance of `BoundingBox` with both corners at x=0, y=0, z=0.
   */
  __host__ __device__ BoundingBox() {
    left_bottom_rear = make_real3(0, 0, 0);
    rigth_top_front = make_real3(0, 0, 0);
  }

  /** Creates new instance of `BoundingBox` from points representing corners.
   *
   * @param left left-bottom-rear corner of the new region
   * @param right right-top-front corner of the new region
   */
  __host__ __device__ BoundingBox(const real3& left, const real3& right): left_bottom_rear(left), rigth_top_front(right) {
    assert(left.x <= right.x);
    assert(left.y <= right.y);
    assert(left.z <= right.z);
  }

  /** Returns the minimum point of the bounding box.
   *
   * @returns left-bottom-rear corner of the bounding box
   */
  __host__ __device__ const real3& getLeftBottomRear() const {
    return left_bottom_rear;
  }

  /** Returns the maximum point of the bounding box.
   *
   *  @returns right-top-front corner of the BoundingBox
   */
  __host__ __device__ const real3& getRightTopFront() const {
    return rigth_top_front;
  }

};

/** Internal namespace for helper classes
 */
namespace helpers {

/** Function object for creating degenerate `BoundingBox` from other object position.
 *
 * Termin "degenerate" means that both minimal and maximal postions of the `BoundingBox`
 * instance will be equal to the position of the given object.
 *
 * @tparam T class with method `real3 getPosition() const`
 */
template <typename T>
class ToBoundingBox : public thrust::unary_function< T, BoundingBox > {

 public:

  ToBoundingBox() {}

  /** Creates `BoundingBox` instance from the given object's position.
   *
   * @param x objeсt whose position will be used for `BoundingBox` instance constraction
   * @returns instance of `BoundingBox` with both fields equal to the `x` position
   */
  __host__ __device__ BoundingBox operator()(const T& x) const {
    const BoundingBox result(x.getPosition(), x.getPosition());
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
    const real3 left_bottom_rear = findMinimumPoint(one, another);
    const real3 rigth_top_front = findMaximumPoint(one, another);
    return BoundingBox(left_bottom_rear, rigth_top_front);
  }

  private:
    /** Finds the minimum point of the given regions.
     *
     * @param one rectangular region of space
     * @param another rectangular region of space
     * @returns minimum point of the given regions
     */
    __host__ __device__ real3 findMinimumPoint(const BoundingBox& one, const BoundingBox& another) const {
    const real min_x = thrust::min(one.getLeftBottomRear().x, another.getLeftBottomRear().x);
    const real min_y = thrust::min(one.getLeftBottomRear().y, another.getLeftBottomRear().y);
    const real min_z = thrust::min(one.getLeftBottomRear().z, another.getLeftBottomRear().z);
    return make_real3(min_x, min_y, min_z);
  }

    /** Finds the maximum point of the given regions.
     *
     * @param one rectangular region of space
     * @param another rectangular region of space
     * @returns maximum point of the given regions
     */
  __host__ __device__ real3 findMaximumPoint(const BoundingBox& one, const BoundingBox& another) const {
    const real max_x = thrust::max(one.getRightTopFront().x, another.getRightTopFront().x);
    const real max_y = thrust::max(one.getRightTopFront().y, another.getRightTopFront().y);
    const real max_z = thrust::max(one.getRightTopFront().z, another.getRightTopFront().z);
    return make_real3(max_x, max_y, max_z);
  }

};

}  // namespace helpers

/** Finds rectangular circumscribed volume for given set of points.
 *
 * This function could work with data both in host and device memory.
 *
 * @tparam InputIterator **Thrust** iterator for the sequnce of objects
 *   with `real3 getPosition() const` method
 *
 * @param first the beginning of the sequence
 * @param last the end of the sequence
 * @returns `RectangularRegion` instance representing circumscribed volume
 */
template <typename InputIterator>
BoundingBox find_bounding_box(const InputIterator& first, const InputIterator& last) {

  typedef typename thrust::iterator_value<InputIterator>::type Point;

  const helpers::ToBoundingBox<Point> to_bounding_box;
  const helpers::MergeBoundingBoxes merge_boxes;

  const BoundingBox initial_bounding_box = to_bounding_box(*first);

  const BoundingBox box = transform_reduce(first, last, to_bounding_box, initial_bounding_box, merge_boxes);
  return box;
}


}  // namespace bounding_box

}  // namespace octree

}  // namespace gydra

#endif // GYDRA_OCTREE_ENCLOSING_VOLUME_H_
