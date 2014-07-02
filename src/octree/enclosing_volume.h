/** @file */
#pragma once

#ifndef GYDRA_OCTREE_ENCLOSING_VOLUME_H_
#define GYDRA_OCTREE_ENCLOSING_VOLUME_H_

#include <cassert>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

#include <gydra.h>

namespace gydra {

/** Namespace containing classes and functions for creating octrees on GPU.
 */
namespace octree {

/** Class representing rectangular region of space.
 *
 *  It's used mainly for representation of circumscribed volume for set of points.
 */
class RectangularRegion {

 private:
  real3 left_bottom_rear;

  real3 rigth_top_front;

 public:
  /** Creates new instance of `RectangularRegion` with both corners at x=0, y=0, z=0.
   */
  __host__ __device__ RectangularRegion() {
    left_bottom_rear = make_real3(0, 0, 0);
    rigth_top_front = make_real3(0, 0, 0);
  }

  /** Creates new instance of `RectangularRegion` from points, corresponding to it's corners.
   *
   * @param left left-bottom-rear corner of the new region
   * @param right right-top-front corner of the new region
   */
  __host__ __device__ RectangularRegion(const real3& left, const real3& right): left_bottom_rear(left), rigth_top_front(right) {
    assert(left.x <= right.x);
    assert(left.y <= right.y);
    assert(left.z <= right.z);
  }

  /** Returns the minimum point of the region.
   *
   * @returns left-bottom-rear corner of the region
   */
  __host__ __device__ const real3& getLeftBottomRear() const {
    return left_bottom_rear;
  }

  /** Returns the maximum point of the region.
   *
   *  @returns right-top-front corner of the region
   */
  __host__ __device__ const real3& getRightTopFront() const {
    return rigth_top_front;
  }

};

/** Internal namespace for helper classes
 */
namespace circumscribed_volume_helpers {

/** Function object for creating degenerate `RectangularRegion` from other object position.
 *
 * Termin "degenerate" means that both minimal and maximal postions of the `RectangularRegion`
 * instance will be equal to the position of the given object.
 *
 * @tparam T class with method `real3 getPosition() const`
 */
template <typename T>
class ToRectangularRegion : public thrust::unary_function< T, RectangularRegion > {

 public:
  /** Creates `RectangularRegion` instance from the given object's position.
   *
   * @param x objeсt whose position will be used to constrution `RectangularRegion` instance
   * @returns instance of `RectangularRegion` with both fields equal to the `x` position
   */
  __host__ __device__ RectangularRegion operator()(const T& x) const {
    RectangularRegion result(x.getPosition(), x.getPosition());
    return result;
  }

};

/** Function object for merging two rectangular regions of space.
 *
 * "Merging" means finding minimum rectangular circumscribed volume
 * for given regions (e.g., minimum rectangle that could hold both of
 * the give regions).
 */
class MergeRegions: public thrust::binary_function< RectangularRegion, RectangularRegion, RectangularRegion > {

 public:
  /** Finds the minimum rectangular circumscribed volume for given regions.
   *
   * @param one rectangular region of space
   * @param another rectangular region of space
   * @returns minimum rectangle that could hold both given regions
   */
  __host__ __device__ RectangularRegion operator()(const RectangularRegion& one, const RectangularRegion& another) const {
    real3 left_bottom_rear = findMinimumPoint(one, another);
    real3 rigth_top_front = findMaximumPoint(one, another);
    RectangularRegion result(left_bottom_rear, rigth_top_front);
    return result;
  }

  private:
    /** Finds the minimum point of the given regions.
     *
     * @param one rectangular region of space
     * @param another rectangular region of space
     * @returns minimum point of the given regions
     */
    __host__ __device__ real3 findMinimumPoint(const RectangularRegion& one, const RectangularRegion& another) const {
    real min_x = thrust::min(one.getLeftBottomRear().x, another.getLeftBottomRear().x);
    real min_y = thrust::min(one.getLeftBottomRear().y, another.getLeftBottomRear().y);
    real min_z = thrust::min(one.getLeftBottomRear().z, another.getLeftBottomRear().z);
    return make_real3(min_x, min_y, min_z);
  }

    /** Finds the maximum point of the given regions.
     *
     * @param one rectangular region of space
     * @param another rectangular region of space
     * @returns maximum point of the given regions
     */
  __host__ __device__ real3 findMaximumPoint(const RectangularRegion& one, const RectangularRegion& another) const {
    real max_x = thrust::max(one.getRightTopFront().x, another.getRightTopFront().x);
    real max_y = thrust::max(one.getRightTopFront().y, another.getRightTopFront().y);
    real max_z = thrust::max(one.getRightTopFront().z, another.getRightTopFront().z);
    return make_real3(max_x, max_y, max_z);
  }

};

}  // namespace circumscribed_volume_helpers

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
RectangularRegion find_circumscribed_volume(const InputIterator& first, const InputIterator& last) {

  typedef typename thrust::iterator_value<InputIterator>::type T;

  circumscribed_volume_helpers::ToRectangularRegion<T> to_region;
  circumscribed_volume_helpers::MergeRegions merge_regions;

  RectangularRegion initial_region = to_region(*first);

  RectangularRegion volume = transform_reduce(first, last, to_region, initial_region, merge_regions);
  return volume;
}

}  // namespace octree

}  // namespace gydra

#endif // GYDRA_OCTREE_ENCLOSING_VOLUME_H_
