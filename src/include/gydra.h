/** @file */
#pragma once

#ifndef GYDRA_H_
#define GYDRA_H_

/** Main namespace for **Gydra** project.
 */
namespace gydra {

/** @defgroup precision Numerical precision
 *
 * Typedefs and macros for easy switching between single and double precision floats.
 *
 * Default choice is single precision floating point numbers. To switch to the
 * double precision numbers compile with the `GYDRA_DOUBLE_PRECISION` macro defined.
 *
 * @addtogroup precision
 *  @{
 */
#ifndef GYDRA_DOUBLE_PRECISION
typedef float real;
typedef float3 real3;
#define make_real3 make_float3
#else
typedef double real;
typedef double3 real3;
#define make_real3 make_double3
#endif  // GYDRA_DOUBLE_PRECISION
/** @}*/

}  // namespace gydra

#endif  // GYDRA_H_
