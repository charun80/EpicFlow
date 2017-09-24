#ifndef __SOLVER_H_
#define __SOLVER_H_


#include "image.h"


#ifdef __cplusplus
namespace ccore {
extern "C" {
#endif


// Perform n iterations of the sor_coupled algorithm for a system of the form as described in opticalflow.c
void sor_coupled( image_t *du,
                  image_t *dv,
                  image_t *a11,
                  image_t *a12,
                  image_t *a22,
                  image_t *b1,
                  image_t *b2,
                  image_t *dpsis_horiz,
                  image_t *dpsis_vert,
                  const int iterations,
                  const float omega);

#ifdef __cplusplus
}  // extern C
}  // namespace ccore
#endif


#endif // __SOLVER_H_
