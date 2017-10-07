#ifndef __VARIATIONAL_H_
#define __VARIATIONAL_H_

#include "image.h"

#ifdef __cplusplus
namespace ccore {
extern "C" {
#endif



typedef struct variational_params_s {
  float alpha;             // smoothness weight
  float gamma;             // gradient constancy assumption weight
  float delta;             // color constancy assumption weight
  float sigma;             // presmoothing of the images
  int niter_outer;         // number of outer fixed point iterations
  int niter_inner;         // number of inner fixed point iterations
  int niter_solver;        // number of solver iterations 
  float sor_omega;         // omega parameter of sor method
} variational_params_t;

#define DLL_PUBLIC __attribute__ ((visibility ("default")))


/* set flow parameters to default */
DLL_PUBLIC void variational_params_default(variational_params_t *params);


/* Compute a refinement of the optical flow (wx and wy are modified) between im1 and im2 */
void variational(image_t *wx,
                 image_t *wy,
                 const color_image_t *im1,
                 const color_image_t *im2,
                 const variational_params_t *params);



#ifdef __cplusplus
}  // extern C
}  // namespace ccore
#endif

#endif // __VARIATIONAL_H_
