#ifndef CTYPESEPICFLOW_H_INCLUDED
#define CTYPESEPICFLOW_H_INCLUDED

#include "epic.h"
#include "image.h"
#include "variational.h"
#include "array_types.h"


#define DLL_PUBLIC __attribute__ ((visibility ("default")))


extern "C"
{

DLL_PUBLIC void computeEpicFlow(
    const color_image_t*        const f_inImg1_p,
    const color_image_t*        const f_inImg2_p,
          float_image*          const f_edges_p,
    const float_image*          const f_matches_p,
    const epic_params_t*        const f_epicParams_p,
    const variational_params_t* const f_flowParams_p,
          image_t*              const f_wx_p,
          image_t*              const f_wy_p  );

} // extern C




#endif // CTYPESEPICFLOW_H_INCLUDED
