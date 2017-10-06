#ifndef CTYPESEPICFLOW_H_INCLUDED
#define CTYPESEPICFLOW_H_INCLUDED

#include <cstddef>
#include "epic.h"
#include "image.h"
#include "variational.h"
#include "array_types.h"


#define DLL_PUBLIC __attribute__ ((visibility ("default")))


extern "C"
{


DLL_PUBLIC size_t getArrayAlignment();


DLL_PUBLIC void computeEpicFlow(
    const ccore::color_image_t*        const f_inImg1_p,
    const ccore::color_image_t*        const f_inImg2_p,
          ccore::float_image*          const f_edges_p,
    const ccore::float_image*          const f_matches_p,
    const epic::epic_params_t*         const f_epicParams_p,
    const ccore::variational_params_t* const f_flowParams_p,
          ccore::image_t*              const f_wx_p,
          ccore::image_t*              const f_wy_p  );

} // extern C




#endif // CTYPESEPICFLOW_H_INCLUDED
