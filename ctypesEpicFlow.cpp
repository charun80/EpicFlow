#include "ctypesEpicFlow.h"
#include "simd.h"


extern "C"
DLL_PUBLIC size_t getArrayAlignment()
{
    return NSimdBytes;
}


extern "C"
DLL_PUBLIC void computeEpicFlow(
    const ccore::color_image_t*        const f_inImg1_p,
    const ccore::color_image_t*        const f_inImg2_p,
          ccore::float_image*          const f_edges_p,
    const ccore::float_image*          const f_matches_p,
    const epic::epic_params_t*         const f_epicParams_p,
    const ccore::variational_params_t* const f_flowParams_p,
          ccore::image_t*              const f_wx_p,
          ccore::image_t*              const f_wy_p  )
{
    static const int l_numThreads = 4;


    // compute interpolation and energy minimization
    {
        ccore::color_image_t* l_imlab_p = rgb_to_lab( f_inImg1_p );

        epic::epic( f_wx_p, f_wy_p, l_imlab_p, f_matches_p, f_edges_p, f_epicParams_p, l_numThreads );

        color_image_delete( l_imlab_p );
    }

    // energy minimization
    variational( f_wx_p, f_wy_p, f_inImg1_p, f_inImg2_p, f_flowParams_p );
}
