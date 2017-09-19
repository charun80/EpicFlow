#include "ctypesEpicFlow.h"




extern "C"
DLL_PUBLIC void computeEpicFlow(
    const color_image_t*        const f_inImg1_p,
    const color_image_t*        const f_inImg2_p,
          float_image*          const f_edges_p,
    const float_image*          const f_matches_p,
    const epic_params_t*        const f_epicParams_p,
    const variational_params_t* const f_flowParams_p,
          image_t*              const f_wx_p,
          image_t*              const f_wy_p  )
{
    static const int l_numThreads = 1;


    // compute interpolation and energy minimization
    {
        color_image_t* l_imlab_p = rgb_to_lab( f_inImg1_p );

        epic( f_wx_p, f_wy_p, l_imlab_p, f_matches_p, f_edges_p, f_epicParams_p, l_numThreads );

        color_image_delete( l_imlab_p );
    }

    // energy minimization
    variational( f_wx_p, f_wy_p, f_inImg1_p, f_inImg2_p, f_flowParams_p );
}
