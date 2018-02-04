#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "variational.h"
#include "variational_aux.h"
#include "solver.h"

#include "simd.h"



#ifdef __cplusplus
namespace ccore
{
#endif



typedef struct variational_ext_params_s {
    const variational_params_t* const p;
    
    const float half_alpha;
    const float half_delta_over3;
    const float half_gamma_over3;
} variational_ext_params_t;



static const convolution_ct* get_deriv()
{
    static const int order = 2;
    static const int even = 0;
    
    static convolution_t derivSingleton = {0,NULL,NULL};
    static float coeffs[5 /*2 * order + 1*/];
    static float coeffs_accu[5 /*2 * order + 1*/];
    
    if (NULL == derivSingleton.coeffs)
    {
        {
            const float deriv_filter[3] = {0.0f, -8.0f/12.0f, 1.0f/12.0f};
            convolve_extract_coeffs(order, deriv_filter, coeffs, coeffs_accu, even);
        }
        
        derivSingleton.order = order;
        derivSingleton.coeffs = coeffs;
        derivSingleton.coeffs_accu = coeffs_accu;
    }
    
    return const_convolution_cast( &derivSingleton );
}



static const convolution_ct* get_deriv_flow()
{
    static const int order = 1;
    static const int even = 0;
    
    static convolution_t derivFlowSingleton = {0,NULL,NULL};
    static float coeffs[3 /*2 * order + 1*/];
    static float coeffs_accu[3 /*2 * order + 1*/];
    
    if (NULL == derivFlowSingleton.coeffs)
    {
        {
            const float deriv_filter_flow[2] = {0.0f, -0.5f};
            convolve_extract_coeffs(order, deriv_filter_flow, coeffs, coeffs_accu, even);
        }
        
        derivFlowSingleton.order = order;
        derivFlowSingleton.coeffs = coeffs;
        derivFlowSingleton.coeffs_accu = coeffs_accu;
    }
    
    return const_convolution_cast( &derivFlowSingleton );
}



/* perform flow computation at one level of the pyramid */
static void compute_one_level( image_t *wx,
                               image_t *wy,
                               const color_image_ct *im1,
                               const color_image_ct *im2,
                               const variational_ext_params_t *params)
{ 
    const int width = wx->width, height = wx->height, stride=wx->stride;

    image_t *du = image_new(width,height), *dv = image_new(width,height), // the flow increment
        *mask = image_new(width,height), // mask containing 0 if a point goes outside image boundary, 1 otherwise
        *smooth_horiz = image_new(width,height), *smooth_vert = image_new(width,height), // horiz: (i,j) contains the diffusivity coeff from (i,j) to (i+1,j) 
        *uu = image_new(width,height), *vv = image_new(width,height), // flow plus flow increment
        *a11 = image_new(width,height), *a12 = image_new(width,height), *a22 = image_new(width,height), // system matrix A of Ax=b for each pixel
        *b1 = image_new(width,height), *b2 = image_new(width,height); // system matrix b of Ax=b for each pixel

    color_image_t *w_im2 = color_image_new(width,height), // warped second image
        *Ix = color_image_new(width,height), *Iy = color_image_new(width,height), *Iz = color_image_new(width,height), // first order derivatives
        *Ixx = color_image_new(width,height), *Ixy = color_image_new(width,height), *Iyy = color_image_new(width,height), *Ixz = color_image_new(width,height), *Iyz = color_image_new(width,height); // second order derivatives
  
  
    const image_ct* const dpsis_weight = const_image_cast( compute_dpsis_weight( im1, 5.0f, get_deriv() ) );  
  
    int i_outer_iteration;
    for(i_outer_iteration = 0 ; i_outer_iteration < params->p->niter_outer ; i_outer_iteration++){
        int i_inner_iteration;
        // warp second image
        image_warp(w_im2, mask, im2, const_image_cast( wx ), const_image_cast( wy ) );
        // compute derivatives
        get_derivatives(im1, const_color_image_cast( w_im2 ), get_deriv(), Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz);
        // erase du and dv
        image_erase(du);
        image_erase(dv);
        // initialize uu and vv
        memcpy(uu->data,wx->data,wx->stride*wx->height*sizeof(float));
        memcpy(vv->data,wy->data,wy->stride*wy->height*sizeof(float));
        // inner fixed point iterations
        for(i_inner_iteration = 0 ; i_inner_iteration < params->p->niter_inner ; i_inner_iteration++){
            //  compute robust function and system
            compute_smoothness(smooth_horiz, smooth_vert, 
                               const_image_cast( uu ), const_image_cast( vv ), 
                               dpsis_weight, get_deriv_flow(), params->half_alpha );
            compute_data_and_match(a11, a12, a22, b1, b2, mask, du, dv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, params->half_delta_over3, params->half_gamma_over3);
            sub_laplacian(b1, const_image_cast( wx ), const_image_cast( smooth_horiz ), const_image_cast( smooth_vert) );
            sub_laplacian(b2, const_image_cast( wy ), const_image_cast( smooth_horiz ), const_image_cast( smooth_vert) );
            // solve system
            sor_coupled(du, dv, a11, a12, a22, b1, b2, smooth_horiz, smooth_vert, params->p->niter_solver, params->p->sor_omega);
            
            // update flow plus flow increment
            simdsf_t *uup = simdsf_ptrcast( uu->data ),
                     *vvp = simdsf_ptrcast( vv->data ),
                     *wxp = simdsf_ptrcast( wx->data ),
                     *wyp = simdsf_ptrcast( wy->data ),
                     *dup = simdsf_ptrcast( du->data ),
                     *dvp = simdsf_ptrcast( dv->data );
            for( int i=0 ; i < (height * stride / NSimdFloats); i++)
            {
                (*uup) = (*wxp) + (*dup);
                (*vvp) = (*wyp) + (*dvp);
                uup+=1; vvp+=1; wxp+=1; wyp+=1;dup+=1;dvp+=1;
	        }
        }
        // add flow increment to current flow
        memcpy(wx->data,uu->data,uu->stride*uu->height*sizeof(float));
        memcpy(wy->data,vv->data,vv->stride*vv->height*sizeof(float));
    }   
    // free memory
    image_delete(du); image_delete(dv);
    image_delete(mask);
    image_delete(smooth_horiz); image_delete(smooth_vert);
    image_delete(uu); image_delete(vv);
    image_delete(a11); image_delete(a12); image_delete(a22);
    image_delete(b1); image_delete(b2);
    image_delete( image_cast( dpsis_weight ) );
    color_image_delete(w_im2); 
    color_image_delete(Ix); color_image_delete(Iy); color_image_delete(Iz);
    color_image_delete(Ixx); color_image_delete(Ixy); color_image_delete(Iyy); color_image_delete(Ixz); color_image_delete(Iyz);
}


/* set flow parameters to default */
void variational_params_default(variational_params_t *params){
    if(!params){
        fprintf(stderr,"Error optical_flow_params_default: argument is null\n");
        exit(1);
    }
    params->alpha = 1.0f;
    params->gamma = 0.71f;
    params->delta = 0.0f;
    params->sigma = 1.00f;
    params->niter_outer = 5;
    params->niter_inner = 1;  
    params->niter_solver = 30;
    params->sor_omega = 1.9f;
}
 
 
/* Compute a refinement of the optical flow (wx and wy are modified) between im1 and im2 */
void variational(image_t *wx,
                 image_t *wy,
                 const color_image_ct *im1,
                 const color_image_ct *im2,
                 const variational_params_t *params)
{

    // initialize variational variables
    const variational_ext_params_t extParams = 
    {
        params,
        0.5f*params->alpha, //half_alpha
        params->gamma*0.5f/3.0f, // half_gamma_over3
        params->delta*0.5f/3.0f  // half_delta_over3
    };  


    // presmooth images
    int width = im1->width, height = im1->height, filter_size;
    color_image_t *smooth_im1 = color_image_new(width, height), 
                  *smooth_im2 = color_image_new(width, height);
    float *presmooth_filter = gaussian_filter(params->sigma, &filter_size);
    const convolution_ct *presmoothing = const_convolution_cast( convolution_new(filter_size, presmooth_filter, 1) );
    color_image_convolve_hv(smooth_im1, im1, presmoothing, presmoothing);
    color_image_convolve_hv(smooth_im2, im2, presmoothing, presmoothing); 
    convolution_delete( convolution_cast( presmoothing ) );
    free(presmooth_filter);
    
    compute_one_level(wx, wy, const_color_image_cast( smooth_im1), const_color_image_cast( smooth_im2 ), &extParams);
  
    // free memory
    color_image_delete(smooth_im1);
    color_image_delete(smooth_im2);
}



#ifdef __cplusplus
}  // namespace ccore
#endif

