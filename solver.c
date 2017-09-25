#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "image.h"
#include "solver.h"

#include "simd.h"




#ifdef __cplusplus
namespace ccore
{
#endif




//THIS IS A SLOW VERSION BUT READABLE
//Perform n iterations of the sor_coupled algorithm
//du and dv are used as initial guesses
//The system form is the same as in opticalflow.c
void sor_coupled_slow_but_readable(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega){
    int i,j,iter;
    float sigma_u,sigma_v,sum_dpsis,A11,A22,A12,B1,B2,det;
    for(iter = 0 ; iter<iterations ; iter++){
        for(j=0 ; j<du->height ; j++){
	        for(i=0 ; i<du->width ; i++){
	            sigma_u = 0.0f;
	            sigma_v = 0.0f;
	            sum_dpsis = 0.0f;
	            if(j>0){
		            sigma_u -= dpsis_vert->data[(j-1)*du->stride+i]*du->data[(j-1)*du->stride+i];
		            sigma_v -= dpsis_vert->data[(j-1)*du->stride+i]*dv->data[(j-1)*du->stride+i];
		            sum_dpsis += dpsis_vert->data[(j-1)*du->stride+i];
		        }
	            if(i>0){
                    sigma_u -= dpsis_horiz->data[j*du->stride+i-1]*du->data[j*du->stride+i-1];
                    sigma_v -= dpsis_horiz->data[j*du->stride+i-1]*dv->data[j*du->stride+i-1];
                    sum_dpsis += dpsis_horiz->data[j*du->stride+i-1];
		        }
	            if(j<du->height-1){
		            sigma_u -= dpsis_vert->data[j*du->stride+i]*du->data[(j+1)*du->stride+i];
		            sigma_v -= dpsis_vert->data[j*du->stride+i]*dv->data[(j+1)*du->stride+i];
		            sum_dpsis += dpsis_vert->data[j*du->stride+i];
		        }
	            if(i<du->width-1){
		            sigma_u -= dpsis_horiz->data[j*du->stride+i]*du->data[j*du->stride+i+1];
		            sigma_v -= dpsis_horiz->data[j*du->stride+i]*dv->data[j*du->stride+i+1];
		            sum_dpsis += dpsis_horiz->data[j*du->stride+i];
		        }
                A11 = a11->data[j*du->stride+i]+sum_dpsis;
                A12 = a12->data[j*du->stride+i];
                A22 = a22->data[j*du->stride+i]+sum_dpsis;
                det = A11*A22-A12*A12;
                B1 = b1->data[j*du->stride+i]-sigma_u;
                B2 = b2->data[j*du->stride+i]-sigma_v;
                du->data[j*du->stride+i] = (1.0f-omega)*du->data[j*du->stride+i] +omega*( A22*B1-A12*B2)/det;
                dv->data[j*du->stride+i] = (1.0f-omega)*dv->data[j*du->stride+i] +omega*(-A12*B1+A11*B2)/det;
	        }
	    }
    }
}


 // THIS IS A FASTER VERSION BUT UNREADABLE
 // the first iteration is separated from the other to compute the inverse of the 2x2 block diagonal
 // each iteration is split in two first line / middle lines / last line, and the left block is computed separately on each line
void sor_coupled(image_t *du, image_t *dv, image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *dpsis_horiz, image_t *dpsis_vert, const int iterations, const float omega){
    //sor_coupled_slow_but_readable(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations,omega); return; printf("test\n");
  
    if(du->width<2 || du->height<2 || iterations < 1){
        sor_coupled_slow_but_readable(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations,omega);
        return;
    }
    
    const int stride = du->stride;
    const int width = du->width;
    const int iterheight = du->height-1;
    const int iterline = (stride) / NSIMDBYTES;
    const int width_minus_1_sizeoffloat = sizeof(float)*(width-1);
    
    float *floatarray = NULL;
    
    const int lMemAlignError = posix_memalign( (void**)(&floatarray), NSimdBytes, 3 * stride * sizeof(float) );
    if ( 0 != lMemAlignError )
    {
        fprintf( stderr, "Error: allocating memory in sor_coupled: %d !\n", lMemAlignError );
        exit(1);
    }
    
    float *f1 = floatarray;
    float *f2 = f1+stride;
    float *f3 = f2+stride;
    f1[0] = 0.0f;
    memset(&f1[width], 0, sizeof(float)*(stride-width));
    memset(&f2[width-1], 0, sizeof(float)*(stride-width+1));
    memset(&f3[width-1], 0, sizeof(float)*(stride-width+1));   	  

    { 
        // first iteration
        simdsf_t *a11p = simdsf_ptrcast( a11->data ),
                 *a12p = simdsf_ptrcast( a12->data ),
                 *a22p = simdsf_ptrcast( a22->data ),
                  *b1p = simdsf_ptrcast( b1->data ),
                  *b2p = simdsf_ptrcast( b2->data ),
                  *hp  = simdsf_ptrcast( dpsis_horiz->data ),
                   *vp = simdsf_ptrcast( dpsis_vert->data );
              
        float *du_ptr = du->data, 
              *dv_ptr = dv->data;
        simdsf_t *dub = (simdsf_t*) (du_ptr+stride), 
                 *dvb = (simdsf_t*) (dv_ptr+stride);
        
        { // first iteration - first line
        
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            
            simdsf_t *hpl = simdsf_ptrcast( f1 ), 
                     *dur = simdsf_ptrcast( f2 ),
                     *dvr = simdsf_ptrcast( f3 );
            
            { // left block
                // reverse 2x2 diagonal block
                const simdsf_t dpsis = (*hpl) + (*hp) + (*vp);
                const simdsf_t A11 = (*a22p)+dpsis, 
                               A22 = (*a11p)+dpsis;
                const simdsf_t det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vp)*(*dub) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vp)*(*dvb) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                
                for( int k=1; k < NSimdFloats; k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats; dv_ptr += NSimdFloats;        
            }
            for( int i=iterline; --i; ){
                // reverse 2x2 diagonal block
                const simdsf_t dpsis = (*hpl) + (*hp) + (*vp);
                const simdsf_t A11 = (*a22p)+dpsis,
                               A22 = (*a11p)+dpsis;
                const simdsf_t det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vp)*(*dub) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vp)*(*dvb) + (*b2p);
                for( int k=0; k < NSimdFloats; k++)
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats;
                dv_ptr += NSimdFloats;
            }
          
        }
        
        simdsf_t *vpt = simdsf_ptrcast( dpsis_vert->data );
        simdsf_t *dut = simdsf_ptrcast( du->data ), 
                 *dvt = simdsf_ptrcast( dv->data );
        
        for( int j=iterheight; --j; ){ // first iteration - middle lines
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            
            simdsf_t *hpl = simdsf_ptrcast( f1 ),
                     *dur = simdsf_ptrcast( f2 ),
                     *dvr = simdsf_ptrcast( f3 );
                 
            { // left block
                // reverse 2x2 diagonal block
                const simdsf_t dpsis = (*hpl) + (*hp) + (*vpt) + (*vp);
                const simdsf_t A11 = (*a22p)+dpsis, 
                               A22 = (*a11p)+dpsis;
                const simdsf_t det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*vp)*(*dub) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*vp)*(*dvb) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                for( int k=1; k < NSimdFloats; k++)
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats;
                dv_ptr += NSimdFloats;           
            }
            
            for( int i=iterline; --i; ) {
                // reverse 2x2 diagonal block
                const simdsf_t dpsis = (*hpl) + (*hp) + (*vpt) + (*vp);
                const simdsf_t   A11 = (*a22p)+dpsis, 
                                 A22 = (*a11p)+dpsis;
                const simdsf_t   det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*vp)*(*dub) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*vp)*(*dvb) + (*b2p);
                for( int k=0; k < NSimdFloats; k++)
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats;
                dv_ptr += NSimdFloats;
            }
        }
        
        { // first iteration - last line
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            simdsf_t *hpl = simdsf_ptrcast( f1 ),
                     *dur = simdsf_ptrcast( f2 ),
                     *dvr = simdsf_ptrcast( f3 );

            { // left block
                // reverse 2x2 diagonal block
                const simdsf_t dpsis = (*hpl) + (*hp) + (*vpt);
                const simdsf_t A11 = (*a22p)+dpsis,
                               A22 = (*a11p)+dpsis;
                const simdsf_t det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                for( int k=1; k < NSimdFloats; k++)
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats;
                dv_ptr += NSimdFloats;           
            }
            
            for( int i=iterline; --i; )
            {
                // reverse 2x2 diagonal block
                const simdsf_t dpsis = (*hpl) + (*hp) + (*vpt);
                const simdsf_t A11 = (*a22p)+dpsis,
                               A22 = (*a11p)+dpsis;
                const simdsf_t det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*b2p);
                for( int k=0; k < NSimdFloats; k++)
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats; dv_ptr += NSimdFloats;
            }

        }
    }

    for( int iter=iterations; --iter;)
    { 
        // other iterations
        simdsf_t *a11p = simdsf_ptrcast( a11->data ),
                 *a12p = simdsf_ptrcast( a12->data ),
                 *a22p = simdsf_ptrcast( a22->data ),
                 *b1p  = simdsf_ptrcast( b1->data ),
                 *b2p  = simdsf_ptrcast( b2->data ),
                 *hp   = simdsf_ptrcast( dpsis_horiz->data ),
                 *vp   = simdsf_ptrcast( dpsis_vert->data );
        
        float *du_ptr = du->data,
              *dv_ptr = dv->data;
        simdsf_t *dub = simdsf_ptrcast( du_ptr + stride ),
                 *dvb = simdsf_ptrcast( dv_ptr + stride );
        
        { // other iteration - first line
        
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            simdsf_t *hpl = simdsf_ptrcast( f1 ),
                     *dur = simdsf_ptrcast( f2 ), 
                     *dvr = simdsf_ptrcast( f3 );
            
            { // left block
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vp)*(*dub) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vp)*(*dvb) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                
                for( int k=1; k < NSimdFloats; k++)
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats;
                dv_ptr += NSimdFloats;        
            }
            for( int i=iterline; --i; )
            {
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vp)*(*dub) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vp)*(*dvb) + (*b2p);
                for( int k=0; k < NSimdFloats; k++)
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats;
                dv_ptr += NSimdFloats;
            }
          
        }
        
        simdsf_t *vpt = simdsf_ptrcast( dpsis_vert->data );
        simdsf_t *dut = simdsf_ptrcast( du->data ), 
                 *dvt = simdsf_ptrcast( dv->data );
        
        for( int j=iterheight; --j; )
        { 
            // other iteration - middle lines
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            simdsf_t  *hpl = simdsf_ptrcast( f1 ),
                      *dur = simdsf_ptrcast( f2 ),
                      *dvr = simdsf_ptrcast( f3 );
                 
            { // left block
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*vp)*(*dub) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*vp)*(*dvb) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                
                for( int k=1; k < NSimdFloats; k++ )
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats;
                dv_ptr += NSimdFloats;           
            }
            
            for( int i=iterline; --i; )
            {
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*vp)*(*dub) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*vp)*(*dvb) + (*b2p);
                
                for( int k=0; k < NSimdFloats; k++ )
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats;
                dv_ptr += NSimdFloats;
            }
                
        }
        
        { 
            // other iteration - last line
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            simdsf_t *hpl = simdsf_ptrcast( f1 ),
                     *dur = simdsf_ptrcast( f2 ),
                     *dvr = simdsf_ptrcast( f3 );

            { // left block
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] ); 
            
                for ( int k=1; k < NSimdFloats; k++)
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats;
                dv_ptr += NSimdFloats;           
            }
            
            for( int i=iterline;--i; )
            {
                // do one iteration
                const simdsf_t s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*b1p);
                const simdsf_t s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*b2p);
                
                for( int k=0; k < NSimdFloats; k++ )
                {
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; b1p+=1; b2p+=1;
                du_ptr += NSimdFloats;
                dv_ptr += NSimdFloats;
            }

        }
    }

    free(floatarray);
}



#ifdef __cplusplus
}  // namespace ccore
#endif

