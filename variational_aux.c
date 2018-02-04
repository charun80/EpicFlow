#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "variational_aux.h"

#include "simd.h"



#ifdef __cplusplus
namespace ccore
{
#endif


#define datanorm 0.1f*0.1f//0.01f // square of the normalization factor
#define epsilon_color (0.001f*0.001f)//0.000001f
#define epsilon_grad (0.001f*0.001f)//0.000001f
#define epsilon_smooth (0.001f*0.001f)//0.000001f

#define RECTIFY(a,b) (((a)<0) ? (0) : ( ((a)<(b)-1) ? (a) : ((b)-1) ) )

/* warp a color image according to a flow. src is the input image, wx and wy, the input flow. dst is the warped image and mask contains 0 or 1 if the pixels goes outside/inside image boundaries */
void image_warp(color_image_t *dst, image_t *mask, const color_image_ct *src, const image_ct *wx, const image_ct *wy) {
    int i, j, offset, x, y, x1, x2, y1, y2;
    float xx, yy, dx, dy;
    for(j=0 ; j<src->height ; j++){
        offset = j*src->stride;
        for(i=0 ; i<src->width ; i++,offset++){
	        xx = i+wx->data[offset];
	        yy = j+wy->data[offset];
	        x = floor(xx);
	        y = floor(yy);
	        dx = xx-x;
	        dy = yy-y;
	        mask->data[offset] = (xx>=0 && xx<=src->width-1 && yy>=0 && yy<=src->height-1);
	        x1 = RECTIFY(x, src->width);
	        x2 = RECTIFY(x+1, src->width);
	        y1 = RECTIFY(y, src->height);
	        y2 = RECTIFY(y+1, src->height);
	        dst->c1[offset] = 
	            src->c1[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
	            src->c1[y1*src->stride+x2]*dx*(1.0f-dy) +
	            src->c1[y2*src->stride+x1]*(1.0f-dx)*dy +
	            src->c1[y2*src->stride+x2]*dx*dy;
	        dst->c2[offset] = 
	            src->c2[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
	            src->c2[y1*src->stride+x2]*dx*(1.0f-dy) +
	            src->c2[y2*src->stride+x1]*(1.0f-dx)*dy +
	            src->c2[y2*src->stride+x2]*dx*dy;
	        dst->c3[offset] = 
	            src->c3[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
	            src->c3[y1*src->stride+x2]*dx*(1.0f-dy) +
	            src->c3[y2*src->stride+x1]*(1.0f-dx)*dy +
	            src->c3[y2*src->stride+x2]*dx*dy;
	    }
    }
}

/* compute image first and second order spatio-temporal derivatives of a color image */
void get_derivatives(const color_image_ct *im1, const color_image_ct *im2, const convolution_ct *deriv,
		     color_image_t *dx, color_image_t *dy, color_image_t *dt, 
		     color_image_t *dxx, color_image_t *dxy, color_image_t *dyy, color_image_t *dxt, color_image_t *dyt) {
    // derivatives are computed on the mean of the first image and the warped second image
    color_image_t *tmp_im2 = color_image_new(im2->width,im2->height);
    
    simdsf_t *tmp_im2p = simdsf_ptrcast( tmp_im2->c1 ),
                  *dtp = simdsf_ptrcast( dt->c1 );
                 
    const simdsf_t *im1p = simdsf_const_ptrcast( im1->c1 ),
                   *im2p = simdsf_const_ptrcast( im2->c1 );
    
    const simdsf_t half = simdsf_init( 0.5f );
    
    
    for( int i=0 ; i < ( 3 * im1->height * im1->stride / NSimdFloats ) ; i++)
    {
        *tmp_im2p = half * ( (*im2p) + (*im1p) );
        *dtp = (*im2p)-(*im1p);
        dtp+=1; im1p+=1; im2p+=1; tmp_im2p+=1;
    } 
    
    // compute all other derivatives
    color_image_convolve_hv(dx, const_color_image_cast( tmp_im2 ), deriv, NULL);
    color_image_convolve_hv(dy, const_color_image_cast( tmp_im2 ), NULL, deriv);
    color_image_convolve_hv(dxx, const_color_image_cast( dx ), deriv, NULL);
    color_image_convolve_hv(dxy, const_color_image_cast( dx ), NULL, deriv);
    color_image_convolve_hv(dyy, const_color_image_cast( dy ), NULL, deriv);
    color_image_convolve_hv(dxt, const_color_image_cast( dt ), deriv, NULL);
    color_image_convolve_hv(dyt, const_color_image_cast( dt ), NULL, deriv);
    // free memory
    color_image_delete(tmp_im2);
}

/* compute the smoothness term */
/* It is represented as two images, the first one for horizontal smoothness, the second for vertical
   in dst_horiz, the pixel i,j represents the smoothness weight between pixel i,j and i,j+1
   in dst_vert, the pixel i,j represents the smoothness weight between pixel i,j and i+1,j */
void compute_smoothness(image_t *dst_horiz, image_t *dst_vert, const image_ct *uu, const image_ct *vv, const image_ct *dpsis_weight, const convolution_ct *deriv_flow, const float half_alpha) {
  int w = uu->width, h = uu->height, s = uu->stride, i, j, offset;
  image_t *ux1 = image_new(w,h), *uy1 = image_new(w,h), *vx1 = image_new(w,h), *vy1 = image_new(w,h), 
    *ux2 = image_new(w,h), *uy2 = image_new(w,h), *vx2 = image_new(w,h), *vy2 = image_new(w,h);  
  // compute ux1, vx1, filter [-1 1]
  for( j=0 ; j<h ; j++)
    {
      offset = j*s;
      for( i=0 ; i<w-1 ; i++, offset++)
	{
	  ux1->data[offset] = uu->data[offset+1] - uu->data[offset];
	  vx1->data[offset] = vv->data[offset+1] - vv->data[offset];
	}
    }
  // compute uy1, vy1, filter [-1;1]
  for( j=0 ; j<h-1 ; j++)
    {
      offset = j*s;
      for( i=0 ; i<w ; i++, offset++)
	{
	  uy1->data[offset] = uu->data[offset+s] - uu->data[offset];
	  vy1->data[offset] = vv->data[offset+s] - vv->data[offset];
	}
    }
  // compute ux2, uy2, vx2, vy2, filter [-0.5 0 0.5]
  convolve_horiz(ux2,uu,deriv_flow);
  convolve_horiz(vx2,vv,deriv_flow);
  convolve_vert(uy2,uu,deriv_flow);
  convolve_vert(vy2,vv,deriv_flow);
  // compute final value, horiz
  for( j=0 ; j<h ; j++)
    {
      offset = j*s;
      for( i=0 ; i<w-1 ; i++, offset++)
	{
	  float tmp = 0.5f*(uy2->data[offset]+uy2->data[offset+1]);
	  float uxsq = ux1->data[offset]*ux1->data[offset] + tmp*tmp;
	  tmp = 0.5f*(vy2->data[offset]+vy2->data[offset+1]);
	  float vxsq = vx1->data[offset]*vx1->data[offset] + tmp*tmp;
	  tmp = uxsq + vxsq;
	  dst_horiz->data[offset] = (dpsis_weight->data[offset]+dpsis_weight->data[offset+1])*half_alpha / sqrt( tmp + epsilon_smooth ) ;
	}
	memset( &dst_horiz->data[j*s+w-1], 0, sizeof(float)*(s-w+1));
    }
  // compute final value, vert
  for( j=0 ; j<h-1 ; j++)
    {
      offset = j*s;
      for( i=0 ; i<w ; i++, offset++)
	{
	  float tmp = 0.5f*(ux2->data[offset]+ux2->data[offset+s]);
	  float uysq = uy1->data[offset]*uy1->data[offset] + tmp*tmp;
	  tmp = 0.5f*(vx2->data[offset]+vx2->data[offset+s]);
	  float vysq = vy1->data[offset]*vy1->data[offset] + tmp*tmp;
	  tmp = uysq + vysq;
	  dst_vert->data[offset] = (dpsis_weight->data[offset]+dpsis_weight->data[offset+s])*half_alpha / sqrt( tmp + epsilon_smooth ) ;
	  /*if( dpsis_weight->data[offset]<dpsis_weight->data[offset+s])
	    dst_vert->data[offset] = dpsis_weight->data[offset]*half_alpha / sqrt( tmp + epsilon_smooth ) ;
	  else
	  dst_vert->data[offset] = dpsis_weight->data[offset+s]*half_alpha / sqrt( tmp + epsilon_smooth ) ;*/
	}
    }
  memset( &dst_vert->data[(h-1)*s], 0, sizeof(float)*s);
  image_delete(ux1); image_delete(uy1); image_delete(vx1); image_delete(vy1);
  image_delete(ux2); image_delete(uy2); image_delete(vx2); image_delete(vy2);
}


/* sub the laplacian (smoothness term) to the right-hand term */
void sub_laplacian(image_t *dst, const image_ct *src, const image_ct *weight_horiz, const image_ct *weight_vert){
    
    const int offsetline = src->stride-src->width;
    const float *src_ptr = src->data,
                *weight_horiz_ptr = weight_horiz->data;
          float *dst_ptr = dst->data;
    // horizontal filtering
    for( int j=src->height+1; --j; ) { // faster than for(j=0;j<src->height;j++)
        int i;
        for(i=src->width;--i;){
	        const float tmp = (*weight_horiz_ptr)*((*(src_ptr+1))-(*src_ptr));
	        *dst_ptr += tmp;
	        *(dst_ptr+1) -= tmp;
	        dst_ptr++;
	        src_ptr++;
	        weight_horiz_ptr++;
	    }
        dst_ptr += offsetline+1;
        src_ptr += offsetline+1;
        weight_horiz_ptr += offsetline+1;
    }
  
    const simdsf_t   *wvp = simdsf_const_ptrcast( weight_vert->data ),
                    *srcp = simdsf_const_ptrcast( src->data ),
                  *srcp_s = simdsf_const_ptrcast( src->data + src->stride );
            
    simdsf_t  *dstp  = simdsf_ptrcast( dst->data ), 
             *dstp_s = simdsf_ptrcast( dst->data + src->stride );
    
    for( int j = 1 + ((src->height-1)*src->stride / NSimdFloats ); --j ;)
    {
        const simdsf_t tmp = (*wvp) * ((*srcp_s)-(*srcp));
        *dstp += tmp;
        *dstp_s -= tmp;
        wvp+=1; srcp+=1; srcp_s+=1; dstp+=1; dstp_s+=1;
    }
}

/* compute local smoothness weight as a sigmoid on image gradient*/
image_t* compute_dpsis_weight( const color_image_ct *im, float coef, const convolution_ct *deriv) {
    image_t *lum = image_new(im->width, im->height), 
          *lum_x = image_new(im->width, im->height),
          *lum_y = image_new(im->width, im->height);
    
    // ocompute luminance
    const simdsf_t *im1p = simdsf_const_ptrcast( im->c1 ),
                   *im2p = simdsf_const_ptrcast( im->c2 ),
                   *im3p = simdsf_const_ptrcast( im->c3 );
    
    simdsf_t *lump = simdsf_ptrcast( lum->data );
    
    for( int i=0 ; i < (im->height * im->stride / NSimdFloats ); i++)
    {
        *lump = (0.299f*(*im1p) + 0.587f*(*im2p) + 0.114f*(*im3p))/255.0f;
        lump+=1; im1p+=1; im2p+=1; im3p+=1;
    }
    // compute derivatives with five-point tencil
    convolve_horiz(lum_x, const_image_cast( lum ), deriv);
    convolve_vert(lum_y, const_image_cast( lum ), deriv);
    
    // compute lum norm
    lump = simdsf_ptrcast( lum->data );
    
    const simdsf_t *lumxp = simdsf_ptrcast( lum_x->data ),
                   *lumyp = simdsf_ptrcast( lum_y->data );
    
    for( int i = 0 ; i < (lum->height * lum->stride / NSimdFloats ); i++)
    {
        *lump = -coef * simdsf_sqrt( (*lumxp)*(*lumxp) + (*lumyp)*(*lumyp) );
        // TODO: Implement expf using SSE / AVX
        for ( int k = 0; k < NSimdFloats; ++k )
            lump[0][k] = 0.5f * expf( lump[0][k] );

        lump+=1; lumxp+=1; lumyp+=1;
    }
    image_delete(lum_x);
    image_delete(lum_y);
    return lum;
}


/* compute the dataterm and the matching term
   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
   other (color) images are input */
void compute_data_and_match(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *du, image_t *dv, color_image_t *Ix, color_image_t *Iy, color_image_t *Iz, color_image_t *Ixx, color_image_t *Ixy, color_image_t *Iyy, color_image_t *Ixz, color_image_t *Iyz, const float half_delta_over3, const float half_gamma_over3){
 
    const simdsf_t dnorm    = simdsf_init( datanorm );
    const simdsf_t hdover3  = simdsf_init( half_delta_over3 );
    const simdsf_t epscolor = simdsf_init( epsilon_color );
    const simdsf_t hgover3  = simdsf_init( half_gamma_over3 );
    const simdsf_t epsgrad  = simdsf_init( epsilon_grad );

    
    const simdsf_t  *dup = simdsf_ptrcast( du->data ),
                    *dvp = simdsf_ptrcast( dv->data ),
                  *maskp = simdsf_ptrcast( mask->data );
            
    simdsf_t *a11p = simdsf_ptrcast( a11->data ),
             *a12p = simdsf_ptrcast( a12->data ),
             *a22p = simdsf_ptrcast( a22->data ),
        
              *b1p = simdsf_ptrcast( b1->data ),
              *b2p = simdsf_ptrcast( b2->data );
         
    const simdsf_t *ix1p = simdsf_ptrcast( Ix->c1 ),
                   *iy1p = simdsf_ptrcast( Iy->c1 ),
                   *iz1p = simdsf_ptrcast( Iz->c1 ),
             
                  *ixx1p = simdsf_ptrcast( Ixx->c1 ),
                  *ixy1p = simdsf_ptrcast( Ixy->c1 ),
                  *iyy1p = simdsf_ptrcast( Iyy->c1 ),
                  *ixz1p = simdsf_ptrcast( Ixz->c1 ),
                  *iyz1p = simdsf_ptrcast( Iyz->c1 ),
 
                   *ix2p = simdsf_ptrcast( Ix->c2 ),
                   *iy2p = simdsf_ptrcast( Iy->c2 ),
                   *iz2p = simdsf_ptrcast( Iz->c2 ),
            
                  *ixx2p = simdsf_ptrcast( Ixx->c2 ),
                  *ixy2p = simdsf_ptrcast( Ixy->c2 ),
                  *iyy2p = simdsf_ptrcast( Iyy->c2 ),
                  *ixz2p = simdsf_ptrcast( Ixz->c2 ),
                  *iyz2p = simdsf_ptrcast( Iyz->c2 ),
        
                   *ix3p = simdsf_ptrcast( Ix->c3 ),
                   *iy3p = simdsf_ptrcast( Iy->c3 ),
                   *iz3p = simdsf_ptrcast( Iz->c3 ),
              
                  *ixx3p = simdsf_ptrcast( Ixx->c3 ),
                  *ixy3p = simdsf_ptrcast( Ixy->c3 ),
                  *iyy3p = simdsf_ptrcast( Iyy->c3 ),
                  *ixz3p = simdsf_ptrcast( Ixz->c3 ),
                  *iyz3p = simdsf_ptrcast( Iyz->c3 );
        
    memset(a11->data, 0, sizeof(float)*du->height*du->stride);
    memset(a12->data, 0, sizeof(float)*du->height*du->stride);
    memset(a22->data, 0, sizeof(float)*du->height*du->stride);
    memset(b1->data , 0, sizeof(float)*du->height*du->stride);
    memset(b2->data , 0, sizeof(float)*du->height*du->stride);
              
    for( int i = 0 ; i < (du->height * du->stride / NSimdFloats); i++){
        simdsf_t tmp, tmp2, tmp3, tmp4, tmp5, tmp6, n1, n2, n3, n4, n5, n6;
        // dpsi color
        if(half_delta_over3)
        {
            tmp  = *iz1p + (*ix1p)*(*dup) + (*iy1p)*(*dvp);
            n1 = (*ix1p) * (*ix1p) + (*iy1p) * (*iy1p) + dnorm;
            tmp2 = *iz2p + (*ix2p)*(*dup) + (*iy2p)*(*dvp);
            n2 = (*ix2p) * (*ix2p) + (*iy2p) * (*iy2p) + dnorm;
            tmp3 = *iz3p + (*ix3p)*(*dup) + (*iy3p)*(*dvp);
            n3 = (*ix3p) * (*ix3p) + (*iy3p) * (*iy3p) + dnorm;
            tmp = (*maskp) * hdover3 / simdsf_sqrt( tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + epscolor );
            tmp3 = tmp/n3; tmp2 = tmp/n2; tmp /= n1;
            *a11p += tmp  * (*ix1p) * (*ix1p);
            *a12p += tmp  * (*ix1p) * (*iy1p);
            *a22p += tmp  * (*iy1p) * (*iy1p);
            *b1p -=  tmp  * (*iz1p) * (*ix1p);
            *b2p -=  tmp  * (*iz1p) * (*iy1p);
            *a11p += tmp2 * (*ix2p) * (*ix2p);
            *a12p += tmp2 * (*ix2p) * (*iy2p);
            *a22p += tmp2 * (*iy2p) * (*iy2p);
            *b1p -=  tmp2 * (*iz2p) * (*ix2p);
            *b2p -=  tmp2 * (*iz2p) * (*iy2p);
            *a11p += tmp3 * (*ix3p) * (*ix3p);
            *a12p += tmp3 * (*ix3p) * (*iy3p);
            *a22p += tmp3 * (*iy3p) * (*iy3p);
            *b1p -=  tmp3 * (*iz3p) * (*ix3p);
            *b2p -=  tmp3 * (*iz3p) * (*iy3p);
        }
        // dpsi gradient
        n1 = (*ixx1p) * (*ixx1p) + (*ixy1p) * (*ixy1p) + dnorm;
        n2 = (*iyy1p) * (*iyy1p) + (*ixy1p) * (*ixy1p) + dnorm;
        tmp  = *ixz1p + (*ixx1p) * (*dup) + (*ixy1p) * (*dvp);
        tmp2 = *iyz1p + (*ixy1p) * (*dup) + (*iyy1p) * (*dvp);
        n3 = (*ixx2p) * (*ixx2p) + (*ixy2p) * (*ixy2p) + dnorm;
        n4 = (*iyy2p) * (*iyy2p) + (*ixy2p) * (*ixy2p) + dnorm;
        tmp3 = *ixz2p + (*ixx2p) * (*dup) + (*ixy2p) * (*dvp);
        tmp4 = *iyz2p + (*ixy2p) * (*dup) + (*iyy2p) * (*dvp);
        n5 = (*ixx3p) * (*ixx3p) + (*ixy3p) * (*ixy3p) + dnorm;
        n6 = (*iyy3p) * (*iyy3p) + (*ixy3p) * (*ixy3p) + dnorm;
        tmp5 = *ixz3p + (*ixx3p) * (*dup) + (*ixy3p) * (*dvp);
        tmp6 = *iyz3p + (*ixy3p) * (*dup) + (*iyy3p) * (*dvp);
        tmp = (*maskp) * hgover3 / simdsf_sqrt( tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + tmp4*tmp4/n4 + tmp5*tmp5/n5 + tmp6*tmp6/n6 + epsgrad );
        tmp6 = tmp/n6; tmp5 = tmp/n5; tmp4 = tmp/n4; tmp3 = tmp/n3; tmp2 = tmp/n2; tmp /= n1;
        *a11p += tmp *(*ixx1p)*(*ixx1p) + tmp2*(*ixy1p)*(*ixy1p);
        *a12p += tmp *(*ixx1p)*(*ixy1p) + tmp2*(*ixy1p)*(*iyy1p);
        *a22p += tmp2*(*iyy1p)*(*iyy1p) + tmp *(*ixy1p)*(*ixy1p);
        *b1p -=  tmp *(*ixx1p)*(*ixz1p) + tmp2*(*ixy1p)*(*iyz1p);
        *b2p -=  tmp2*(*iyy1p)*(*iyz1p) + tmp *(*ixy1p)*(*ixz1p);
        *a11p += tmp3*(*ixx2p)*(*ixx2p) + tmp4*(*ixy2p)*(*ixy2p);
        *a12p += tmp3*(*ixx2p)*(*ixy2p) + tmp4*(*ixy2p)*(*iyy2p);
        *a22p += tmp4*(*iyy2p)*(*iyy2p) + tmp3*(*ixy2p)*(*ixy2p);
        *b1p -=  tmp3*(*ixx2p)*(*ixz2p) + tmp4*(*ixy2p)*(*iyz2p);
        *b2p -=  tmp4*(*iyy2p)*(*iyz2p) + tmp3*(*ixy2p)*(*ixz2p);
        *a11p += tmp5*(*ixx3p)*(*ixx3p) + tmp6*(*ixy3p)*(*ixy3p);
        *a12p += tmp5*(*ixx3p)*(*ixy3p) + tmp6*(*ixy3p)*(*iyy3p);
        *a22p += tmp6*(*iyy3p)*(*iyy3p) + tmp5*(*ixy3p)*(*ixy3p);
        *b1p -=  tmp5*(*ixx3p)*(*ixz3p) + tmp6*(*ixy3p)*(*iyz3p);
        *b2p -=  tmp6*(*iyy3p)*(*iyz3p) + tmp5*(*ixy3p)*(*ixz3p); 
        
        dup+=1; dvp+=1; maskp+=1; a11p+=1; a12p+=1; a22p+=1; b1p+=1; b2p+=1; 
        ix1p+=1; iy1p+=1; iz1p+=1; ixx1p+=1; ixy1p+=1; iyy1p+=1; ixz1p+=1; iyz1p+=1;
        ix2p+=1; iy2p+=1; iz2p+=1; ixx2p+=1; ixy2p+=1; iyy2p+=1; ixz2p+=1; iyz2p+=1;
        ix3p+=1; iy3p+=1; iz3p+=1; ixx3p+=1; ixy3p+=1; iyy3p+=1; ixz3p+=1; iyz3p+=1;
    }
}



#ifdef __cplusplus
}  // namespace ccore
#endif


