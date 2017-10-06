#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "simd.h"

#include "image.h"



#define MINMAX(a,b) MIN( MAX(a,0) , b-1 )


#ifdef __cplusplus
namespace ccore
{
#endif




/********** Create/Delete **********/

/* allocate a new image of size width x height */
image_t *image_new(const int width, const int height){
    image_t *image = (image_t*) malloc(sizeof(image_t));
    if(image == NULL){
        fprintf(stderr, "Error: image_new() - not enough memory !\n");
        exit(1);
    }
    image->width = width;
    image->height = height;  
    image->stride = ( (width + NSimdFloats - 1) / NSimdFloats ) * NSimdFloats;
    assert( 0 == (image->stride % NSimdFloats) );
    
    const int lMemAlignError = posix_memalign( (void**)(&(image->data)), NSimdBytes, image->stride * height * sizeof(float) );
    
    if( 0 != lMemAlignError )
    {
        fprintf( stderr, "Error: allocating memory in image_new(): %d !\n", lMemAlignError );
        exit(1);
    }
    return image;
}

/* allocate a new image and copy the content from src */
image_t *image_cpy(const image_t *src){
    image_t *dst = image_new(src->width, src->height);
    memcpy(dst->data, src->data, src->stride*src->height*sizeof(float));
    return dst;
}

/* set all pixels values to zeros */
void image_erase(image_t *image){
    memset(image->data, 0, image->stride*image->height*sizeof(float));
}


/* multiply an image by a scalar */
void image_mul_scalar(image_t *image, const float scalar){
    int i;
    simdsf_t* imp = simdsf_ptrcast( image->data );
    const simdsf_t scalarp = simdsf_init( scalar );
    for( i=0 ; i < ((image->stride / NSimdFloats) * image->height); i++){
        (*imp) *= scalarp;
        imp+=1;
    }
}

/* free memory of an image */
void image_delete(image_t *image){
    if(image == NULL){
        //fprintf(stderr, "Warning: Delete image --> Ignore action (image not allocated)\n");
    }else{
    free(image->data);
    free(image);
    }
}


/* allocate a new color image of size width x height */
color_image_t *color_image_new(const int width, const int height)
{
    color_image_t *image = (color_image_t*) malloc(sizeof(color_image_t));
    if(image == NULL){
        fprintf(stderr, "Error: color_image_new() - not enough memory !\n");
        exit(1);
    }
    image->width = width;
    image->height = height;  
    image->stride = ( (width + NSimdFloats - 1) / NSimdFloats ) * NSimdFloats;
    assert( 0 == (image->stride % NSimdFloats) );
    
    const int lMemAlignError = posix_memalign( (void**)(&(image->c1)), NSimdBytes, 3 * image->stride * height * sizeof(float) );
    if( 0 != lMemAlignError )
    {
        fprintf(stderr, "Error: allocating memory in color_image_new(): %d !\n", lMemAlignError);
        exit(1);
    }
    image->c2 = image->c1 + image->stride*height;
    image->c3 = image->c2 + image->stride*height;
    return image;
}

/* allocate a new color image and copy the content from src */
color_image_t *color_image_cpy(const color_image_t *src){
    color_image_t *dst = color_image_new(src->width, src->height);
    
    if (src->c1 == src->c2)
    {
        // gray scale image
        assert( src->c2 == src->c3 );
        memcpy(dst->c1, src->c1, src->stride * src->height * sizeof(float));
        memcpy(dst->c2, src->c1, src->stride * src->height * sizeof(float));
        memcpy(dst->c3, src->c1, src->stride * src->height * sizeof(float));
    }
    else 
    {
        // image with three channels
        assert( src->c1 != src->c3  );
        assert( src->c2 != src->c3  );
        memcpy( dst->c1, src->c1, 3*src->stride * src->height*sizeof(float) );
    }
    
    return dst;
}

/* set all pixels values to zeros */
void color_image_erase(color_image_t *image)
{
    if (image->c1 == image->c2)
    {
        // gray scale image
        assert( image->c2 == image->c3 );
        
        memset( image->c1, 0, image->stride * image->height * sizeof(float));
        memset( image->c2, 0, image->stride * image->height * sizeof(float));
        memset( image->c3, 0, image->stride * image->height * sizeof(float));
    }
    else 
    {
        // image with three channels
        assert( image->c1 != image->c3  );
        assert( image->c2 != image->c3  );
        memset( image->c1, 0, 3 * image->stride * image->height*sizeof(float) );
    }
}


/* free memory of a color image */
void color_image_delete(color_image_t *image){
    if(image){
        free(image->c1); // c2 and c3 was allocated at the same moment
        free(image);
    }
}


/************ Convolution ******/

/* return half coefficient of a gaussian filter
Details:
- return a float* containing the coefficient from middle to border of the filter, so starting by 0,
- it so contains half of the coefficient.
- sigma is the standard deviation.
- filter_order is an output where the size of the output array is stored */
float *gaussian_filter(const float sigma, int *filter_order){
    if(sigma == 0.0f){
        fprintf(stderr, "gaussian_filter() error: sigma is zeros\n");
        exit(1);
    }
    if(!filter_order){
        fprintf(stderr, "gaussian_filter() error: filter_order is null\n");
        exit(1);
    }
    // computer the filter order as 1 + 2* floor(3*sigma)
    *filter_order = floor(3*sigma)+1; 
    if ( *filter_order == 0 )
        *filter_order = 1; 
    // compute coefficients
    float data[ 2*(*filter_order) + 1 ];
    {
        const float alpha = 1.0f/(2.0f*sigma*sigma);
        float sum = 0.0f;

        for( int i = -(*filter_order) ; i<=*filter_order ; i++){
            data[i+(*filter_order)] = expf( -i*i*alpha );
            sum += data[i+(*filter_order)];
        }
        for( int i = -(*filter_order) ; i<=*filter_order ; i++){
            data[i+(*filter_order)] /= sum;
        }
    }
    
    // fill the output
    float *data2 = (float*) malloc(sizeof(float)*(*filter_order+1));
    if(data2 == NULL ){
        fprintf(stderr, "gaussian_filter() error: not enough memory\n");
        exit(1);
    }
    memcpy(data2, &data[*filter_order], sizeof(float)*(*filter_order)+sizeof(float));
    
    return data2;
}

/* given half of the coef, compute the full coefficients and the accumulated coefficients */
static void convolve_extract_coeffs(const int order, const float *half_coeffs, float *coeffs, float *coeffs_accu, const int even){
    int i;
    float accu = 0.0;
    if(even){
        for(i = 0 ; i <= order; i++){
	        coeffs[order - i] = coeffs[order + i] = half_coeffs[i];
        }
        for(i = 0 ; i <= order; i++){
	        accu += coeffs[i];
	        coeffs_accu[2 * order - i] = coeffs_accu[i] = accu;
        }
    }else{
        for(i = 0; i <= order; i++){
	        coeffs[order - i] = +half_coeffs[i];
	        coeffs[order + i] = -half_coeffs[i];
        }
        for(i = 0 ; i <= order; i++){
            accu += coeffs[i];
	        coeffs_accu[i] = accu;
	        coeffs_accu[2 * order - i]= -accu;
        }
    }
}

/* create a convolution structure with a given order, half_coeffs, symmetric or anti-symmetric according to even parameter */
convolution_t *convolution_new(const int order, const float *half_coeffs, const int even){
    convolution_t *conv = (convolution_t *) malloc(sizeof(convolution_t));
    if(conv == NULL){
        fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
        exit(1);
    }
    conv->order = order;
    conv->coeffs = (float *) malloc((2 * order + 1) * sizeof(float));
    if(conv->coeffs == NULL){
        fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
        free(conv);
        exit(1);
    }
    conv->coeffs_accu = (float *) malloc((2 * order + 1) * sizeof(float));
    if(conv->coeffs_accu == NULL){
        fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
        free(conv->coeffs);
        free(conv);
        exit(1);
    }
    convolve_extract_coeffs(order, half_coeffs, conv->coeffs,conv->coeffs_accu, even);
    return conv;
}

static void convolve_vert_fast_3(image_t *dst, const image_t *src, const convolution_t *conv){
    const int iterline = (src->stride / NSimdFloats) + 1;
    const float *coeff = conv->coeffs;
    //const float *coeff_accu = conv->coeffs_accu;
    simdsf_t *srcp = simdsf_ptrcast( src->data ), 
             *dstp = simdsf_ptrcast( dst->data );
    simdsf_t *srcp_p1 = simdsf_ptrcast( src->data + src->stride );
    int i;
    for(i=iterline ; --i ; ){ // first line
        *dstp = (coeff[0]+coeff[1])*(*srcp) + coeff[2]*(*srcp_p1);
        dstp+=1; srcp+=1; srcp_p1+=1;
    }
    simdsf_t* srcp_m1 = simdsf_ptrcast( src->data ); 
    for(i=src->height-1 ; --i ; ){ // others line
        int j;
        for(j=iterline ; --j ; ){
            *dstp = coeff[0]*(*srcp_m1) + coeff[1]*(*srcp) + coeff[2]*(*srcp_p1);
            dstp+=1; srcp_m1+=1; srcp+=1; srcp_p1+=1;
        }
    }       
    for(i=iterline ; --i ; ){ // last line
        *dstp = coeff[0]*(*srcp_m1) + (coeff[1]+coeff[2])*(*srcp);
        dstp+=1; srcp_m1+=1; srcp+=1; 
    }  
}

static void convolve_vert_fast_5(image_t *dst, const image_t *src, const convolution_t *conv){
    const int iterline = (src->stride / NSimdFloats) + 1;
    const float *coeff = conv->coeffs;
    //const float *coeff_accu = conv->coeffs_accu;
    simdsf_t *srcp = simdsf_ptrcast( src->data ), 
             *dstp = simdsf_ptrcast( dst->data );
    simdsf_t *srcp_p1 = simdsf_ptrcast( src->data + src->stride );
    simdsf_t *srcp_p2 = simdsf_ptrcast( src->data + (2*src->stride) );
    int i;
    for(i=iterline ; --i ; ){ // first line
        *dstp = (coeff[0]+coeff[1]+coeff[2])*(*srcp) + coeff[3]*(*srcp_p1) + coeff[4]*(*srcp_p2);
        dstp+=1; srcp+=1; srcp_p1+=1; srcp_p2+=1;
    }
    simdsf_t* srcp_m1 = simdsf_ptrcast( src->data );
    for(i=iterline ; --i ; ){ // second line
        *dstp = (coeff[0]+coeff[1])*(*srcp_m1) + coeff[2]*(*srcp) + coeff[3]*(*srcp_p1) + coeff[4]*(*srcp_p2);
        dstp+=1; srcp_m1+=1; srcp+=1; srcp_p1+=1; srcp_p2+=1;
    }   
    simdsf_t* srcp_m2 = simdsf_ptrcast( src->data );
    for(i=src->height-3 ; --i ; ){ // others line
        int j;
        for(j=iterline ; --j ; ){
            *dstp = coeff[0]*(*srcp_m2) + coeff[1]*(*srcp_m1) + coeff[2]*(*srcp) + coeff[3]*(*srcp_p1) + coeff[4]*(*srcp_p2);
            dstp+=1; srcp_m2+=1;srcp_m1+=1; srcp+=1; srcp_p1+=1; srcp_p2+=1;
        }
    }    
    for(i=iterline ; --i ; ){ // second to last line
        *dstp = coeff[0]*(*srcp_m2) + coeff[1]*(*srcp_m1) + coeff[2]*(*srcp) + (coeff[3]+coeff[4])*(*srcp_p1);
        dstp+=1; srcp_m2+=1;srcp_m1+=1; srcp+=1; srcp_p1+=1;
    }          
    for(i=iterline ; --i ; ){ // last line
        *dstp = coeff[0]*(*srcp_m2) + coeff[1]*(*srcp_m1) + (coeff[2]+coeff[3]+coeff[4])*(*srcp);
        dstp+=1; srcp_m2+=1;srcp_m1+=1; srcp+=1; 
    }  
}


static void convolve_horiz_fast_3(image_t *dst, const image_t *src, const convolution_t *conv){
    const int stride_minus_1 = src->stride-1;
    const int iterline = src->stride / NSimdFloats;
    const float *coeff = conv->coeffs;
    simdsf_t *srcp = simdsf_ptrcast( src->data ), 
             *dstp = simdsf_ptrcast( dst->data );
    
    // create shifted version of src
    float *src_p1 = NULL;
    {
        const int lMemAlignError = posix_memalign( (void**)(&(src_p1)), NSimdBytes, src->stride * sizeof(float) * 2 );

        if( 0 != lMemAlignError )
        {
            fprintf( stderr, "Error: allocating memory in convolve_horiz_fast_3(): %d !\n", lMemAlignError );
            exit(1);
        }
    }
    float *src_m1 = src_p1 + src->stride;
    
    for( int j=0; j<src->height; j++){
        
        float *srcptr = (float*) srcp;
        const float right_coef = srcptr[src->width-1];
        for( int i = src->width; i < src->stride; i++)
            srcptr[i] = right_coef;
        
        src_m1[0] = srcptr[0];
        memcpy(src_m1+1, srcptr , sizeof(float)*stride_minus_1);
        src_p1[stride_minus_1] = right_coef;
        memcpy(src_p1, srcptr+1, sizeof(float)*stride_minus_1);
        
        // this requires memory alignemnts
        simdsf_t *srcp_p1 = simdsf_ptrcast( src_p1 ), 
                 *srcp_m1 = simdsf_ptrcast( src_m1 );
        
        for(int i=0; i < iterline; i++){
            *dstp = coeff[0]*(*srcp_m1) + coeff[1]*(*srcp) + coeff[2]*(*srcp_p1);
            dstp+=1; srcp_m1+=1; srcp+=1; srcp_p1+=1;
        }
    }
    
    free( src_p1 );
    src_p1 = src_m1 = NULL;
}


static void convolve_horiz_fast_5(image_t *dst, const image_t *src, const convolution_t *conv){
    const int stride_minus_1 = src->stride-1;
    const int stride_minus_2 = src->stride-2;
    const int iterline = src->stride / NSimdFloats;
    const float *coeff = conv->coeffs;
    
    simdsf_t *srcp = simdsf_ptrcast( src->data ),
             *dstp = simdsf_ptrcast( dst->data );
    
    float *src_p1 = NULL;
    {
        const int lMemAlignError = posix_memalign( (void**)(&(src_p1)), NSimdBytes, src->stride * sizeof(float) * 4 );

        if( 0 != lMemAlignError )
        {
            fprintf( stderr, "Error: allocating memory in convolve_horiz_fast_5(): %d !\n", lMemAlignError );
            exit(1);
        }
    }
    float *src_p2 = src_p1 + src->stride;
    float *src_m1 = src_p2 + src->stride;
    float *src_m2 = src_m1 + src->stride;
    
    for( int j=0;j<src->height;j++){
        
        float *srcptr = (float*) srcp;
        const float right_coef = srcptr[src->width-1];
        for( int i=src->width;i<src->stride;i++)
            srcptr[i] = right_coef;
        
        src_m1[0] = srcptr[0];
        memcpy(src_m1+1, srcptr , sizeof(float)*stride_minus_1);
        src_m2[0] = srcptr[0];
        src_m2[1] = srcptr[0];
        memcpy(src_m2+2, srcptr , sizeof(float)*stride_minus_2);
        src_p1[stride_minus_1] = right_coef;
        memcpy(src_p1, srcptr+1, sizeof(float)*stride_minus_1);
        src_p2[stride_minus_1] = right_coef;
        src_p2[stride_minus_2] = right_coef;
        memcpy(src_p2, srcptr+2, sizeof(float)*stride_minus_2);
        
        // this requires memory alignemnts
        simdsf_t *srcp_p1 = simdsf_ptrcast( src_p1 ), 
                 *srcp_p2 = simdsf_ptrcast( src_p2 ), 
                 *srcp_m1 = simdsf_ptrcast( src_m1 ), 
                 *srcp_m2 = simdsf_ptrcast( src_m2 );
        
        for( int i=0;i<iterline;i++){
            *dstp = coeff[0]*(*srcp_m2) + coeff[1]*(*srcp_m1) + coeff[2]*(*srcp) + coeff[3]*(*srcp_p1) + coeff[4]*(*srcp_p2);
            dstp+=1; srcp_m2 +=1; srcp_m1+=1; srcp+=1; srcp_p1+=1; srcp_p2+=1;
        }
    }
    free(src_p1);
    src_p1 = src_p2 = src_m1 = src_m2 = NULL;
}


/* perform an horizontal convolution of an image */
void convolve_horiz(image_t *dest, const image_t *src, const convolution_t *conv){
    if(conv->order==1){
        convolve_horiz_fast_3(dest,src,conv);
        return;
    }else if(conv->order==2){
        convolve_horiz_fast_5(dest,src,conv);
        return;    
    }
    float *in = src->data;
    float * out = dest->data;
    int i, j, ii;
    float *o = out;
    int i0 = -conv->order;
    int i1 = +conv->order;
    float *coeff = conv->coeffs + conv->order;
    float *coeff_accu = conv->coeffs_accu + conv->order;
    for(j = 0; j < src->height; j++){
        const float *al = in + j * src->stride;
        const float *f0 = coeff + i0;
        float sum;
        for(i = 0; i < -i0; i++){
	        sum=coeff_accu[-i - 1] * al[0];
	        for(ii = i1 + i; ii >= 0; ii--){
	            sum += coeff[ii - i] * al[ii];
            }
	        *o++ = sum;
        }
        for(; i < src->width - i1; i++){
	        sum = 0;
	        for(ii = i1 - i0; ii >= 0; ii--){
	            sum += f0[ii] * al[ii];
            }
	        al++;
	        *o++ = sum;
        }
        for(; i < src->width; i++){
	        sum = coeff_accu[src->width - i] * al[src->width - i0 - 1 - i];
	        for(ii = src->width - i0 - 1 - i; ii >= 0; ii--){
	            sum += f0[ii] * al[ii];
            }
	        al++;
	        *o++ = sum;
        }
        for(i = 0; i < src->stride - src->width; i++){
	        o++;
        }
    }
}

/* perform a vertical convolution of an image */
void convolve_vert(image_t *dest, const image_t *src, const convolution_t *conv){
    if(conv->order==1){
        convolve_vert_fast_3(dest,src,conv);
        return;
    }else if(conv->order==2){
        convolve_vert_fast_5(dest,src,conv);
        return;    
    }
    float *in = src->data;
    float *out = dest->data;
    int i0 = -conv->order;
    int i1 = +conv->order;
    float *coeff = conv->coeffs + conv->order;
    float *coeff_accu = conv->coeffs_accu + conv->order;
    int i, j, ii;
    float *o = out;
    const float *alast = in + src->stride * (src->height - 1);
    const float *f0 = coeff + i0;
    for(i = 0; i < -i0; i++){
        float fa = coeff_accu[-i - 1];
        const float *al = in + i * src->stride;
        for(j = 0; j < src->width; j++){
	        float sum = fa * in[j];
	        for(ii = -i; ii <= i1; ii++){
	            sum += coeff[ii] * al[j + ii * src->stride];
            }
	        *o++ = sum;
        }
        for(j = 0; j < src->stride - src->width; j++) 
	    {
	        o++;
        }
    }
    for(; i < src->height - i1; i++){
        const float *al = in + (i + i0) * src->stride;
        for(j = 0; j < src->width; j++){
	        float sum = 0;
	        const float *al2 = al;
	        for(ii = 0; ii <= i1 - i0; ii++){
	            sum += f0[ii] * al2[0];
	            al2 += src->stride;
            }
	        *o++ = sum;
	        al++;
        }
        for(j = 0; j < src->stride - src->width; j++){
	        o++;
        }
    }
    for(;i < src->height; i++){
        float fa = coeff_accu[src->height - i];
        const float *al = in + i * src->stride;
        for(j = 0; j < src->width; j++){
	        float sum = fa * alast[j];
	        for(ii = i0; ii <= src->height - 1 - i; ii++){
	            sum += coeff[ii] * al[j + ii * src->stride];
            }
	        *o++ = sum;
        }
        for(j = 0; j < src->stride - src->width; j++){
	        o++;
        }
    }
}

/* free memory of a convolution structure */
void convolution_delete(convolution_t *conv){
    if(conv)
    {
        free(conv->coeffs);
        free(conv->coeffs_accu);
        free(conv);
    }
}

/* perform horizontal and/or vertical convolution to a color image */
void color_image_convolve_hv(color_image_t *dst, const color_image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv){
    const int width = src->width, 
             height = src->height,
             stride = src->stride;
    // separate channels of images
    const image_t src_red   = {width,height,stride,src->c1}, 
                  src_green = {width,height,stride,src->c2},
                  src_blue  = {width,height,stride,src->c3};
                    
    image_t   dst_red = {width,height,stride,dst->c1},
            dst_green = {width,height,stride,dst->c2}, 
             dst_blue = {width,height,stride,dst->c3};
    // horizontal and vertical
    if ( (horiz_conv != NULL) && (vert_conv != NULL) )
    {
        float *tmp_data = NULL;
        const int lMemAlignError = posix_memalign( (void**)(&(tmp_data)), NSimdBytes, stride * height * sizeof(float) );
        if( 0 != lMemAlignError )
        {
            fprintf( stderr, "Error: allocating memory in color_image_convolve_hv(): %d !\n", lMemAlignError );
            exit(1);
        }
        
        image_t tmp = {width,height,stride,tmp_data};   
        
        // perform convolution for each channel
        convolve_horiz(&tmp,&src_red,horiz_conv); 
        convolve_vert(&dst_red,&tmp,vert_conv); 
        convolve_horiz(&tmp,&src_green,horiz_conv);
        convolve_vert(&dst_green,&tmp,vert_conv); 
        convolve_horiz(&tmp,&src_blue,horiz_conv); 
        convolve_vert(&dst_blue,&tmp,vert_conv);
        free(tmp_data);
    } else if( (horiz_conv != NULL) && (vert_conv == NULL) ) { // only horizontal
        convolve_horiz(&dst_red,&src_red,horiz_conv);
        convolve_horiz(&dst_green,&src_green,horiz_conv);
        convolve_horiz(&dst_blue,&src_blue,horiz_conv);
    } else if( (vert_conv != NULL) && (horiz_conv == NULL) ) { // only vertical
        convolve_vert(&dst_red,&src_red,vert_conv);
        convolve_vert(&dst_green,&src_green,vert_conv);
        convolve_vert(&dst_blue,&src_blue,vert_conv);
    }
}

/************ Others **********/

static inline float pow2( float f ) {return f*f;}

/* return a new image in lab color space */
color_image_t *rgb_to_lab(const color_image_t *im){

    color_image_t *res = color_image_new(im->width, im->height);
    const int npix = im->stride*im->height;

    const float T=0.008856f;
    const float color_attenuation = 1.5f;
    int i;
    for(i=0 ; i<npix ; i++){
        const float r = im->c1[i]/255.f;
        const float g = im->c2[i]/255.f;
        const float b = im->c3[i]/255.f;
        float X = 0.412453f * r + 0.357580f * g + 0.180423f * b;
        float Y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
        float Z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
        X /= 0.950456f;
        Z /= 1.088754f;
        const float Y3 = powf( Y, 1.f / 3.f );
        const float fX = X>T ? powf( X, 1.f / 3.f ) : 7.787f * X + 16.f/116.f;
        const float fY = Y>T ? Y3 : 7.787f * Y + 16.f/116.;
        const float fZ = Z>T ? powf( Z, 1.f/3.f) : 7.787f * Z + 16.f/116.f;
        const float L = Y>T ? 116.f * Y3 - 16.f : 903.3f * Y;
        const float A = 500.f * (fX - fY);
        const float B = 200.f * (fY - fZ);
        // correct L*a*b*: dark area or light area have less reliable colors
        const float correct_lab = expf( -color_attenuation * pow2(pow2(L/100.f) - 0.6f)); 
        res->c1[i] = L;
        res->c2[i] = A*correct_lab;
        res->c3[i] = B*correct_lab;
    }
    return res;

}   

/* compute the saliency of a given image */
image_t* saliency(const color_image_t *im, float sigma_image, float sigma_matrix ){
    const int width = im->width;
    const int height = im->height; 
    
    
    // smooth image
    color_image_t *sim = color_image_new(width, height);
    {
        int filter_size = -1;
        float* const presmooth_filter = gaussian_filter(sigma_image, &filter_size);
        convolution_t* const presmoothing = convolution_new(filter_size, presmooth_filter, 1);
        
        color_image_convolve_hv(sim, im, presmoothing, presmoothing);
        convolution_delete(presmoothing);
        free(presmooth_filter);
    }
  
    // compute derivatives
    color_image_t *imx = color_image_new(width, height), 
                  *imy = color_image_new(width, height);
    {
        float deriv_filter[2] = {0.0f, -0.5f};
        convolution_t* const deriv = convolution_new(1, deriv_filter, 0);
        
        color_image_convolve_hv(imx, sim, deriv, NULL);
        color_image_convolve_hv(imy, sim, NULL, deriv);
        convolution_delete(deriv);
    }
    
    // compute autocorrelation matrix
    image_t *imxx = image_new(width, height),
            *imxy = image_new(width, height),
            *imyy = image_new(width, height);
    {
        simdsf_t *imx1p = simdsf_ptrcast( imx->c1 ),
                 *imx2p = simdsf_ptrcast( imx->c2 ),
                 *imx3p = simdsf_ptrcast( imx->c3 ),

                 *imy1p = simdsf_ptrcast( imy->c1 ),
                 *imy2p = simdsf_ptrcast( imy->c2 ),
                 *imy3p = simdsf_ptrcast( imy->c3 ),

                 *imxxp = simdsf_ptrcast( imxx->data ),
                 *imxyp = simdsf_ptrcast( imxy->data ),
                 *imyyp = simdsf_ptrcast( imyy->data );

        for( int i = 0 ; i < (height * im->stride / NSimdFloats); i++){
            *imxxp = (*imx1p)*(*imx1p) + (*imx2p)*(*imx2p) + (*imx3p)*(*imx3p);
            *imxyp = (*imx1p)*(*imy1p) + (*imx2p)*(*imy2p) + (*imx3p)*(*imy3p);
            *imyyp = (*imy1p)*(*imy1p) + (*imy2p)*(*imy2p) + (*imy3p)*(*imy3p);
            imxxp+=1; imxyp+=1; imyyp+=1;
            imx1p+=1; imx2p+=1; imx3p+=1;
            imy1p+=1; imy2p+=1; imy3p+=1;
        }
    }
    
    // integrate autocorrelation matrix
    image_t* const tmp = image_new(width, height);
    {
        int filter_size = -1;
        float* const smooth_filter = gaussian_filter(sigma_matrix, &filter_size);
        convolution_t* const smoothing = convolution_new(filter_size, smooth_filter, 1);
        
        convolve_horiz(tmp, imxx, smoothing);
        convolve_vert(imxx, tmp, smoothing);
        convolve_horiz(tmp, imxy, smoothing);
        convolve_vert(imxy, tmp, smoothing);    
        convolve_horiz(tmp, imyy, smoothing);
        convolve_vert(imyy, tmp, smoothing);    
        convolution_delete(smoothing);
        free(smooth_filter);
    }
    
    // compute smallest eigenvalue
    {
        simdsf_t vzeros = simdsf_init( 0.0f );
        simdsf_t vhalf = simdsf_init( 0.5f );
        simdsf_t *tmpp = simdsf_ptrcast( tmp->data );
        simdsf_t *imxxp = simdsf_ptrcast( imxx->data ); 
        simdsf_t *imxyp = simdsf_ptrcast( imxy->data ); 
        simdsf_t *imyyp = simdsf_ptrcast( imyy->data );
        
        for( int i = 0 ; i < (height * im->stride / NSimdFloats); i++){
            (*tmpp) = vhalf * ( (*imxxp)+(*imyyp) );
            (*tmpp) = simdsf_sqrt( simdsf_max( vzeros, 
                                              (*tmpp) - simdsf_sqrt( simdsf_max( vzeros, 
                                                                                 (*tmpp)*(*tmpp) + (*imxyp)*(*imxyp) - (*imxxp)*(*imyyp) ) )) );
            tmpp+=1;
            imxyp+=1;
            imxxp+=1;
            imyyp+=1;
        }
    }
    
    image_delete(imxx); 
    image_delete(imxy); 
    image_delete(imyy);
    color_image_delete(imx); 
    color_image_delete(imy);
    color_image_delete(sim);
    
    return tmp;
}


#ifdef __cplusplus
}  // namespace ccore
#endif


