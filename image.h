#ifndef __IMAGE_H_
#define __IMAGE_H_


#ifdef __cplusplus
namespace ccore {
extern "C" {
#endif




/********** STRUCTURES *********/

/* structure for 1-channel image */
typedef struct image_s
{
  int width;        /* Width of the image */
  int height;       /* Height of the image */
  int stride;       /* Width of the memory (width + paddind such that it is a multiple of 4) */
  float *data;      /* Image data, aligned */
} image_t;


typedef struct const_image_s
{
  int width;            /* Width of the image */
  int height;           /* Height of the image */
  int stride;           /* Width of the memory (width + paddind such that it is a multiple of 4) */
  const float *data;    /* Image data, aligned */
} image_ct;


/* structure for 3-channels image stored with one layer per color, it assumes that c2 = c1+width*height and c3 = c2+width*height. */
typedef struct color_image_s
{
    int width;            /* Width of the image */
    int height;            /* Height of the image */
    int stride;         /* Width of the memory (width + paddind such that it is a multiple of 4) */
    float *c1;            /* Color 1, aligned */
    float *c2;            /* Color 2, consecutive to c1*/
    float *c3;            /* Color 3, consecutive to c2 */
} color_image_t;


typedef struct const_color_image_s
{
    int width;             /* Width of the image */
    int height;            /* Height of the image */
    int stride;            /* Width of the memory (width + paddind such that it is a multiple of 4) */
    const float *c1;       /* Color 1, aligned */
    const float *c2;       /* Color 2, consecutive to c1*/
    const float *c3;       /* Color 3, consecutive to c2 */
} color_image_ct;


/* structure for convolutions */
typedef struct convolution_s
{
    int order;            /* Order of the convolution */
    float *coeffs;        /* Coefficients */
    float *coeffs_accu;    /* Accumulated coefficients */
} convolution_t;


typedef struct const_convolution_s
{
    int order;                   /* Order of the convolution */
    const float *coeffs;         /* Coefficients */
    const float *coeffs_accu;    /* Accumulated coefficients */
} convolution_ct;


/********** const casting **********/

const image_ct *const_image_cast( const image_t *image );
const color_image_ct *const_color_image_cast( const color_image_t *image );
const convolution_ct *const_convolution_cast( const convolution_t *conv );

image_t *image_cast( const image_ct *image );
color_image_t *color_image_cast( const color_image_ct *image );
convolution_t *convolution_cast( const convolution_ct *conv );

/********** Create/Delete **********/

/* allocate a new image of size width x height */
image_t *image_new(const int width, const int height);

/* allocate a new image and copy the content from src */
image_t *image_cpy(const image_ct *src);

/* set all pixels values to zeros */
void image_erase(image_t *image);

/* free memory of an image */
void image_delete(image_t *image);

/* multiply an image by a scalar */
void image_mul_scalar(image_t *image, const float scalar);

/* allocate a new color image of size width x height */
color_image_t *color_image_new(const int width, const int height);

/* allocate a new color image and copy the content from src */
color_image_t *color_image_cpy(const color_image_ct *src);

/* set all pixels values to zeros */
void color_image_erase(color_image_t *image);

/* free memory of a color image */
void color_image_delete(color_image_t *image);

/************ Convolution ******/

/* return half coefficient of a gaussian filter */
float *gaussian_filter(const float sigma, int *fSize);

/* helper function for convolution_new  */
void convolve_extract_coeffs(const int order, const float *half_coeffs, float *coeffs, float *coeffs_accu, const int even);

/* create a convolution structure with a given order, half_coeffs, symmetric or anti-symmetric according to even parameter */
convolution_t *convolution_new(int order, const float *half_coeffs, const int even);

/* perform an horizontal convolution of an image */
void convolve_horiz(image_t *dest, const image_ct *src, const convolution_ct *conv);

/* perform a vertical convolution of an image */
void convolve_vert(image_t *dest, const image_ct *src, const convolution_ct *conv);

/* free memory of a convolution structure */
void convolution_delete(convolution_t *conv);

/* perform horizontal and/or vertical convolution to a color image */
void color_image_convolve_hv(color_image_t *dst, const color_image_ct *src, const convolution_ct *horiz_conv, const convolution_ct *vert_conv);

/************ Others **********/

/* return a new image in lab color space */
color_image_t *rgb_to_lab(const color_image_ct *im);

/* compute the saliency of a given image */
image_t* saliency(const color_image_ct *im, float sigma_image, float sigma_matrix );



#ifdef __cplusplus
}  // extern C
}  // namespace ccore
#endif

#endif // __IMAGE_H_

