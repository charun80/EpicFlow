#ifndef ___EPIC_AUX_H___
#define ___EPIC_AUX_H___


/* This file contains auxiliary function for the interpolation: looking fpr the nearest neighbors with a geodesic distance, fitting/applying the interpolation */
#include "array_types.h"


namespace epic
{


/* structure for distance transform parameters */
typedef struct {
  int max_iter;
  float min_change;
} dt_params_t;


/* Compute the closest seeds using a geodesic distance for a subset of points, and the assignment of each pixel to the closest seeds
    best:    output containing the closest seeds for each query point
    dist:    output containing the distances to the closest seeds
    labels:  output containing the assignment of each pixel to the closest seed
    seeds:   2D positions of the seeds
    cost:    cost of going throw a pixel (ie that defines the geodesic distance)
    dt_params: distance transform parameters (NULL for default parameters)
    pixels:  2D positions of the query points
*/
void dist_trf_nnfield_subset( ccore::int_image* best, ccore::float_image* dist, ccore::int_image *labels,
                              const ccore::int_image* seeds, const ccore::float_image* cost, dt_params_t* dt_params,
                              const ccore::int_image* pixels, const int n_thread );
                       


/* fit nadaraya watson for a set of seeds 
    res:  (output) a nseeds*2 array containing the estimated displacement for each seed 
    nnf:   array of size nseeds*nn with index of the nn-closest seed
    dis:   as nnf but containing the distances to the corresponding seed
    vects: 2D vector of the matches
*/
void fit_nadarayawatson(ccore::float_image *res, const ccore::int_image *nnf, const ccore::float_image *dis, const ccore::float_image *vects, const int n_thread);

/* apply nadaraya watson interpolation
    newvects:    output containing the flow vector for each pixel 
    seedsvects:  input containing the estimated flow vector for each seed
    labels:      closest seed for each pixel
*/
void apply_nadarayawatson(ccore::float_image *newvects, const ccore::float_image *seedsvects, const ccore::int_image *labels, const int n_thread);


/* fit locally-weighted affine interpolation
    res: (output) a nseeds*6 array containing the estimated affine model for each seed
    nnf:   array of size nseeds*nn with index of the nn-closest seed
    dis:   as nnf but containing the distances to the corresponding seed
    seeds: original point of matches
    vects: 2D vector of the matches
*/
void fit_localaffine(ccore::float_image *res, const ccore::int_image *nnf, const ccore::float_image *dis, const ccore::int_image *seeds, const ccore::float_image *vects);


/* apply locally-weighted affine interpolation 
    newvects:    output containing the flow vector for each pixel
    seedaffine:  esimated affine transformation for each seed
    labels:      closest seed for each pixel
*/
void apply_localaffine(ccore::float_image *newvects, const ccore::float_image *seedsaffine, const ccore::int_image *labels, const int n_thread);


} // namespace epic



#endif // ___EPIC_AUX_H___
