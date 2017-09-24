#include <cstdlib>
#include <cstring>
#include <iostream>

#include "epic.h"
#include "image.h"
#include "io.h"
#include "variational.h"


/* show usage information */
void usage(){
    std::cout << "usage:" << std::endl
              << "    ./epicflow image1 image2 edges matches outputfile [options]" << std::endl
              << "Compute EpicFlow between two images using given matches and edges and store it into a .flo file" << std::endl
              << "Images must be in PPM, JPG or PNG format." << std::endl
              << "Edges are read as width*height float32 values in a binary file" << std::endl
              << "Matches are read from a text file, each match in a different line, each line starting with 4 numbers corresponding to x1 y1 x2 y2" << std::endl
              << "" << std::endl
              << "options:" << std::endl 
              << "    -h, -help                                                print this message" << std::endl
              << "  interpolation parameters" << std::endl
              << "    -nw                                                      use Nadaraya-Watson instead of LA interpolator in the interpolation" << std::endl
              << "    -p, -prefnn             <int>(25)                        number of neighbors for consisteny checking in the interpolation" << std::endl
              << "    -n, -nn                 <int>(100)                       number of neighnors for the interpolation" << std::endl
              << "    -k                      <float>(0.8)                     coefficient of the sigmoid of the Gaussian kernel used in the interpolation" << std::endl
              << "  energy minimization parameters" << std::endl
              << "    -i, -iter               <int>(5)                         number of iterations for the energy minimization" << std::endl
              << "    -a, -alpha              <float>(1.0)                     weight of smoothness term" << std::endl
              << "    -g, -gamma              <float>(3.0)                     weight of gradient constancy assumption" << std::endl
              << "    -d, -delta              <float>(2.0)                     weight of color constancy assumption" << std::endl
              << "    -s, -sigma              <float>(0.8)                     standard deviation of Gaussian presmoothing kernel" << std::endl
              << "  predefined parameters" << std::endl
              << "    -sintel                                                  set the parameters to the one optimized on (a subset of) the MPI-Sintel dataset" << std::endl
              << "    -middlebury                                              set the parameters to the one optimized on the Middlebury dataset" << std::endl
              << "    -kitti                                                   set the parameters to the one optimized on the KITTI dataset" << std::endl
              << std::endl;
}


int main(int argc, char **argv){
    if( argc<6){
        if(argc>1) std::cerr << "Error, not enough arguments" << std::endl;
        usage();
        exit(1);
    }

    // read arguments
    ccore::color_image_t *im1 = ccore::color_image_load(argv[1]);
    ccore::color_image_t *im2 = ccore::color_image_load(argv[2]);
    ccore::float_image edges = ccore::read_edges(argv[3], im1->width, im1->height);
    ccore::float_image matches = ccore::read_matches(argv[4]);
    const char *outputfile = argv[5];

    // prepare variables
    epic::epic_params_t epic_params;
    epic::epic_params_default(&epic_params);
    ccore::variational_params_t flow_params;
    ccore::variational_params_default(&flow_params);
    ccore::image_t *wx = ccore::image_new(im1->width, im1->height), 
                   *wy = ccore::image_new(im1->width, im1->height);
    
    // read optional arguments 
    #define isarg(key)  !strcmp(a,key)
    int current_arg = 6;
    while(current_arg < argc ){
        const char* a = argv[current_arg++];
        if( isarg("-h") || isarg("-help") ) 
            usage();
        else if( isarg("-nw") ) 
            strcpy(epic_params.method, "NW");
        else if( isarg("-p") || isarg("-prefnn") ) 
            epic_params.pref_nn = atoi(argv[current_arg++]);
        else if( isarg("-n") || isarg("-nn") ) 
            epic_params.nn = atoi(argv[current_arg++]); 
        else if( isarg("-k") ) 
            epic_params.coef_kernel = atof(argv[current_arg++]);
        else if( isarg("-i") || isarg("-iter") ) 
            flow_params.niter_outer = atoi(argv[current_arg++]); 
        else if( isarg("-a") || isarg("-alpha") ) 
            flow_params.alpha= atof(argv[current_arg++]);  
        else if( isarg("-g") || isarg("-gamma") ) 
            flow_params.gamma= atof(argv[current_arg++]);                                  
        else if( isarg("-d") || isarg("-delta") ) 
            flow_params.delta= atof(argv[current_arg++]);  
        else if( isarg("-s") || isarg("-sigma") ) 
            flow_params.sigma= atof(argv[current_arg++]); 
        else if( isarg("-sintel") ){ 
            epic_params.pref_nn= 25; 
            epic_params.nn= 160; 
            epic_params.coef_kernel = 1.1f; 
            flow_params.niter_outer = 5;
            flow_params.alpha = 1.0f;
            flow_params.gamma = 0.72f;
            flow_params.delta = 0.0f;
            flow_params.sigma = 1.1f;            
        }  
        else if( isarg("-kitti") ){ 
            epic_params.pref_nn= 25; 
            epic_params.nn= 160; 
            epic_params.coef_kernel = 1.1f;
            flow_params.niter_outer = 2;
            flow_params.alpha = 1.0f;
            flow_params.gamma = 0.77f;
            flow_params.delta = 0.0f;
            flow_params.sigma = 1.7f; 
        }
        else if( isarg("-middlebury") ){ 
            epic_params.pref_nn= 15; 
            epic_params.nn= 65; 
            epic_params.coef_kernel = 0.2f;       
            flow_params.niter_outer = 25;
            flow_params.alpha = 1.0f;
            flow_params.gamma = 0.72f;
            flow_params.delta = 0.0f;
            flow_params.sigma = 1.1f;  
        }
        else{
            std::cerr << "unknown argument " << a << std::endl;
            usage();
            exit(1);
        }   
    }
    
    // compute interpolation and energy minimization
    ccore::color_image_t *imlab = rgb_to_lab(im1);
    epic::epic(wx, wy, imlab, &matches, &edges, &epic_params, 1);
    // energy minimization
    ccore::variational(wx, wy, im1, im2, &flow_params);
    // write output file and free memory
    ccore::writeFlowFile(outputfile, wx, wy);
    
    ccore::color_image_delete(im1);
    ccore::color_image_delete(imlab);
    ccore::color_image_delete(im2);
    free(matches.pixels);
    free(edges.pixels);
    ccore::image_delete(wx);
    ccore::image_delete(wy);

    return 0;
}
