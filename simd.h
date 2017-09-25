#ifndef __SIMD_H_
#define __SIMD_H_


#ifdef __AVX__
    
    #define SIMD_MESSAGE "Using AVX instruction set"


    #include <immintrin.h>


    // Number of bytes in SIMD registers
    #define NSIMDBYTES 32

    // SIMD vector type
    typedef __v8sf simdsf_t;
    
    #define simdsf_sqrt(x)   __builtin_ia32_sqrtps256(x)
    #define simdsf_max(x,y)  __builtin_ia32_maxps256(x,y)
    
    inline static simdsf_t simdsf_init( float x ) 
    { 
        simdsf_t sx = {x,x,x,x, x,x,x,x};
        return ( sx ); 
    }

#else
#ifdef __SSE__

    #define SIMD_MESSAGE "Using SSE instruction set"

    #include <xmmintrin.h>

    // Number of bytes in SIMD registers
    #define NSIMDBYTES 16
    
    typedef __v4sf simdsf_t;
    
    
    #define simdsf_sqrt(x)   __builtin_ia32_sqrtps(x)
    #define simdsf_max(x,y)  __builtin_ia32_maxps(x,y)
    
    inline static simdsf_t simdsf_init( float x ) 
    { 
        simdsf_t sx = {x,x,x,x};
        return ( sx ); 
    }

#else

    #define "Neither SSE nor AVX instruction set available - no compilation possible"
    #error SIMD_MESSAGE 

#endif // __SSE__
#endif // __AVX__


static const size_t NSimdBytes = NSIMDBYTES;
static const size_t NSimdFloats = NSIMDBYTES / sizeof(float);


inline static simdsf_t* simdsf_ptrcast( void* fptr ) 
{
    return ((simdsf_t*) fptr);
}

#pragma message( SIMD_MESSAGE )

#endif // __SIMD_H_
