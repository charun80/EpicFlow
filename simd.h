#ifndef __SIMD_H_
#define __SIMD_H_


#ifdef __cplusplus
    #include <cstddef>
#endif



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
#if defined __ARM_NEON__ || defined __ARM_NEON

    #define SIMD_MESSAGE "Using NEON instruction set"

    #include <arm_neon.h>
    
    // Number of bytes in SIMD registers
    #define NSIMDBYTES 16
    
    typedef float32x4_t simdsf_t;
    
    static inline simdsf_t simdsf_reciprocal( simdsf_t x ) 
    { 
        float32x4_t estimate = vrecpeq_f32( x );
        estimate = vmulq_f32(vrecpsq_f32( estimate, x ), estimate);
        estimate = vmulq_f32(vrecpsq_f32( estimate, x ), estimate);
        return estimate;
    }

    static inline float32x4_t __simdsf_rsqrt_1iteration( float32x4_t x, float32x4_t estimate)
    {
        float32x4_t estimate2 = vmulq_f32(estimate, x );
        return vmulq_f32(estimate, vrsqrtsq_f32(estimate2, estimate));
    }

    static inline float32x4_t __simdsf_rsqrt1( float32x4_t x )
    {
        float32x4_t estimate = vrsqrteq_f32( x );
        return __simdsf_rsqrt_1iteration( x, estimate );
    }

    static inline float32x4_t __simdsf_rsqrt2( float32x4_t x )
    {
        float32x4_t estimate = vrsqrteq_f32( x );
        
        estimate = __simdsf_rsqrt_1iteration( x, estimate );
        return __simdsf_rsqrt_1iteration( x, estimate );
    }

    static inline float32x4_t __simdsf_rsqrt3( float32x4_t x )
    {
        float32x4_t estimate = vrsqrteq_f32( x );
        
        estimate = __simdsf_rsqrt_1iteration( x, estimate );
        estimate = __simdsf_rsqrt_1iteration( x, estimate );
        return __simdsf_rsqrt_1iteration( x, estimate );
    }
    
    
    static inline simdsf_t simdsf_rsqrt( simdsf_t x ) {
        return __simdsf_rsqrt3( x );
    }
    
    
    
    static inline simdsf_t simdsf_sqrt( simdsf_t x  )
    {
        // see https://github.com/scoopr/vectorial/blob/master/include/vectorial/simd4f_neon.h
        
        return vreinterpretq_f32_u32(vandq_u32( vtstq_u32(vreinterpretq_u32_f32(x),  
                                                          vreinterpretq_u32_f32(x)), 
                                                vreinterpretq_u32_f32( simdsf_reciprocal(simdsf_rsqrt(x)) )
                                              ) );
    }
    
    
    static inline simdsf_t simdsf_max( simdsf_t x, simdsf_t y )
    {
        return vmaxq_f32( x, y );
    }
    
    
    inline static simdsf_t simdsf_init( float x ) 
    { 
        simdsf_t sx = {x,x,x,x};
        return ( sx ); 
    }
    

#else

    #define SIMD_MESSAGE "Neither SSE nor AVX instruction set available - no compilation possible"
    #error SIMD_MESSAGE 

#endif // __NEON__
#endif // __SSE__
#endif // __AVX__


static const size_t NSimdBytes = NSIMDBYTES;
static const size_t NSimdFloats = NSIMDBYTES / sizeof(float);


inline static simdsf_t* simdsf_ptrcast( void* fptr ) 
{
    return ((simdsf_t*) fptr);
}

inline static const simdsf_t* simdsf_const_ptrcast( const void* fptr ) 
{
    return ((const simdsf_t*) fptr);
}

#pragma message( SIMD_MESSAGE )
#undef SIMD_MESSAGE

#undef NSIMDBYTES

#endif // __SIMD_H_
