# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:51:07 2017

@author: Matthias Höffken
"""

__all__ = [ "computeEpicFlow", "IllegalEpicFlowArgumentError", \
            "defaultVariationalParams", "defaultEpicFlowParams", \
            "sintelParams", "kittiParams", "middleburyParams" ]


_EpicLibName = 'libctypesEpicFlow.so'
_EpicLibPath = '.'

_VariParamDefaultCall = None
_EpicFlowParamDefaultCall = None
_EpicFlowCall = None


import numpy as np
from numpy.ctypeslib import as_ctypes
import ctypes as ct


####################################################################################

#_nd2ArrayAligned = np.ctypeslib.ndpointer( dtype=np.float32, ndim=2, flags='ALIGNED' )
#_nd2ArrayContinousAligned = np.ctypeslib.ndpointer( dtype=np.float32, ndim=2, flags=('ALIGNED','CONTIGUOUS') )

_floatPtr = ct.POINTER(ct.c_float)
_ndarray2pointer = lambda ndarray: _floatPtr( as_ctypes( ndarray ) )


def _isAligned( fNdArray, fAlignedTo=16 ):
    rest = fNdArray.ctypes.data % fAlignedTo
    if (0 == rest):
        return True
    
    print fNdArray.ctypes.data
    
    return False



class _image_t(ct.Structure):
    _fields_=[("width",  ct.c_int),   # Width(cols) of the image
              ("height", ct.c_int),   # Height(rows) of the image
              ("stride", ct.c_int),   # Width of the memory (width + padding such that it is a multiple of 4) - probably in bytes
              ("data",   ct.POINTER(ct.c_float) ) ]  # Image data, aligned

    m_ndImage = None
    
    @classmethod
    def fromArray( cls, f_ndimage ):
        f_ndimage = np.ascontiguousarray( f_ndimage, dtype=np.float32 )
        
        assert( f_ndimage.strides[1] == f_ndimage.itemsize )
        assert( f_ndimage.strides >= (f_ndimage.shape[0] * f_ndimage.itemsize) )
        assert( _isAligned( f_ndimage ) )       
        
        obj = cls( f_ndimage.shape[1], f_ndimage.shape[0], f_ndimage.strides[1], _ndarray2pointer( f_ndimage ) )
        obj.m_ndImage = f_ndimage
        
        return obj
    
    
    def getArray( self ):
        assert( self.stride >= (self.width * np.float32().itemsize) )
        assert( 0 == (self.stride % np.float32().itemsize) )
        
        arry = np.ctypeslib.as_array( self.data, shape=(self.height, self.stride / np.float32().itemsize) )
        if arry.shape[1] > self.width:
            arry = arry[:,:self.width]
        
        return arry



class _color_image_t(ct.Structure):
    """ structure for 3-channels image stored with one layer per color """
    _fields_=[("width",  ct.c_int),   # Width(cols) of the image
              ("height", ct.c_int),   # Height(rows) of the image
              ("stride", ct.c_int),   # Width of the memory (width + padding such that it is a multiple of 4)
              ("c1",   ct.POINTER(ct.c_float) ),  # Color 1, aligned
              ("c2",   ct.POINTER(ct.c_float) ),  # Color 2, aligned
              ("c3",   ct.POINTER(ct.c_float) )]  # Color 3, aligned
    
    m_ndImage = None
    
    
    @classmethod
    def fromArray( cls, f_ndimage ):
        
        if f_ndimage.ndim > 3:
            pass
        elif 3 == f_ndimage.ndim:
            if 3 != f_ndimage.shape[2]:
                f_ndimage = f_ndimage.swapaxes( 0, 2 )
        
        f_ndimage = np.ascontiguousarray( f_ndimage, dtype=np.float32 )
        assert( f_ndimage.strides[-1] == f_ndimage.itemsize )
        
        if f_ndimage.ndim == 2:
            # one channel (gray) image
            c1 = f_ndimage
            c2 = f_ndimage
            c3 = f_ndimage
        elif f_ndimage.ndim == 3:
            # color image
            c1 = f_ndimage[0,:,:]
            c2 = f_ndimage[1,:,:]
            c3 = f_ndimage[2,:,:]
        
        assert( c1.strides[1] == f_ndimage.itemsize )
        assert( _isAligned( c1 ) )
        assert( _isAligned( c2 ) )
        assert( _isAligned( c3 ) )
        
        obj = cls( c1.shape[1], c1.shape[0], c1.strides[1], 
                   _ndarray2pointer( c1 ), 
                   _ndarray2pointer( c2 ),
                   _ndarray2pointer( c3 ) )
        obj.m_ndImage = f_ndimage
        
        return obj

        


class _float_image_t( ct.Structure ):
    _fields_=[("pixels", ct.POINTER(ct.c_float) ),
              ("tx",     ct.c_int),   # n cols
              ("ty",     ct.c_int) ]  # n_rows
    
    m_ndImage = None    
    
    @classmethod
    def fromArray( cls, f_ndimage ):
        f_ndimage = np.ascontiguousarray( f_ndimage, dtype=np.float32 )
        
        assert( f_ndimage.strides[1] == f_ndimage.itemsize )
        assert( _isAligned( f_ndimage ) )
        
        obj = cls( _ndarray2pointer( f_ndimage ), f_ndimage.shape[1], f_ndimage.shape[0] )
        obj.m_ndImage = f_ndimage
        
        return obj
    


class _epic_params_t( ct.Structure ):
    _fields_=[("method",       ct.c_char * 20),  # method for interpolation: la (locally-weighted affine) or nw (nadaraya-watson)
              ("saliency_th",  ct.c_float),      # matches coming from pixels with a saliency below this threshold are removed before interpolation
              ("pref_nn",      ct.c_int),        # number of neighbors for consistent checking
              ("pref_th",      ct.c_float),      # threshold for the first prefiltering step
              ("nn",           ct.c_int),        # number of neighbors to consider for the interpolation
              ("coef_kernel",  ct.c_float),      # coefficient in the sigmoid of the interpolation kernel
              ("euc",          ct.c_float),      # constant added to the edge cost
              ("verbose",      ct.c_int)]        # verbose mode
    
    
    @classmethod
    def getDefault( cls ):
        params = cls()
        _EpicFlowParamDefaultCall( ct.byref(params) )
        return params

              
class _variational_params_t( ct.Structure ):
    _fields_=[("alpha",        ct.c_float),     # smoothness weight
              ("gamma",        ct.c_float),     # gradient constancy assumption weight
              ("delta",        ct.c_float),     # color constancy assumption weight
              ("sigma",        ct.c_float),     # presmoothing of the images
              ("niter_outer",  ct.c_int),       # number of outer fixed point iterations
              ("niter_inner",  ct.c_int),       # number of inner fixed point iterations
              ("niter_solver", ct.c_int),       # number of solver iterations 
              ("sor_omega",    ct.c_float)]     # omega parameter of sor method
    
    
    @classmethod
    def getDefault( cls ):
        params = cls()
        _VariParamDefaultCall( ct.byref(params) )
        return params


####################################################################################
## Loading DLL or shared library file

def _loadEpicFlowLibrary():
    eflib = np.ctypeslib.load_library( _EpicLibName, _EpicLibPath )
    
    # variational parameters
    global _VariParamDefaultCall
    _VariParamDefaultCall = eflib.variational_params_default
    _VariParamDefaultCall.argtypes = [ ct.POINTER( _variational_params_t ) ]
    
    # epic flow parameters
    global _EpicFlowParamDefaultCall
    _EpicFlowParamDefaultCall = eflib.epic_params_default  #(epic_params_t* params);
    _EpicFlowParamDefaultCall.argtypes = [ ct.POINTER( _epic_params_t ) ]
    
    # epic flow computation call
    global _EpicFlowCall
    _EpicFlowCall = eflib.computeEpicFlow
    _EpicFlowCall.argtypes = [ ct.POINTER( _color_image_t ),          # f_inImg1_p
                               ct.POINTER( _color_image_t ),          # f_inImg2_p
                               ct.POINTER( _float_image_t ),          # f_edges_p
                               ct.POINTER( _float_image_t ),          # f_matches_p
                               ct.POINTER( _epic_params_t ),          # f_epicParams_p
                               ct.POINTER( _variational_params_t ),   # f_flowParams_p
                               ct.POINTER( _image_t ),                # f_wx_p
                               ct.POINTER( _image_t ) ]               # f_wy_p

pass
_loadEpicFlowLibrary()

####################################################################################


def defaultVariationalParams():
    return _variational_params_t.getDefault()

def defaultEpicFlowParams():
    return _epic_params_t.getDefault()


def sintelParams():
    epicParams = defaultEpicFlowParams()
    epicParams.pref_nn      = 25      # number of neighbors for consistent checking
    epicParams.nn           = 160     # number of neighbors to consider for the interpolation
    epicParams.coef_kernel  = 1.1     # coefficient in the sigmoid of the interpolation kernel
    
    variParams = defaultVariationalParams()
    variParams.niter_outer  = 5       # number of outer fixed point iterations
    variParams.alpha        = 1.0     # smoothness weight
    variParams.gamma        = 0.72    # gradient constancy assumption weight
    variParams.delta        = 0.0     # color constancy assumption weight
    variParams.sigma        = 1.1     # presmoothing of the images
    
    return (epicParams, variParams)



def kittiParams():
    epicParams = defaultEpicFlowParams()
    epicParams.pref_nn      = 25      # number of neighbors for consistent checking
    epicParams.nn           = 160     # number of neighbors to consider for the interpolation
    epicParams.coef_kernel  = 1.1     # coefficient in the sigmoid of the interpolation kernel
    
    variParams = defaultVariationalParams()
    variParams.niter_outer  = 2       # number of outer fixed point iterations
    variParams.alpha        = 1.0     # smoothness weight
    variParams.gamma        = 0.77    # gradient constancy assumption weight
    variParams.delta        = 0.0     # color constancy assumption weight
    variParams.sigma        = 1.7     # presmoothing of the images
    
    return (epicParams, variParams)


def middleburyParams():
    epicParams = defaultEpicFlowParams()
    epicParams.pref_nn      = 15      # number of neighbors for consistent checking
    epicParams.nn           = 65      # number of neighbors to consider for the interpolation
    epicParams.coef_kernel  = 0.2     # coefficient in the sigmoid of the interpolation kernel
    
    variParams = defaultVariationalParams()
    variParams.niter_outer  = 25      # number of outer fixed point iterations
    variParams.alpha        = 1.0     # smoothness weight
    variParams.gamma        = 0.72    # gradient constancy assumption weight
    variParams.delta        = 0.0     # color constancy assumption weight
    variParams.sigma        = 1.1     # presmoothing of the images
    
    return (epicParams, variParams)
    
####################################################################################


class IllegalEpicFlowArgumentError(ValueError):
    pass




def _memId( fNdArray ):
    return fNdArray.ctypes.data


def computeEpicFlow( fImg1, fImg2, fEdgeImg, fMatches, fVariParams=None, fEpicFlowParams=None, fAllowEdgeImgModification=False ):
    
    if fImg1.shape != fImg2.shape:
        raise IllegalEpicFlowArgumentError("Input image shapes to not match: %s != %s", str(fImg1.shape), str(fImg2.shape) )
    elif fImg1.shape != fEdgeImg.shape:
        raise IllegalEpicFlowArgumentError("Edge Image shape does not fit: %s != %s", str(fImg1.shape), str(fEdgeImg.shape) )
    elif (fMatches.min() < 0) or ( fMatches[:,(0,2)].max() >= fImg1.shape[1] ) or ( fMatches[:,(1,3)].max() >= fImg1.shape[0] ):
        raise IllegalEpicFlowArgumentError("Matching indices are out of range" )
    
    lImg1    = _color_image_t.fromArray( fImg1 )
    lImg2    = _color_image_t.fromArray( fImg2 )
    lEdgeImg = _float_image_t.fromArray( fEdgeImg )
    lMatches = _float_image_t.fromArray( fMatches )
    
    # check if a copy of fEdgeImg is required
    if (not fAllowEdgeImgModification) and (_memId( fEdgeImg ) == _memId( lEdgeImg.m_ndImage ) ):
        lEdgeImg = _float_image_t.fromArray( lEdgeImg.m_ndImage )
    
    if None == fVariParams:
        fVariParams = defaultVariationalParams()
    
    if None == fEpicFlowParams:
        fEpicFlowParams = defaultEpicFlowParams()
    
    # create Output Memory
    lFlowRes = np.zeros( (2,) + fImg1.shape, dtype=np.float32 )
    
    # is it sure that newly created arrays are always aligned?
    assert( _isAligned( lFlowRes ) )
    
    lResWx = _image_t.fromArray( lFlowRes[0,:,:] )
    lResWy = _image_t.fromArray( lFlowRes[1,:,:] )
    
    assert( _memId( lResWx.m_ndImage ) == _memId( lFlowRes[0,:,:] ) )
    assert( _memId( lResWy.m_ndImage ) == _memId( lFlowRes[1,:,:] ) )
    
    
    # function call
    _EpicFlowCall( ct.byref( lImg1 ), ct.byref( lImg2 ),
                   ct.byref( lEdgeImg ), ct.byref( lMatches ),
                   ct.byref( fEpicFlowParams ), ct.byref( fVariParams ),
                   lResWx, lResWy )
    
    return lFlowRes

