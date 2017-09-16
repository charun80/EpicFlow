# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:51:07 2017

@author: Matthias Höffken
"""

__all__ = [ "defaultVariationalParams", "defaultEpicFlowParams" ]


_EpicLibName = 'libctypesEpicFlow.so'
_EpicLibPath = '.'

_VariParamDefaultCall = None
_EpicFlowParamDefaultCall = None
_EpicFlowCall = None


import numpy as np
import ctypes as ct


####################################################################################


class _image_t(ct.Structure):
    _fields_=[("width",  ct.c_int),   # Width(cols) of the image
              ("height", ct.c_int),   # Height(rows) of the image
              ("stride", ct.c_int),   # Width of the memory (width + padding such that it is a multiple of 4) - probably in bytes
              ("data",   np.ctypeslib.ndpointer( dtype=np.float32, ndim=2, flags='ALIGNED' )) ]  # Image data, aligned
    
    @classmethod
    def fromArray( cls, f_ndimage ):
        f_ndimage = np.ascontiguousarray( f_ndimage, dtype=np.float32 )
        
        assert( f_ndimage.stride[1] == f_ndimage.itemsize )
        assert( f_ndimage.stride >= (f_ndimage.shape[0] * f_ndimage.itemsize) )
        obj = cls( f_ndimage.shape[1], f_ndimage.shape[0], f_ndimage.stride[1], f_ndimage )
        return obj
    
    
    def getArray( self ):
        assert( self.stride >= (self.width * np.float32().itemsize) )
        assert( 0 == (self.stride % np.float32().itemsize) )
        
        arry = np.ctypeslib.as_array( self.data, shape=(self.height, self.stride / np.float32().itemsize) )
        if arry.shape[1] > self.width:
            arry = arry[:,:self.width]
        
        return arry


class _EpicFlowRes( ct.Structure ):
    _fields_=[("m_wx_p", _image_t),   # x flow component
              ("m_wy_p", _image_t)]   # y flow component
    
    def flowx(self):
        return self.m_wx_p.getArray()
    
    def flowy(self):
        return self.m_wy_p.getArray()



class _color_image_t(ct.Structure):
    """ structure for 3-channels image stored with one layer per color """
    _fields_=[("width",  ct.c_int),   # Width(cols) of the image
              ("height", ct.c_int),   # Height(rows) of the image
              ("stride", ct.c_int),   # Width of the memory (width + padding such that it is a multiple of 4)
              ("c1",   np.ctypeslib.ndpointer( dtype=np.float32, ndim=2, flags='ALIGNED' )),  # Color 1, aligned
              ("c2",   np.ctypeslib.ndpointer( dtype=np.float32, ndim=2, flags='ALIGNED' )),  # Color 2, aligned
              ("c3",   np.ctypeslib.ndpointer( dtype=np.float32, ndim=2, flags='ALIGNED' ))]  # Color 3, aligned
    
    @classmethod
    def fromArray( cls, f_ndimage ):
        
        if f_ndimage.ndim > 3:
            pass
        elif 3 == f_ndimage.ndim:
            if 3 != f_ndimage.shape[2]:
                f_ndimage = f_ndimage.swapaxes( 0, 2 )
        
        f_ndimage = np.ascontiguousarray( f_ndimage, dtype=np.float32 )
        assert( f_ndimage.stride[-1] == f_ndimage.itemsize )
        
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
        
        assert( c1.stride[1] == f_ndimage.itemsize )
        obj = cls( c1.shape[1], c1.shape[0], c1.stride[1], c1, c2, c3 )
        return obj

        


class _float_image_t( ct.Structure ):
    _fields_=[("pixels", np.ctypeslib.ndpointer( dtype=np.float32, ndim=2, flags=('ALIGNED','CONTIGUOUS') )),
              ("tx",     ct.c_int),   # n cols
              ("ty",     ct.c_int) ]  # n_rows
    
    @classmethod
    def fromArray( cls, f_ndimage ):
        f_ndimage = np.ascontiguousarray( f_ndimage, dtype=np.float32 )
        
        assert( f_ndimage.stride[1] == f_ndimage.itemsize )
        obj = cls( f_ndimage.shape[1], f_ndimage.shape[0], f_ndimage )
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
    _EpicFlowCall.restype = _EpicFlowRes    # Return type
    _EpicFlowCall.argtypes = [ ct.POINTER( _color_image_t ),          # f_inImg1_p
                                ct.POINTER( _color_image_t ),          # f_inImg2_p
                                ct.POINTER( _float_image_t ),          # f_edges_p
                                ct.POINTER( _float_image_t ),          # f_matches_p
                                ct.POINTER( _epic_params_t ),          # f_epicParams_p
                                ct.POINTER( _variational_params_t ) ]  # f_flowParams_p

pass
_loadEpicFlowLibrary()

####################################################################################


def defaultVariationalParams():
    return _variational_params_t.getDefault()

def defaultEpicFlowParams():
    return _epic_params_t.getDefault()



