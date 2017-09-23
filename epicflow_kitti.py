# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:56:05 2017

@author: Matthias Höffken
"""

from __future__ import print_function
from pyEpicFlow import computeEpicFlow, kittiParams
import sys
import numpy as np
import cv2
import array as ar


####################################################################################

def readMatches( fFname_s ):
    # fMatches: [ N x 4 ]
    
    lMatchFlines = file( fFname_s, "r" ).readlines()
    N = len( lMatchFlines )
    lMatches = np.empty( (N,4), np.float32 )
    lMatches[:] = float('nan')
    
    for i in xrange(N):
        lMatches[i,:] = map( float, lMatchFlines[i].split(" ") )
    
    return lMatches

def readEdges( fFname_s, fShape ):
    edges = None    
    
    with file( fFname_s, "rb" ) as f:
        data = f.read()
        edges = np.frombuffer( data, dtype=np.float32 )
    edges = edges.reshape( fShape )
    
    return edges.copy()


def writeFlowFile( fFname_s, fFlow ):
    # fFlow: [ 2 x H x W ]
    
    if fFlow.shape[0] == 2:
        fFlow = fFlow.transpose((1,2,0))
        
    fFlow = np.ascontiguousarray( fFlow, dtype=np.float32 )
    
    with file( fFname_s, "wb") as fout:
        ar.array( "f", [202021.25] ).tofile( fout )
        ar.array( "i", [ fFlow.shape[1], fFlow.shape[0] ] ).write( fout )
        fout.write( fFlow.data )
        fout.flush()


####################################################################################


def printUsage():
    print( "Usage:\n\tpython epicflow_kitti.py <image1> <image2> <edges> <matches> <outputfile>\n" )
    print( "Compute EpicFlow between two images using given matches and edges and store it into a .flo file" )
    print( "Images must be in PPM, JPG or PNG format.")
    print( "Edges are read as width*height float32 values in a binary file" )
    print( "Matches are read from a text file, each match in a different line, each line starting with 4 numbers corresponding to x1 y1 x2 y2" )


def Main():
    img1Fname  = sys.argv[1]
    img2Fname  = sys.argv[2]
    edgeFname  = sys.argv[3]
    matchFname = sys.argv[4]
    outFname   = sys.argv[5]
    
    img1 = cv2.imread( img1Fname, flags=cv2.IMREAD_GRAYSCALE )
    #img1 = cv2.imread( img1Fname ).transpose((2,0,1)).copy()
    img2 = cv2.imread( img2Fname, flags=cv2.IMREAD_GRAYSCALE )
    #img2 = cv2.imread( img2Fname ).transpose((2,0,1)).copy()
    
    edges = readEdges( edgeFname, img1.shape[-2:] )
    matches = readMatches( matchFname )
    
    #print img1.shape
    #print img2.shape
    #print edges.shape
    #print matches.shape
    
    # using the kitti data set
    EpicFlowpParams, VariParms = kittiParams()
    flow = computeEpicFlow( img1, img2, edges, matches, VariParms, EpicFlowpParams )
    #print flow.min(), flow.max(), flow.shape
    
    writeFlowFile( outFname, flow )


if __name__ == "__main__":
    if len(sys.argv) != 6:
        printUsage()
    else:
        Main()



