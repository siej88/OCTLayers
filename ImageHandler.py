# -*- coding: utf-8 -*-

import numpy as N
from PIL import Image as img

import MathTools as mt

class ImageHandler(object):
    """Image File and Image Manipulator"""

    def getGrayscaleMatrix(self, path):
        """numpy.array getGrayscaleMatrix(str path)"""
        inputImage = img.open(path).convert('L')
        grayscaleMatrix = N.array(inputImage)/255.
        return N.copy(grayscaleMatrix)
    
    def saveImage(self, matrix, path):
        """saveImage(numpy.array matrix, str path)"""
        outputImage = img.fromarray(N.uint8(matrix*255))
        outputImage.save(path)

    def binarizeMatrix(self, matrix, threshold):
        """numpy.array binarizeMatrix(numpy.array matrix, float threshold)"""
        binarizedMatrix = N.zeros_like(matrix)
        binarizedMatrix[N.where(matrix > threshold)] = 1.
        binarizedMatrix[N.where(matrix <= threshold)] = 0.
        return N.copy(binarizedMatrix)

    def getRelevantRegion(self, imageMatrix, weight = 5., threshold = 0.4):
        """numpy.array getRelevantRegion(numpy.array imageMatrix, float weight, float threshold)"""
        mathTools = mt.MathTools()
        regionMatrix  = N.copy(imageMatrix)
        regionMatrix  = mathTools.gaussianFilter(regionMatrix, weight)
        regionMatrix  = mathTools.gradient(regionMatrix)
        regionMatrix  = mathTools.normalize(regionMatrix)
        regionMatrix /= N.max(regionMatrix)
        thresholdMatrix = self.binarizeMatrix(regionMatrix, threshold)
        histogram = N.sum(thresholdMatrix, axis = 1)
        reverse   = N.flipud(histogram)
        l  = N.where(histogram > 0)[0][0]
        h  = imageMatrix.shape[0] - 1
        h -= N.where(reverse   > 0)[0][0]
        return N.array([l, h])

    def selectLayers(self, matrix, count):
        """numpy.array selectLayers(numpy.array matrix, int count)"""
        resultMatrix = N.zeros_like(matrix)
        height = matrix.shape[0]
        width  = matrix.shape[1]
        for i in xrange(width):
            column  = N.copy(matrix[:, i])
            indexes = column.argsort()
            for k in xrange(count):
                index = N.int(indexes[height - 1 - k])
                resultMatrix[index, i] = 1.0
        return resultMatrix