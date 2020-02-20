# -*- coding: utf-8 -*-

import numpy as N
import scipy.signal as sig
import scipy.ndimage.filters as flt
import cv2 as cv

class MathTools(object):
    """Mathematical Toolset"""

    def gradient(self, inputMatrix, diagonal = False, horizontalWeight = 1., verticalWeight = 1.):
        """numpy.array gradient(numpy.array inputMatrix[, bool diagonal = False,
        float horizontalWeight = 1., float verticalWeight = 1.])"""
        if (not diagonal):
            horizontalMask = N.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])*horizontalWeight
            verticalMask = N.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])*verticalWeight
        else:
            horizontalMask = N.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])*horizontalWeight
            verticalMask = N.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]])*verticalWeight
        horizontalMatrix = self.convolve(inputMatrix, horizontalMask)
        verticalMatrix = self.convolve(inputMatrix, verticalMask)
        gradientMatrix = N.sqrt(N.square(horizontalMatrix) + N.square(verticalMatrix))
        return gradientMatrix
    
    def directionalGradient(self, inputMatrix, directionIndex, thickness = 1):
        """numpy.array directionalGradient(numpy.array inputMatrix, int directionIndex, int thickness)"""
        diameter = 2 * thickness + 1
        maskArray = N.ones(diameter)
        maskArray[thickness] = 0
        maskArray[thickness + 1:] = -N.ones(thickness)
        maskArray /= N.float(thickness)
        if (directionIndex == 1 or directionIndex == 3):
            mask = N.diag(maskArray)
            if (directionIndex == 3):
                mask = N.fliplr(mask)
        elif (directionIndex == 0 or directionIndex == 2):
            mask = N.zeros((diameter, diameter))
            mask[thickness] = maskArray
            if (directionIndex == 0):
                mask = mask.T
        else:
            mask = N.ones((diameter, diameter)) / N.float(N.square(diameter))
        gradientMatrix = N.abs(self.convolve(inputMatrix, mask))
        return gradientMatrix

    def gaussianFilter(self, inputMatrix, standardDeviation = 2.):
        """numpy.array gaussianFilter(numpy.array inputMatrix[, float standardDeviation = 2.])"""
        return flt.gaussian_filter(inputMatrix, standardDeviation)
    
    def medianFilter(self, inputMatrix, diameter = 11):
        """numpy.array medianFilter(numpy.array inputMatrix[, int diameter = 11])"""
        return flt.median_filter(inputMatrix, diameter)

    def bilateralFilter(self, inputMatrix, iterations = 3, diameter = 10, sigma = 15.):
        """numpy.array bilateralFilter(numpy.array inputMatrix[, int iterations = 3, int diameter = 10, float sigma = 15.])"""
        filteredMatrix = N.uint8(inputMatrix * 255.)
        for _ in xrange(iterations):
            filteredMatrix = cv.bilateralFilter(filteredMatrix, diameter, sigma, sigma)
        return filteredMatrix / 255.

    def normalize(self, inputMatrix):
        """numpy.array normalize(numpy.array inputMatrix)"""
        normalizedMatrix = N.copy(inputMatrix)
        normalizedMatrix[N.where(inputMatrix < 0.)] = 0.
        normalizedMatrix[N.where(inputMatrix > 1.)] = 1.
        return normalizedMatrix

    def convolve(self, matrix, mask):
        """numpy.array convolve(numpy.array matrix, numpy.array mask)"""
        return sig.convolve2d(matrix, mask, 'same', 'symm')

    def clamp(self, value, boundary):
        """int clamp(int value, int boundary)"""
        value = min(max(0, value), boundary)
        return value