# -*- coding: utf-8 -*-

import time
import numpy as N
import os.path as P

import ImageHandler as ih
import DataHandler as dh
import AntColonyRacer as acr
import MathTools as mt

def antColony(subject, imageNumber):

    # Toolboxes
    imageHandler = ih.ImageHandler()
    dataHandler  = dh.DataHandler()
    mathTools    = mt.MathTools()

    # Ant Colony Racer
    antColonyRacer = acr.AntColonyRacer()

    # Construct Image Name
    imageName = str(imageNumber)
    if (imageNumber < 10):
        imageName = '0' + imageName
    imageName = 'Subject' + str(subject) + '_' + imageName

    # Load Image Matrix
    imageMatrix = imageHandler.getGrayscaleMatrix(P.join('images', imageName + '.png'))

    # Fill Parameters
    parameterSet = {}
    parameterSet['cycleCount'] = 10
    parameterSet['rho'] = 1.0
    parameterSet['psi'] = 0.1

    # Start Timer
    start = time.time()

    # Algorithm Run
    traceMatrix = antColonyRacer.run(imageMatrix, parameterSet)

    # End Timer
    end = time.time()

    # Select Layers
    layerMatrix = imageHandler.selectLayers(traceMatrix, 8)

    # Evaluate
    truthLayers = getGroundTruth(subject, imageNumber)
    truthMatrix = getTruthMatrix(truthLayers, imageMatrix.shape)
    accuracy    = getAccuracy(layerMatrix, truthMatrix)
    coverage    = getCoverage(layerMatrix, truthLayers)

    # Save Images
    imageHandler.saveImage(imageMatrix, P.join('output', imageName + ' 0.png'))
    imageHandler.saveImage(traceMatrix, P.join('output', imageName + ' 1.png'))
    imageHandler.saveImage(layerMatrix, P.join('output', imageName + ' 2.png'))
    imageHandler.saveImage(truthMatrix, P.join('output', imageName + ' 3.png'))

    # Save Statistics
    stats = {}
    stats['name'] = imageName
    stats['time'] = int(end - start)
    stats['tp'] = accuracy['tp']
    stats['fp'] = accuracy['fp']
    stats['fn'] = accuracy['fn']
    stats['acc'] = accuracy['acc']
    stats['coverage'] = coverage[0: 8] / coverage[-1]

    # Return Statistics
    return stats

def getGroundTruth(subject, imageNumber):
    dataHandler = dh.DataHandler()
    reference = dataHandler.textToMatrix(P.join('data', 'Subject' + str(subject) + '_1.txt')).T
    width  = reference.shape[1]
    result = N.zeros((8, width))
    for l in xrange(1, 9):
        matrix = dataHandler.textToMatrix(P.join('data', 'Subject' + str(subject) + '_' + str(l) + '.txt')).T
        result[l - 1, :] = matrix[imageNumber - 1, :]
    return result

def getTruthMatrix(truthLayers, matrixShape):
    truthMatrix = N.zeros(matrixShape)
    layers = truthLayers.shape[0]
    width  = truthLayers.shape[1]
    for i in xrange(layers):
        for j in xrange(width):
            row = int(truthLayers[i, j])
            truthMatrix[row, j] = 1
    return truthMatrix

def getAccuracy(layerMatrix, truthMatrix):
    tp = 0
    fp = 0
    fn = 0
    height = truthMatrix.shape[0]
    width  = truthMatrix.shape[1]
    for i in xrange(height):
        for j in xrange(width):
            truth = truthMatrix[i, j]
            layer = layerMatrix[i, j]
            if (truth == 1 and layer == 1):
                tp += 1
            elif (truth == 1 and layer == 0):
                fn += 1
            elif (truth == 0 and layer == 1):
                fp += 1
    acc = float(tp) / float(tp + fp + fn)
    return { 'tp': tp, 'fp': fp, 'fn': fn, 'acc': acc }

def getCoverage(layerMatrix, truthLayers):
    layers = truthLayers.shape[0]
    width  = truthLayers.shape[1]
    result = N.zeros(9).T
    result[8] = width
    for i in xrange(layers):
        for j in xrange(width):
            row = int(truthLayers[i, j])
            result[i] += int(layerMatrix[row, j] == 1)
    return result

if (__name__ == '__main__'):
    dataHandler = dh.DataHandler()
    experimentData = []
    totalCoverage  = N.zeros(8).astype(float)
    totalAccuracy  = 0.
    dataPoints     = 0.
    for subject in xrange(1, 11):
        for imageNumber in xrange(1, 11):
            stats = antColony(subject, imageNumber)
            totalAccuracy += stats['acc']
            totalCoverage += stats['coverage']
            dataPoints += 1.
            experimentData.append(stats)
    accMean = totalAccuracy / dataPoints
    covMean = totalCoverage / dataPoints
    accStdev = 0
    covStdev = N.zeros(8).astype(float)
    for i in xrange(int(dataPoints)):
        accStdev += N.square(experimentData[i]['acc']      - accMean)
        covStdev += N.square(experimentData[i]['coverage'] - covMean)
    accStdev = N.sqrt(accStdev / dataPoints)
    covStdev = N.sqrt(covStdev / dataPoints)
    lastData = {}
    lastData['accMean']  = accMean
    lastData['covMean']  = covMean
    lastData['accStdev'] = accStdev
    lastData['covStdev'] = covStdev
    experimentData.append(lastData)
    dataHandler.saveReport(experimentData, P.join('output', 'report.txt'))
            
            