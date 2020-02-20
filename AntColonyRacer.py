# -*- coding: utf-8 -*-

import sys
import numpy as N
import ImageHandler as ih
import MathTools as mt

class AntColonyRacer(object):
    """Ant Colony Optimisation Engine"""

    def __init__(self):
        # Toolboxes
        self.imageHandler = ih.ImageHandler()
        self.mathTools    = mt.MathTools()
        # Matrices
        self.bilateralMatrix = None
        self.filteredMatrix  = None
        self.pheromoneMatrix = None
        self.traceMatrix     = None
        # Lookups
        self.directionVectors = []
        self.directionVectors.append([ 0, 1])
        self.directionVectors.append([-1, 1])
        self.directionVectors.append([ 1, 0])
        self.directionVectors.append([ 1, 1])
        self.directionIndexes = {}
        self.directionIndexes[-1] = {-1:3, 0:2, 1:1}
        self.directionIndexes[ 0] = {-1:0, 0:0, 1:0}
        self.directionIndexes[ 1] = {-1:1, 0:2, 1:3}

    def generateHeuristicMatrices(self, imageMatrix):
        """list generateHeuristicMatrices(numpy.array imageMatrix)"""

        # Image Filter
        filteredMatrix = self.mathTools.bilateralFilter(imageMatrix, 3, 10, 15)
        self.bilateralMatrix = N.copy(filteredMatrix)
        filteredMatrix = self.mathTools.medianFilter(filteredMatrix, (1, 15))
        self.filteredMatrix = filteredMatrix

        # Heuristic Direction Matrices
        horizontalMatrix = self.mathTools.directionalGradient(filteredMatrix, 0)
        uDiagonalMatrix  = self.mathTools.directionalGradient(filteredMatrix, 1)
        verticalMatrix   = self.mathTools.directionalGradient(filteredMatrix, 2)
        lDiagonalMatrix  = self.mathTools.directionalGradient(filteredMatrix, 3)

        # Scale Brightness
        horizontalMatrix /= N.max(horizontalMatrix)
        uDiagonalMatrix  /= N.max(uDiagonalMatrix)
        verticalMatrix   /= N.max(verticalMatrix)
        lDiagonalMatrix  /= N.max(lDiagonalMatrix)

        # Heuristic Matrix Container
        heuristicMatrices = []
        heuristicMatrices.append(horizontalMatrix)
        heuristicMatrices.append(uDiagonalMatrix)
        heuristicMatrices.append(verticalMatrix)
        heuristicMatrices.append(lDiagonalMatrix)

        # Return Container
        return heuristicMatrices

    def generateTraceMatrix(self, imageMatrix, parameterSet):
        """numpy.array generateTraceMatrix(numpy.array imageMatrix, dict parameterSet)"""

        # Parameters
        cycleCount = parameterSet['cycleCount']
        rho        = parameterSet['rho']
        psi        = parameterSet['psi']

        # Heuristic Matrices
        heuristicMatrices = self.generateHeuristicMatrices(imageMatrix)

        # Pheromone Matrix
        pheromoneMatrix = self.mathTools.gradient(self.filteredMatrix) + N.ones_like(imageMatrix) * 0.01
        initialPheromoneMatrix = N.copy(pheromoneMatrix)

        # Trace Matrix
        self.traceMatrix = N.zeros_like(imageMatrix, dtype = float)

        # Image Dimensions
        matrixHeight = imageMatrix.shape[0]
        matrixWidth  = imageMatrix.shape[1]
        stepCount = matrixWidth - 1

        # Ant Starting Search Space
        limit = self.imageHandler.getRelevantRegion(imageMatrix)
        indexes  = list(N.ndindex(matrixHeight, 1))
        indexes  = indexes[limit[0]: limit[1] + 1]
        antCount = len(indexes)

        # For Each Cycle
        for cycle in xrange(cycleCount):

            # Random Ant Distribution
            N.random.shuffle(indexes)
            antSet = indexes[0: antCount]

            # For Each Ant
            for i in xrange(antCount):

                # Default Direction Index
                previousIndex = 1

                # For Each Step
                for step in xrange(stepCount):

                    # Current Ant Position
                    x = antSet[i][0]
                    y = antSet[i][1]

                    # Pheromone Window
                    u = self.mathTools.clamp(x - 1, matrixHeight - 1)
                    v = self.mathTools.clamp(x + 1, matrixHeight - 1)
                    pheromoneWindow = N.copy(pheromoneMatrix[u: v + 1, y + 1])

                    # Weighted Random Ant Direction
                    probabilities = pheromoneWindow / N.sum(pheromoneWindow)
                    destinationIndex = N.random.choice(N.arange(len(probabilities)), p = probabilities)

                    # Update Previous Index
                    previousIndex = destinationIndex

                    # Compute Movement Delta
                    dx = destinationIndex - 1
                    dy = 1

                    # Update Ant Position
                    newX = self.mathTools.clamp(x + dx, matrixHeight - 1)
                    newY = self.mathTools.clamp(y + dy, matrixWidth  - 1)
                    antSet[i] = (newX, newY)

                    # Heuristic Matrix
                    directionIndex = self.directionIndexes[dx][dy]
                    heuristicMatrix = heuristicMatrices[directionIndex]

                    # Local Pheromone Update
                    pheromone = pheromoneMatrix[newX, newY]
                    heuristic = heuristicMatrix[newX, newY]
                    pheromoneMatrix[newX, newY] += rho * heuristic

                    # Trace Matrix Update
                    self.traceMatrix[newX, newY] += 0.01

            # Global Pheromone Update
            pheromoneMatrix = (1 - psi) * pheromoneMatrix

        # Truncate Matrix Values to [0.,1.]
        pheromoneMatrix = self.mathTools.normalize(pheromoneMatrix)

        # Store Pheromone Matrix
        self.pheromoneMatrix = pheromoneMatrix

        # Return Trace Matrix
        return self.traceMatrix
        
    def run(self, imageMatrix, parameterSet):
        """numpy.array run(numpy.array imageMatrix, dict parameterSet)"""

        # Algorithm - First Pass
        traceMatrix    = self.generateTraceMatrix(imageMatrix, parameterSet)

        # Algorithm - Second Pass
        reverseMatrix  = self.generateTraceMatrix(N.fliplr(imageMatrix), parameterSet)
        traceMatrix   += N.fliplr(reverseMatrix)

        # Adjust Trace Image
        traceMatrix /= N.max(traceMatrix)

        # Return Trace Matrix
        return traceMatrix

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print 'python AntColonyRacer.py sourceFile destinationFile'
    else:
        # Toolboxes
        imageHandler = ih.ImageHandler()

        # Load Image Matrix
        sourceFile  = sys.argv[1]
        imageMatrix = imageHandler.getGrayscaleMatrix(sourceFile)

        # Fill Parameters
        parameterSet = {}
        parameterSet['cycleCount'] = 10
        parameterSet['rho'] = 1.0
        parameterSet['psi'] = 0.1

        # Run Algorithm
        antColonyRacer = AntColonyRacer()
        traceMatrix = antColonyRacer.run(imageMatrix, parameterSet)

        # Save Image
        destinationFile = sys.argv[2]
        imageHandler.saveImage(traceMatrix, destinationFile)