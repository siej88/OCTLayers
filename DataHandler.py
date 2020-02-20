# -*- coding: utf-8 -*-

import numpy as N

class DataHandler(object):
    """Data File Manipulator"""

    def format(self, number):
        """format(int number)"""
        if (number == 0):
            return '0.00'
        string = str(number)
        length = len(string)
        if (length < 4):
            string += '0'
        return string[0: 4]

    def textToMatrix(self, path):
        """numpy.array textToImage(str path)"""
        data = open(path)
        lines = data.readlines()
        data.close()
        rowList = []
        for line in lines:
            row = line.strip('\n').split(',')
            row = list(map(float, row))
            rowList.append(N.array(row).astype(int))
        imageMatrix = rowList[0]
        for i in xrange(1, len(rowList)):
            imageMatrix = N.vstack([imageMatrix, rowList[i]])
        return imageMatrix

    def saveReport(self, experimentData, path):
        """saveReport(list experimentData, str path)"""
        data = open(path, 'w')
        data.write('ACCURACY\n\n')
        data.write('Name TP FP FN Acc\n')
        data.write('=================\n')
        length = len(experimentData)
        for i in xrange(length - 1):
            name =     experimentData[i]['name'] + ' '
            tp   = str(experimentData[i]['tp'])  + ' '
            fp   = str(experimentData[i]['fp'])  + ' '
            fn   = str(experimentData[i]['fn'])  + ' '
            acc  = self.format(experimentData[i]['acc']) + '\n'
            line = name + tp + fp + fn + acc
            data.write(line)
        data.write('Mean: ' + str(experimentData[-1]['accMean'])  + '\n')
        data.write('SD  : ' + str(experimentData[-1]['accStdev']) + '\n')
        data.write('\nCOVERAGE\n\n')
        data.write('Name L1 L2 L3 L4 L5 L6 L7 L8\n')
        data.write('============================\n')
        layers = len(experimentData[0]['coverage'])
        for i in xrange(length - 1):
            line     = experimentData[i]['name'] + ' '
            coverage = experimentData[i]['coverage']
            for j in xrange(layers):
                line += self.format(coverage[j]) + ' '
            line += '\n'
            data.write(line)
        meanLine  = 'Mean: '
        stdevLine = 'SD  : '
        for j in xrange(layers):
            meanLine  += self.format(experimentData[-1]['covMean' ][j]) + ' '
            stdevLine += self.format(experimentData[-1]['covStdev'][j]) + ' '
        meanLine  += '\n'
        stdevLine += '\n'
        data.write(meanLine)
        data.write(stdevLine)
        data.write('\nTIME\n\n')
        data.write('Name Seconds\n')
        data.write('============\n')
        for i in xrange(length - 1):
            name = experimentData[i]['name'] + ' '
            time = str(experimentData[i]['time']) + '\n'
            line = name + time
            data.write(line)
        data.close()