import cv2
import numpy as np

class BoardProcessor:
    '''
    Class containing all functionality to process chessboard image.
    '''

    def __init__(self, mat=None, debug=False):
        '''
        Constructor for BoardProcessor. Takes mat-image to be processed as input (optional).
        '''
        self.mat = mat
        self.processed = False
        self.failed = False
        self.errorMessage = None
        self.partArray = None
        self.debug = debug

    def wasSuccesfull(self):
        '''
        Returns true if last mat-image was processed succesfully.
        '''
        return self.failed

    def setDebug(self, bool):
        '''
        Define if debug options should be used.
        '''
        self.debug = bool

    def getErrorMessage(self):
        '''
        Returns errorMessage from last mat-image process. If no errors have occurred,
        None will be returned.
        '''
        return self.errorMessage

    def getProcessedImageParts(self):
        '''
        Returns array of mat-image parts that were extracted from the original picture.
        If the processing was successful, the array should contain 64 smaller mat-images,
        where each mat-image corresponds to single board square. If there was error,
        None is returned.
        '''
        return self.partArray

    def setMatImage(self, mat):
        '''
        Set mat-image that needs to be processed.
        '''
        self.reset()
        self.mat = mat

    def reset(self):
        '''
        Reset BoardProcessor.
        '''
        self.mat = None
        self.processed = False
        self.errorMessage = None
        self.failed = False
        self.partArray = None

    def process(self):
        '''
        Try to process image.
        '''
        self.processed = True
        self.failed = False
        self.errorMessage = None
        # convert mat-image to gray scale
        gray = convertToGrayScale(self.mat)
        if self.debug:
            cv2.imwrite('images/debug/gray.jpg', gray)
        # blur the image
        blur = blurImage(gray)
        if self.debug:
            cv2.imwrite('images/debug/blur.jpg', blur)
        # detect edges from the image
        edges = autoCanny(blur)
        if self.debug:
            cv2.imwrite('images/debug/edges.jpg', edges)
        # detect lines from the image
        lines = detecLines(edges)
        if self.debug:
            print("NUMBER OF LINES DETECTED WITH HOUGHLINES: " + str(len(lines)))
        # divide detected lines to horizontal and vertical lines
        hLines, vLines = horVertLines(lines)
        if self.debug:
            print("NUMBER OF HORIZONTAL LINES: " + str(len(hLines)))
            print("NUMBER OF VERTICAL LINES: " + str(len(vLines)))
        # merge similar lines together
        hLines = mergeSimilarLines(hLines)
        if self.debug:
            print("NUMBER OF HORIZONTAL LINES AFTER MERGING: " + str(len(hLines)))
            tmpMat = self.mat.copy()
            debugImage = drawLinesToMat(tmpMat, hLines)
            cv2.imwrite('images/debug/hLines.jpg', debugImage)
        vLines = mergeSimilarLines(vLines)
        if self.debug:
            print("NUMBER OF VERTICAL LINES AFTER MERGING: " + str(len(vLines)))
            tmpMat = self.mat.copy()
            debugImage = drawLinesToMat(tmpMat, vLines)
            cv2.imwrite('images/debug/vLines.jpg', debugImage)





''' FUNCTIONS TO PROCESS BOARD '''

def convertToGrayScale(mat):
    '''
    Convert given mat-image to gray scale. 
    '''
    return cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

def blurImage(mat):
    '''
    Blur given mat-image.
    '''
    return cv2.blur(mat, (10, 10))

def autoCanny(mat, sigma=0.33):
    '''
    Canny edge detection with automatic thresholds.
    '''
    # compute the median of the single channel pixel intensities
    v = np.median(mat)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(mat, lower, upper)

def detecLines(mat):
    '''
    Detect lines from mat-image by using HoughLines.
    '''
    lines = cv2.HoughLines(mat, 1, np.pi/180, 200)
    return np.reshape(lines, (-1, 2))

def horVertLines(lines):
    '''
    A line is given by rho and theta. Given a list of lines, returns a list of
    horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
    '''
    h = []
    v = []
    for distance, angle in lines:
        if angle < np.pi / 4 or angle > np.pi - np.pi / 4:
            v.append([distance, angle])
        else:
            h.append([distance, angle])
    return h, v

def mergeSimilarLines(lines):
    '''
    Merge similar lines together. Similar lines are lines that are close each other
    and their angle does not differ too much. Merging means that one of the 
    lines will be dropped.
    '''
    resultLines = []
    resultLines.append(lines[0])
    for line in lines:
        distanceL, angleL = line
        isNewLine = True
        for distanceR, angleR in resultLines:
            if distanceL == distanceR and angleL == angleR:
                isNewLine = False
                continue # same line
            elif abs(distanceL-distanceR) <= 30 and abs(angleL-angleR) <= 0.1:
                isNewLine = False
                continue # same line
            # not the same line
        if isNewLine:
            resultLines.append(line)
    return resultLines

def drawLinesToMat(mat, lines):
    '''
    Takes a mat-image and list of lines as input and draws the lines to image.
    '''
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))
        cv2.line(mat, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return mat
