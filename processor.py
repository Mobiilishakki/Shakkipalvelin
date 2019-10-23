import cv2
import numpy as np
<<<<<<< HEAD
import math
import statistics
import heapq
=======
import statistics
import math
>>>>>>> 80b10f053b3b75cccdef33d044d71b21711bb119

############################################################################################################
############################################################################################################

class Line:
    '''
    Class representing line in polar coordinates.
    '''
    def __init__(self, rho, theta):
        self.rho = rho
        self.theta = theta

    def get_start_point(self):
        '''
        Return a "starting point" for linesegment.
        '''
        a = math.cos(self.theta)
        b = math.sin(self.theta)
        x0 = a * self.rho
        y0 = b * self.rho
        return (int(round(x0 + 10000 * (-b))), int(round(y0 + 10000 * a)))

    def get_end_point(self):
        '''
        Return a "starting point" for linesegment.
        '''
        a = math.cos(self.theta)
        b = math.sin(self.theta)
        x0 = a * self.rho
        y0 = b * self.rho
        return (int(round(x0 - 10000 * (-b))), int(round(y0 - 10000 * a)))


############################################################################################################
############################################################################################################


class IntersectionPoint:
    '''
    Class representing intersection point for two lines.
    '''
    def __init__(self, x, y, line1, line2):
        self.x = x
        self.y = y
        self.line1 = line1
        self.line2 = line2


############################################################################################################
############################################################################################################


class Linegroup:
    '''
    Class representing group of lines.
    '''
    def __init__(self):
        self.lines = []


############################################################################################################
############################################################################################################


class BoardProcessor:
    '''
    Class containing all functionality to process chessboard image.
    Main function is to process image and divide chessboard to 64 squares.
    '''

    def __init__(self, mat=None, debug=False):
        '''
        Constructor for BoardProcessor. Takes mat-image to be processed as input (optional).
        '''
        self.mat = mat
        self.processed = False
        self.failed = False
        self.error_msg = None
        self.squares = None
        self.debug = debug

    def reset(self):
        '''
        Reset BoardProcessor.
        '''
        self.mat = None
        self.processed = False
        self.error_msg = None
        self.failed = False
        self.squares = None

    def process(self):
        '''
        Try to process image.
        '''
        self.processed = True
        self.failed = False
        self.error_msg = None

        # convert mat-image to gray scale
        gray = convert_to_grayscale(self.mat)
        if self.debug:
            cv2.imwrite('images/debug/gray.jpg', gray)

        # blur the image
        blur = blur_image(gray)
        if self.debug:
            cv2.imwrite('images/debug/blur.jpg', blur)
        
        # detect edges from the image
        edges = auto_canny(blur)
        if self.debug:
            cv2.imwrite('images/debug/edges.jpg', edges)

        # detect lines from the image
        lines = detect_lines(edges)
        if self.debug:
            print("DEBUG: NUMBER OF LINES DETECTED WITH HOUGHLINES: " + str(len(lines)))
            tmp_mat = self.mat.copy()
            debug_image = draw_lines_to_mat(tmp_mat, lines)
            cv2.imwrite('images/debug/lines.jpg', debug_image)

        # divide detected lines to horizontal and vertical lines
        h_lines, v_lines = hor_vert_lines(lines)
        if self.debug:
            print("DEBUG: NUMBER OF HORIZONTAL LINES: " + str(len(h_lines)))
            print("DEBUG: NUMBER OF VERTICAL LINES: " + str(len(v_lines)))
        
        # merge similar lines together
        h_lines = merge_similar_lines(h_lines)
        if self.debug:
            print("DEBUG: NUMBER OF HORIZONTAL LINES AFTER MERGING: " + str(len(h_lines)))
            tmp_mat = self.mat.copy()
            debug_image = draw_lines_to_mat(tmp_mat, h_lines)
            cv2.imwrite('images/debug/hLines.jpg', debug_image)
        v_lines = merge_similar_lines(v_lines)
        if self.debug:
            print("DEBUG: NUMBER OF VERTICAL LINES AFTER MERGING: " + str(len(v_lines)))
            tmp_mat = self.mat.copy()
            debug_image = draw_lines_to_mat(tmp_mat, v_lines)
            cv2.imwrite('images/debug/vLines.jpg', debug_image)

        # check that enough lines were detected
        if len(h_lines) < 9 or len(v_lines) < 9:
            self.failed = True
            self.processed = True
            self.error_msg = "ERROR: TOO FEW LINES DETECTED"
            return

        # calculate intersection points for each line and vertical image divider line
        v_inter_points = vertical_intersection_points(h_lines, self.mat.shape[0], self.mat.shape[1])
        if self.debug:
            print("DEBUG: NUMBER OF INTERSECTION POINTS WITH VERTICAL DIVIDER: " + str(len(v_inter_points)))
        h_inter_points = horizontal_intersection_points(v_lines, self.mat.shape[0], self.mat.shape[1])
        if self.debug:
            print("DEBUG: NUMBER OF INTERSECTION POINTS WITH HORIZONTAL DIVIDER: " + str(len(h_inter_points)))
        
        # check that enough points were detected
        if len(v_inter_points) < 9 or len(h_inter_points) < 9:
            self.failed = True
            self.processed = True
            self.error_msg = "ERROR: TOO FEW INTERSECTION POINTS"
            return

        # group lines together by using the coordinates of the intersection points
        h_linegroups = horizontal_linegroups(v_inter_points)
        if self.debug:
            print("DEBUG: NUMBER OF HORIZONTAL LINEGROUPS FOUND: " + str(len(h_linegroups)))
        v_linegroups = vertical_linegroups(h_inter_points)
        if self.debug:
<<<<<<< HEAD
            print("DEBUG: NUMBER OF VERTICAL LINEGROUPS FOUND: " + str(len(v_linegroups)))

        # TODO: Remove lines that cross with lines from other linegroups
        
        # TODO: Remove outermost lines --> if threshold values in limits

        # TODO: Calculate intersection points for horizontal and vertical lines

        # TODO: Pick 4 corner points by using horizontal and vertical intersection points
=======
            print("NUMBER OF VERTICAL LINES AFTER MERGING: " + str(len(vLines)))
            tmpMat = self.mat.copy()
            debugImage = drawLinesToMat(tmpMat, vLines)
            cv2.imwrite('images/debug/vLines.jpg', debugImage)
        # remove lines that can not be actual grid lines (angle differs too much)
        hLines = removeWeirdLines(hLines)
        if self.debug:
            print("NUMBER OF HORIZONTAL LINES AFTER REMOVING WEIRD LINES: " + str(len(hLines)))
            tmpMat = self.mat.copy()
            debugImage = drawLinesToMat(tmpMat, hLines)
            cv2.imwrite('images/debug/hNoWeirdLines.jpg', debugImage)
        vLines = removeWeirdLines(vLines)
        if self.debug:
            print("NUMBER OF VERTICAL LINES AFTER REMOVING WEIRD LINES: " + str(len(vLines)))
            tmpMat = self.mat.copy()
            debugImage = drawLinesToMat(tmpMat, vLines)
            cv2.imwrite('images/debug/vNoWeirdLines.jpg', debugImage)
        # check that enough lines were detected
        if len(hLines) < 9 or len(vLines) < 9:
            self.failed = True
            self.errorMessage = "Too few lines!"
        
>>>>>>> 80b10f053b3b75cccdef33d044d71b21711bb119

        # TODO: do perspective transformation and finally save the image


############################################################################################################
############################################################################################################


''' FUNCTIONS TO PROCESS BOARD '''

def draw_lines_to_mat(mat, lines):
    '''
    Takes a mat-image and list of lines as input and draws the lines to image.
    '''
    for line in lines:
        rho = line.rho
        theta = line.theta
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

def convert_to_grayscale(mat):
    '''
    Convert given mat-image to grayscale. 
    '''
    return cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

def blur_image(mat):
    '''
    Blur given mat-image.
    '''
    return cv2.blur(mat, (10, 10))

def auto_canny(mat, sigma=0.33):
    '''
    Canny edge detection with automatic thresholds.
    '''
    # compute the median of the single channel pixel intensities
    v = np.median(mat)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(mat, lower, upper)

def detect_lines(mat):
    '''
    Detect lines from mat-image by using HoughLines.
    '''
    lines = cv2.HoughLines(mat, 1, np.pi/180, 200)
    lines = np.reshape(lines, (-1, 2))
    line_arr = []
    for distance, angle in lines:
        line_arr.append(Line(distance, angle))
    return line_arr

def adaptive_threshold(mat):
    '''
    Apply adaptive thresholding to mat-image.
    '''
    thres = cv2.adaptiveThreshold(mat,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return thres

def hor_vert_lines(lines):
    '''
    A line is given by rho and theta. Given a list of lines, returns a list of
    horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
    '''
    h = []
    v = []
    for line in lines:
        if line.theta < np.pi / 4 or line.theta > np.pi - np.pi / 4:
            v.append(line)
        else:
            h.append(line)
    return h, v

def merge_similar_lines(lines, distance_threshold = 30, angle_threshold = 0.1):
    '''
    Merge similar lines together. Similar lines are lines that are close to each other
    and their angle diffes less than threshold value. Merging means that one of the 
    lines will be dropped.
    '''
    results = []
    results.append(lines[0])
    for line1 in lines:
        is_new_line = True
        for line2 in results:
            if line1.rho == line2.rho and line2.theta == line2.theta: # same line
                is_new_line = False
                continue
            elif abs(line1.rho - line2.rho) <= distance_threshold and abs(line1.theta - line2.theta) <= angle_threshold: # similar line
                is_new_line = False
                continue
        if is_new_line:
            results.append(line1)
    return results

def vertical_intersection_points(lines, image_width, image_height):
    '''
    Takes a list of lines as input and calculates intersection point for each 
    line and vertical divider line. Returns sorted list of intersection points.
    '''
    points = []
    p1 = (image_width/2, 0)
    p2 = (image_width/2, image_height)
    for line in lines:
        p3 = line.get_start_point()
        p4 = line.get_end_point()
        a1 = p2[1] - p1[1]
        b1 = p1[0] - p2[0]
        c1 = a1 * p1[0] + b1 * p1[1]
        a2 = p4[1] - p3[1]
        b2 = p3[0] - p4[0]
        c2 = a2 * p3[0] + b2 * p3[1]
        det = a1 * b2 - a2 * b1

        if det != 0: # if determinant is not zero --> lines intersect
            x = (b2 * c1 - b1 * c2) / det
            y = (a1 * c2 - a2 * c1) / det
            points.append(IntersectionPoint(x, y, line, None))
    points.sort(key = lambda point: point.y)
    return points

def horizontal_intersection_points(lines, image_width, image_height):
    '''
    Takes a list of lines as input and calculates intersection point for each 
    line and horizontal divider line. Returns sorted list of intersection points.
    '''
    points = []
    p1 = (0, image_height/2)
    p2 = (image_width, image_height/2)
    for line in lines:
<<<<<<< HEAD
        p3 = line.get_start_point()
        p4 = line.get_end_point()
        a1 = p2[1] - p1[1]
        b1 = p1[0] - p2[0]
        c1 = a1 * p1[0] + b1 * p1[1]
        a2 = p4[1] - p3[1]
        b2 = p3[0] - p4[0]
        c2 = a2 * p3[0] + b2 * p3[1]
        det = a1 * b2 - a2 * b1

        if det != 0: # if determinant is not zero --> lines intersect
            x = (b2 * c1 - b1 * c2) / det
            y = (a1 * c2 - a2 * c1) / det
            points.append(IntersectionPoint(x, y, line, None))
    points.sort(key = lambda point: point.x)
    return points

def horizontal_linegroups(points):
    '''
    Try to group lines by analyzing vertical intersection points. Lines that have
    intersection points close to each other are assumed to belong to same linegroup.
    '''
    # calculate differences in y-coordinates
    distances = []
    for i in range(len(points) - 1):
        point1 = points[i]
        point2 = points[i+1]
        d = abs(point1.y - point2.y)        
        distances.append(d)
    # Pick 5 largest values and check if they are similar.
    top5 = heapq.nlargest(5, distances)
    # It is possible that there is couple larger values that are caused by missing lines
    # but ones there are similar values we can assume that they are chessboard grid widths
    threshold = 0
    if abs(top5[0]- top5[4]) < top5[2] / 3:
        threshold = statistics.mean(top5) / 3
    elif abs(top5[1]- top5[4]) < top5[3] / 3:
        threshold = statistics.mean(top5[1:]) / 3
    elif abs(top5[2]- top5[4]) < top5[3] / 3:
        threshold = statistics.mean(top5[2:]) / 3
    elif abs(top5[3]- top5[4]) < top5[4] / 3:
        threshold = statistics.mean(top5[3:]) / 3
    else:
        threshold = top5[4] / 2
    # group lines by using the "sophisticated" threshold value
    linegroups = []
    lg = Linegroup()
    for i in range(len(points) - 1):
        lg.lines.append(point1.line1)
        point1 = points[i]
        point2 = points[i+1]
        d = abs(point1.y - point2.y)
        if d > threshold:   # distance between points too big
            linegroups.append(lg)
            lg = Linegroup()    # create new linegroup
        if i == len(points) - 2:
            lg.lines.append(point2.line1)
            linegroups.append(lg)
    return linegroups

def vertical_linegroups(points):
    '''
    Try to group lines by analyzing horizontal intersection points. Lines that have
    intersection points close to each other are assumed to belong to same linegroup.
    '''
    # calculate differences in y-coordinates
    distances = []
    for i in range(len(points) - 1):
        point1 = points[i]
        point2 = points[i+1]
        d = abs(point1.x - point2.x)        
        distances.append(d)
    # Pick 5 largest values and check if they are similar.
    top5 = heapq.nlargest(5, distances)
    # It is possible that there is couple larger values that are caused by missing lines
    # but ones there are similar values we can assume that they are chessboard grid widths
    threshold = 0
    if abs(top5[0]- top5[4]) < top5[2] / 3:
        threshold = statistics.mean(top5) / 3
    elif abs(top5[1]- top5[4]) < top5[3] / 3:
        threshold = statistics.mean(top5[1:]) / 3
    elif abs(top5[2]- top5[4]) < top5[3] / 3:
        threshold = statistics.mean(top5[2:]) / 3
    elif abs(top5[3]- top5[4]) < top5[4] / 3:
        threshold = statistics.mean(top5[3:]) / 3
    else:
        threshold = top5[4] / 2
    # group lines by using the "sophisticated" threshold value
    linegroups = []
    lg = Linegroup()
    for i in range(len(points) - 1):
        lg.lines.append(point1.line1)
        point1 = points[i]
        point2 = points[i+1]
        d = abs(point1.x - point2.x)
        if d > threshold:   # distance between points too big
            linegroups.append(lg)
            lg = Linegroup()    # create new linegroup
        if i == len(points) - 2:
            lg.lines.append(point2.line1)
            linegroups.append(lg)
    return linegroups  

def remove_crossing_linegroup_lines(linegroups, image_width, image_height):
    '''
    Check that lines in linegroups do not intersect other with other linegroups lines
    inside the image borders. If lines intersect, the one with bigger difference from
    it groups average theta will be removed.
    '''
    pass
=======
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

def removeWeirdLines(lines, delta=math.pi/8):
    '''
    Takes a list of lines as input and calculates the median angle.
    Removes lines that differ too much from median value.
    '''
    # calculate median angle of lines
    angles = []
    for distance, angle in lines:
        angles.append(angle)
    medianAngle = statistics.median(angles)
    # remove lines that differ too much
    resultList = []
    for line in lines:
        distance, angle = line
        if abs(angle-medianAngle) <= delta:
            resultList.append(line)
    return resultList
    
>>>>>>> 80b10f053b3b75cccdef33d044d71b21711bb119
