import cv2
import numpy as np
import math
import statistics
import heapq

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

    def get_edited_theta(self):
        if self.theta > math.pi / 4 * 3:
            return self.theta - math.pi
        return self.theta


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

        # print image information for debugging
        if self.debug:
            print("DEBUG: IMAGE WIDTH: " + str(self.mat.shape[1]))
            print("DEBUG: IMAGE HEIGHT: " + str(self.mat.shape[0]))

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
        v_inter_points = vertical_intersection_points(h_lines, self.mat.shape[1], self.mat.shape[0])
        if self.debug:
            print("DEBUG: NUMBER OF INTERSECTION POINTS WITH VERTICAL DIVIDER: " + str(len(v_inter_points)))
        h_inter_points = horizontal_intersection_points(v_lines, self.mat.shape[1], self.mat.shape[0])
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
            print("DEBUG: NUMBER OF VERTICAL LINEGROUPS FOUND: " + str(len(v_linegroups)))

        # Remove lines that cross with lines from other linegroups
        h_linegroups = remove_crossing_linegroup_lines(h_linegroups, self.mat.shape[1], self.mat.shape[0])
        if self.debug:
            print("DEBUG: REMOVED CROSSING LINES FROM HORIZONTAL LINEGROUPS")
            tmp_mat = self.mat.copy()
            debug_image = draw_lines_to_mat(tmp_mat, lines_from_linegroups(h_linegroups))
            cv2.imwrite('images/debug/hLinesCrossing.jpg', debug_image)
        v_linegroups = remove_crossing_linegroup_lines(v_linegroups, self.mat.shape[1], self.mat.shape[0])
        if self.debug:
            print("DEBUG: REMOVED CROSSING LINES FROM VERTICAL LINEGROUPS")
            tmp_mat = self.mat.copy()
            debug_image = draw_lines_to_mat(tmp_mat, lines_from_linegroups(v_linegroups))
            cv2.imwrite('images/debug/vLinesCrossing.jpg', debug_image)

        # Remove outermost lines --> if threshold values in limits
        h_linegroups = remove_outermost_lines(h_linegroups)
        if self.debug:
            print("DEBUG: REMOVED OUTERMOST LINES FROM HORIZONTAL LINES")
            tmp_mat = self.mat.copy()
            debug_image = draw_lines_to_mat(tmp_mat, lines_from_linegroups(h_linegroups))
            cv2.imwrite('images/debug/hLinesOuter.jpg', debug_image)
        v_linegroups = remove_outermost_lines(v_linegroups)
        if self.debug:
            print("DEBUG: REMOVED OUTERMOST LINES FROM VERTICAL LINES")
            tmp_mat = self.mat.copy()
            debug_image = draw_lines_to_mat(tmp_mat, lines_from_linegroups(v_linegroups))
            cv2.imwrite('images/debug/vLinesOuter.jpg', debug_image)

        # Calculate intersection points for horizontal and vertical lines
        h_lines = lines_from_linegroups(h_linegroups)
        v_lines = lines_from_linegroups(v_linegroups)
        intersection_points = hor_vert_intersection_points(h_lines, v_lines, self.mat.shape[1], self.mat.shape[0])
        if self.debug:
            print("DEBUG: NUMBER OF INTERSECTION POINTS BETWEEN HORIZONTAL AND VERTICAL LINES: " + str(len(intersection_points)))

        # Pick 4 corner points by using horizontal and vertical intersection points
        corner_points = four_corner_points(intersection_points, h_linegroups, v_linegroups)
        if self.debug:
            print("DEBUG: NUMBER OF CORNER POINTS FOUND: " + str(len(corner_points)))
            for point in corner_points:
                print(point)
        
        # Do perspective transformation and finally save the image
        self.mat = four_point_transform(self.mat, corner_points)
        if self.debug:
            print("DEBUG: FOUR CORNER TRANSFORM COMPLETED")
            tmp_mat = self.mat.copy()
            cv2.imwrite('images/debug/perspective.jpg', tmp_mat)
        
        # Divide board to parts
        self.squares = split_board(self.mat)
        if self.debug:
            print("DEBUG: BOARD SPLITTED TO 64 PARTS")


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
    #h = []
    #v = []
    #for line in lines:
    #    if line.theta < np.pi / 4 or line.theta > np.pi - np.pi / 4:
    #        v.append(line)
    #    else:
    #        h.append(line)
    h = []
    v = []
    for line in lines:
        if len(h) == 0 and len(v) == 0:
            h.append(line)
        elif abs(line.theta - h[0].theta) < math.pi/4 or abs(line.theta - math.pi - h[0].theta) < math.pi/4:
            h.append(line)
        else:
            v.append(line)
    if h[0].theta < np.pi / 4 or h[0].theta > np.pi - np.pi / 4:
        return v, h
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
    for i in range(len(points)-1):
        point1 = points[i]
        point2 = points[i+1]
        lg.lines.append(point1.line1)
        d = abs(point1.y - point2.y)
        if d > threshold:   # distance between points too big
            linegroups.append(lg)
            lg = Linegroup()    # create new linegroup
        elif i == len(points) - 2:
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
        point1 = points[i]
        point2 = points[i+1]
        lg.lines.append(point1.line1)
        d = abs(point1.x - point2.x)
        if d > threshold:   # distance between points too big
            linegroups.append(lg)
            lg = Linegroup()    # create new linegroup
        elif i == len(points) - 2:
            lg.lines.append(point2.line1)
            linegroups.append(lg)
    return linegroups  

def remove_crossing_linegroup_lines(linegroups, image_width, image_height):
    '''
    Check that lines in linegroups do not intersect other with other linegroups lines
    inside the image borders. If lines intersect, the one with bigger difference from
    it groups average theta will be removed.
    '''
    for lg1 in linegroups:
        if len(lg1.lines) == 0: # no lines in linegroup --> some kind of error is possible
            continue
        intersections_count = [0] * len(lg1.lines) # array to keep count of intersections for each line
        for lg2 in linegroups:
            if lg1 == lg2 or len(lg2.lines) == 0: # group itself or error?
                continue
            for i in range(len(lg1.lines)):
                l1 = lg1.lines[i]
                for l2 in lg2.lines:
                    ipoint = intersection_point(l1, l2)
                    if(ipoint != None and ipoint.x >= 0 and ipoint.y >= 0 and ipoint.x <= image_width and ipoint.y <= image_height):
                        intersections_count[i] += 1
                        break
        # remove lines with more than 1 intersection
        tmp = []
        for i in range(len(intersections_count)):
            if intersections_count[i] <= 1:
                tmp.append(lg1.lines[i])
        lg1.lines = tmp
    return linegroups 

def intersection_point(line1, line2):
    '''
    Calculate intersection point for two lines.
    '''
    p1 = line1.get_start_point()
    p2 = line1.get_end_point()
    p3 = line2.get_start_point()
    p4 = line2.get_end_point()
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
        return IntersectionPoint(x, y, line1, line2)
    
    # no intersection point
    return None

def remove_outermost_lines(linegroups):
    '''
    Try to remove outermost lines. Chessboard might have other lines besides the grid lines,
    so this function tries to remove those lines. This function only operates for the
    first and last linegroup in array.
    '''
    # pick first and last linegroups (they should be ordered so it's fine)
    lg_first = linegroups[0]
    lg_last = linegroups[len(linegroups) - 1]

    lg_first.lines = [lg_first.lines[-1]]
    lg_last.lines = [lg_last.lines[0]]

    linegroups[0] = lg_first
    linegroups[len(linegroups) - 1] = lg_last

    return linegroups

def lines_from_linegroups(linegroups):
    '''
    Takes a list of linegroups as input and returns list of lines.
    '''
    lines = []
    for lg in linegroups:
        lines = lines + lg.lines 
    return lines

def hor_vert_intersection_points(horizontal_lines, vertical_lines, image_width, image_height):
    '''
    Calculate all intersection points for horizontal lines and vertical lines.
    Intersection points are within image borders.
    '''
    print
    points = []
    for line1 in horizontal_lines:
        for line2 in vertical_lines:
            point = intersection_point(line1, line2)
            if point != None and point.x >= 0 and point.y >= 0 and point.x <= image_width and point.y <= image_height:
                points.append(point)
    return points

def four_corner_points(intersection_points, h_linegroups, v_linegroups):
    '''
    Pick four chessboard cornerpoints.
    '''
    # pick boarder lines
    h1 = h_linegroups[0].lines[0]
    h2 = h_linegroups[-1].lines[0]
    v1 = v_linegroups[0].lines[0]
    v2 = v_linegroups[-1].lines[0]

    # corner points must be picked in following order:
    # top left, top right, bottom left, bottom right

    points = [0, 0, 0, 0]

    # find intersection points
    for point in intersection_points:
        if (point.line1 in (h1, v1) and point.line2 in (h1, v1)):
            points[0] = (point.x, point.y)
        elif (point.line1 in (h1, v2) and point.line2 in (h1, v2)):
            points[1] = (point.x, point.y)
        elif (point.line1 in (h2, v1) and point.line2 in (h2, v1)):
            points[2] = (point.x, point.y)
        elif (point.line1 in (h2, v2) and point.line2 in (h2, v2)):
            points[3] = (point.x, point.y)

    return points

def four_point_transform(img, points, square_length=1816):
    '''
    Do perspective transform for image.
    '''

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [square_length, 0], [0, square_length], [square_length, square_length]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (square_length, square_length))

def split_board(img):
    """
    Given a board image, returns an array of 64 smaller images.
    """
    arr = []
    sq_len = int(img.shape[0] / 8)
    for i in range(8):
        for j in range(8):
            arr.append(img[i * sq_len : (i + 1) * sq_len, j * sq_len : (j + 1) * sq_len])
    return arr