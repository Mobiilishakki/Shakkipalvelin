
import cv2
import numpy as np

from server import auto_canny, hor_vert_lines, intersections, cluster, find_corners, four_point_transform, split_board

url = 'shakki1.jpg'

with open(url, 'rb') as file:
    test = np.asarray(bytearray(file.read()))


img = cv2.imdecode(test, 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (10, 10))

edges = auto_canny(gray)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

lines = np.reshape(lines, (-1, 2))

h, v = hor_vert_lines(lines)
   
points = intersections(h, v)

points = cluster(points)

img_shape = np.shape(img)
points = find_corners(points, (img_shape[1], img_shape[0]))
    
new_img = four_point_transform(img, points)

cv2.imwrite('new_chess1.jpg', new_img)

arr = split_board(new_img)
cv2.imwrite('part.jpg', arr[0])

