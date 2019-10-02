
import cv2
import numpy as np

from server import auto_canny, hor_vert_lines, intersections, cluster, find_corners, four_point_transform, split_board

url = 'shakki1.jpg'

with open(url, 'rb') as file:
    test = np.asarray(bytearray(file.read()))


img = cv2.imdecode(test, 1)


# Test showing the image

#cv2.imshow('img', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (10, 10))

edges = auto_canny(gray)

print(np.count_nonzero(edges) / float(gray.shape[0] * gray.shape[1]))

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

lines = np.reshape(lines, (-1, 2))

h, v = hor_vert_lines(lines)
   
print('h: {} | v: {}'.format(h, v))

points = intersections(h, v)

points = cluster(points)

img_shape = np.shape(img)
points = find_corners(points, (img_shape[1], img_shape[0]))
    
new_img = four_point_transform(img, points)

cv2.imwrite('new_chess1.jpg', new_img)

arr = split_board(new_img)
cv2.imwrite('part.jpg', arr[0])

#cv2.imshow('test', new_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
