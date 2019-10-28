import sys
import cv2
import numpy as np

from server import auto_canny, hor_vert_lines, intersections, cluster, find_corners, four_point_transform, split_board, find_board
from processor import BoardProcessor

# load input image
url = 'images/shakki1.jpg'
with open(url, 'rb') as file:
    test = np.asarray(bytearray(file.read()))

# create mat-image
img = cv2.imdecode(test, 1)

# create BoardProcessor and process the image
processor = BoardProcessor(img, debug=True)
processor.process()

# check if errors
if processor.processed and processor.failed:
    print(processor.error_msg)
else:
    cv2.imwrite('images/debug/part0.jpg', processor.squares[0])
