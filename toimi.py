import sys
import cv2
import numpy as np

from processor import BoardProcessor

# load input image
url = 'images/shakki5.jpg'
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
