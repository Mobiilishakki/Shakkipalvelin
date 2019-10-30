import sys
import cv2
import numpy as np
import torch

from processor import BoardProcessor
from model import KitModel

# load input image
url = 'shakki1.jpg'
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

# prepare the neural network
MODEL_PATH = './model.pth'
net = KitModel(MODEL_PATH)
checkpoint = torch.load(MODEL_PATH)
try:
    checkpoint.eval()
except AttributeError as error:
    print(error)

net.load_state_dict(checkpoint)
net.eval()

# feed squares to the neural network and get the results
outputs = net(processor.squares)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
