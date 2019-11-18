#!/usr/bin/python

import os
import cv2
import hashlib
import numpy as np
from flask import Flask, request, redirect, url_for

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


PATH = os.environ.get('HOME')
PATH += "/chess/"

CATEGORIES = ["black_pawns", "black_knights", "black_rooks", "black_bishops", "black_queens", "black_kings", "white_pawns", "white_knights", "white_rooks", "white_bishops", "white_queens", "white_kings", "empty_tiles"]

def write_data(squares, image_hash):
    
    cv2.imwrite(PATH + CATEGORIES[2] + "/" + str(image_hash) + ".jpg", squares[0])
    cv2.imwrite(PATH + CATEGORIES[1] + "/" + str(image_hash) + ".jpg", squares[1])
    cv2.imwrite(PATH + CATEGORIES[3] + "/" + str(image_hash) + ".jpg", squares[2])
    cv2.imwrite(PATH + CATEGORIES[4] + "/" + str(image_hash) + ".jpg", squares[3])
    cv2.imwrite(PATH + CATEGORIES[5] + "/" + str(image_hash) + ".jpg", squares[4])
    cv2.imwrite(PATH + CATEGORIES[0] + "/" + str(image_hash) + ".jpg", squares[9])
    cv2.imwrite(PATH + CATEGORIES[8] + "/" + str(image_hash) + ".jpg", squares[63])
    cv2.imwrite(PATH + CATEGORIES[7] + "/" + str(image_hash) + ".jpg", squares[62])
    cv2.imwrite(PATH + CATEGORIES[9] + "/" + str(image_hash) + ".jpg", squares[61])
    cv2.imwrite(PATH + CATEGORIES[11] + "/" + str(image_hash) + ".jpg", squares[60])
    cv2.imwrite(PATH + CATEGORIES[10] + "/" + str(image_hash) + ".jpg", squares[59])
    cv2.imwrite(PATH + CATEGORIES[6] + "/" + str(image_hash) + ".jpg", squares[54])
    cv2.imwrite(PATH + CATEGORIES[12] + "/" + str(image_hash) + "_white.jpg", squares[25])
    cv2.imwrite(PATH + CATEGORIES[12] + "/" + str(image_hash) + "_black.jpg", squares[26])

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Chess ID. usage: /upload'

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file = file.read()
            image_hash = hashlib.sha512(file).hexdigest()
            img = np.asarray(bytearray(file))
            img = cv2.imdecode(img, 1)
            squares = split_board(img)
            write_data(squares, image_hash)            

            return "OK"
    return '''
    <!doctype html>
    <title>Chess ID</title>
    <h1>Upload board picture</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    for cat in CATEGORIES:
        if not os.path.exists(PATH + cat):
            os.makedirs(PATH + cat)
    app.run(host='0.0.0.0', debug=False)

