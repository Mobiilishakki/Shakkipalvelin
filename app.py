from PIL import Image
import sys, math, torch
from torchvision import transforms
from flask import Flask, request, redirect, url_for, jsonify

""" CONSTANTS """

CATEGORIES = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])
CURRENT_STATE = 'None'
POLL = True

""" LOADING """

# Load pre-trained ML model for inference
model_file = 'pytorch_chessmodel_1.pth'
# Currently only CPU is available on server
device = torch.device('cpu')
model = torch.load(model_file)
model.eval()

# Prepare image transformation parameters
test_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

""" LOGIC """

def predict_image(img):
    '''Take PIL image as input and return the
    pytorch NN classification.'''
    # Transform to standardize image sizes etc.
    img_tensor = test_transforms(img).float()
    img_tensor = img_tensor.unsqueeze_(0)
    input = img_tensor.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    '''Categories are in the same order both in the model
    and the list so they can be found directly by index'''
    return CATEGORIES[index]

def split_board(img):
    '''Split original image into 64 tiles by the
    crop() function of PIL.'''
    tiles = []
    sq_len = math.floor(min(img.size[0], img.size[1]) / 8)
    for j in range(8):
        for i in range(8):
            x = i * sq_len
            y = j * sq_len
            tile = img.crop((x, y, x + sq_len, y + sq_len))
            tiles.append(tile)
    return tiles

def shrink_blanks(fen):
    '''Count consecutive blanks and replaces
    by their number.'''
    if '_' not in fen:
        return fen
    new_fen = ''
    blanks = 0
    for char in fen:
        if char == '_':
            blanks += 1
        else:
            if blanks != 0:
                new_fen += str(blanks)
                blanks = 0
            new_fen += char
    if blanks != 0:
        new_fen += str(blanks)
    return new_fen

def get_fen(result):
    '''Transform list of results into standard
    FEN notation.'''
    fen = ''
    for sq in result:
        if sq == 'empty':
            fen += '_'
        elif sq[0] == 'b':
            fen += sq[1]
        else:
            fen += str(sq[1]).upper()
    fens = [fen[i:i+8] for i in range(0, 64, 8)]
    fens = map(shrink_blanks, fens)
    fen = '/'.join(fens)
    return fen

""" ROUTES """

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Chess ID. usage: /upload'

@app.route('/state')
def current_state():
    return jsonify({'state': CURRENT_STATE})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def get_poll():
    global POLL
    return POLL

def set_poll(value):
    global POLL
    POLL = value

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        set_poll(False)
        file = request.files['file']
        if file and allowed_file(file.filename):
            img = Image.open(file)
            squares = split_board(img)
            result = []
            for square in squares:
                result.append(predict_image(square))
            global CURRENT_STATE
            CURRENT_STATE = get_fen(result)
            return CURRENT_STATE
    return '''
    <!doctype html>
    <title>Chess ID</title>
    <h1>Upload board picture</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/snapshot', methods=['GET', 'POST'])
def snapshot():
    if request.method == 'POST':
       set_poll(True)
       return "True"
    
    if request.method == 'GET':
        if get_poll() == True:
            set_poll(False)
            return "True"
        return "False"


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=80, debug=False)
    app.run(host='0.0.0.0', debug=False)
