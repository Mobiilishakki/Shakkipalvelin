# Shakkipalvelin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Mobiilishakki/Shakkipalvelin/blob/master/LICENSE)

Server for recognizing chess pieces using computer vision

## Production usage

To start a production server running on port 80:

```sh
docker run -p 80:5000 mshakki/shakkipalvelin:0.1-fedora-31
```

## Development

Use the following to clone and run a development version of the server:

```sh
git clone git@github.com:Mobiilishakki/Shakkipalvelin.git
git checkout dev
```

### Installing dependencies

A recent (as of 2019) version of python 3 is required for running this server. Start by cloning this repository and changing to its root directory.

Create python virtual environment:

```sh
python3 -m venv venv
```

Activate virtual environment:
```sh
. venv/bin/activate
```

Install dependencies:
```sh
pip install -r requirements.txt
```


The code for automatic board detection in the **autodetect** subdirectory is from [neural-chessboard](https://github.com/maciejczyzewski/neural-chessboard).
