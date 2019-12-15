# Shakkipalvelin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Mobiilishakki/Shakkipalvelin/blob/master/LICENSE)
[![Docker Pulls](https://img.shields.io/docker/pulls/mshakki/shakkipalvelin.svg)](https://hub.docker.com/r/mshakki/shakkipalvelin)

Server for recognizing chess pieces using computer vision

## Credits

Credits for ideas, inspiration and code samples go to, among others:

The flask server code is partly based on https://github.com/daylen/chess-id (MIT Licence: Copyright (c) 2015 Daylen Yang)

The code for automatic board detection in the **autodetect** subdirectory is from https://github.com/maciejczyzewski/neural-chessboard (MIT Licence: Copyright (c) 2017-present Maciej A. Czyzewski and other contributors) 

Nvidia CUDA Dockerfile is from https://github.com/anibali/docker-pytorch (MIT Licence: Copyright (c) 2016 Aiden Nibali)

## Production usage

There are some docker images to be found at our [DockerHub repository](https://hub.docker.com/r/mshakki/shakkipalvelin).
For the most reliable experience, use the tag "release-0.1.0" together with the [web app](https://hub.docker.com/r/mshakki/webapp) tag "release-0.0.1".

To start a production container (CPU version) running on port 80:

```sh
docker run -p 80:5000 --rm -v /home/shakki/models/pytorch_chessmodel_1.pth:/App/pytorch_chessmodel_1.pth mshakki/shakkipalvelin:release-0.2.0
```

This command mounts the machine learning model from the server's home directory into the container (the -v option). Remember to include the model file! 

To start web app, write:

```sh
docker run -p 80:3000 mshakki/webapp:release-0.0.1
```

__Important note:__ at least when using the UFW firewall on Linux, Docker automatically changes firewall rules to allow incoming traffic on the port defined above. When the container stops, the firewall rule is reset. This is usually practical and even desirable, but is good to know.

## Recognition

You can try out the container's recognition abilities by sending it a JPG image of a chessboard:

```sh
curl -F 'file=@./example.jpg' http://[url-or-IP-to-server]/upload
```

An example image is included in the repository files. The output for the example file is close to perfect
for our prototype model:

```sh
rnbqkbnr/pppppppp/r7/8/8/8/PPPPPPPP/RNBQKBNR
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
