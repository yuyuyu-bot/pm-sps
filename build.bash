#!/bin/bash

set -uex

export PATH=~/git/libSGM:~/git/libSGM/build/src:${PATH}

# rm build -rf
mkdir -p build
cd build
cmake ../ -DCMAKE_MODULE_PATH=~/git/libSGM
make
cd -
