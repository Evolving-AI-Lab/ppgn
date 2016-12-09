#!/bin/bash

f=bvlc_reference_caffenet.caffemodel

if [ ! -f "${f}" ]; then 
  echo "Downloading..."
  wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
fi

echo "Done."
