#!/bin/bash

f=lrcn_caffenet_iter_110000.caffemodel

if [ ! -f "${f}" ]; then 
  echo "Downloading ..."
  wget http://www.cs.uwyo.edu/~anguyen8/share/${f}
fi

ls ${f}
echo "Done."
