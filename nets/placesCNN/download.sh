#!/bin/bash
f=placesCNN.tar.gz

if [ ! -f ${f} ]; then
  echo "Downloading..."
  wget http://places.csail.mit.edu/model/${f}
fi

echo "Extracting ${f}..."
tmp="tmp"
mkdir -p ${tmp}
tar xvf placesCNN.tar.gz -C ${tmp}/
mv ${tmp}/places205CNN_iter_300000.caffemodel ./
rm -rf ${tmp}

echo "Done."
