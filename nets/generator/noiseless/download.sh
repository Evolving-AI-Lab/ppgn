#!/bin/bash

f=generator.caffemodel

if [ ! -f "${f}" ]; then 
  echo "Downloading ..."
  wget http://s.anhnguyen.me/181007__generator.caffemodel -O ${f}
fi

ls ${f}
echo "Done."
