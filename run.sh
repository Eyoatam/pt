#!/bin/zsh

buildtype=""
out=""

if [[ -z "$1" ]]; then
    buildtype="r"
else 
    buildtype="$1" 
fi

if [[ -z "$2" ]]; then
    out="./out/test.ppm"
else
    out="./out/$2.ppm"
fi

if [[ -z "$3" ]]; then
    out_file="main"
else
    out_file="$3"
fi

./build/${out_file}_${buildtype} > $out
