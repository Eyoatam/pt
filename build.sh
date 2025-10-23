#!/bin/zsh

set -e

if [[ -z "$1" ]]; then
    src_file=("main.cpp")
else
    src_file=("$1")
fi

src_base=("${src_file:t}")
exe_base=("${src_base%.*}") 

mkdir -p build
pushd build > /dev/null

if command -v clang >/dev/null 2>&1; then
  clang++ -g -Wall "../$src_file" -o "${exe_base}_d"
  clang++ -O3 -g -Wall "../$src_file" -o "${exe_base}_r"
fi

popd > /dev/null
