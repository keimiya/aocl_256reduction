#!/bin/bash
TARGET=reduction_float16_cell_unroll
find ./device -name "${TARGET}*" | awk -F"/" '{gsub(".cl",""); print $3}' | xargs -P 12 -I% -t bash -c 'nohup ./aocx_go % &'
