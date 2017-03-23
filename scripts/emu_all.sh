#!/bin/bash
function make_alias_and_run_emu () {
TARGET=$1
rm ./device/reduction_float16.cl; ln -s ./${TARGET}.cl ./device/reduction_float16.cl;
echo ${TARGET}
./emu_go
}

make_alias_and_run_emu reduction_float16_cell_no-unroll
#make_alias_and_run_emu reduction_float16_cell_unroll2
#make_alias_and_run_emu reduction_float16_cell_unroll3
#make_alias_and_run_emu reduction_float16_cell_unroll4
#make_alias_and_run_emu reduction_float16_cell_unroll5
#make_alias_and_run_emu reduction_float16_cell_unroll6
