/**
 * @file reduction_float16.cl
 * @brief Device kernel code for 256-elements reduction written in Altera OpenCL.
 * @par   Input width is float16(32bit*16=512bit=64byte)
 * @date  2017/03/14 Copy noda-chan's reduction_single_1k
 * @date  2017/03/15 Start implementing float16 version
 * @ToDo  The number of DDR4's bank is 2.
 * @author T.Miyajima
 */

/**
 * AOCL device code
 * @param input_ddr float16 (32bit*16=512bit=64byte) from DDR4-2133. It's non Stall-free, and what's Stall-free?
 * @param num_of_cell 270*310=83,700 cells
 * @param output_ddr 
 */
__kernel void reduction_float16(__global const float16* restrict input_ddr,
				unsigned int cells, // 83,700 cells
				__global float* restrict output_ddr)
{
    const int REDUCTION_SIZE	  = 256;
    const int INPUT_WIDTH	  = 16;	// equal to float16
    const int REDUCTION_ITERATION = REDUCTION_SIZE/INPUT_WIDTH; // 16 

    float level2[REDUCTION_SIZE>>1]; // 128
    float level3[REDUCTION_SIZE>>2]; // 64
    float level4[REDUCTION_SIZE>>3]; // 32
    float level5[REDUCTION_SIZE>>4]; // 16
    float sum_temp;

    //#pragma unroll 1
    for (int i = 0; i < cells*REDUCTION_ITERATION; i++){
	// 1st level: 
	level2[0] = input_ddr[i].s0 + input_ddr[i].s1;
	level2[1] = input_ddr[i].s2 + input_ddr[i].s3;
	level2[2] = input_ddr[i].s4 + input_ddr[i].s5;
	level2[3] = input_ddr[i].s6 + input_ddr[i].s7;
	level2[4] = input_ddr[i].s8 + input_ddr[i].s9;
	level2[5] = input_ddr[i].sA + input_ddr[i].sB;
	level2[6] = input_ddr[i].sC + input_ddr[i].sD;
	level2[7] = input_ddr[i].sE + input_ddr[i].sF;
	// 2nd level
	level3[0] = level2[0] + level2[1];
	level3[1] = level2[2] + level2[3];
	level3[2] = level2[4] + level2[5];
	level3[3] = level2[6] + level2[7];
	// 3rd level
	level4[0] = level3[0] + level3[1];
	level4[1] = level3[2] + level3[3];
	// 4th level
	level5[i % REDUCTION_ITERATION] = level4[0] + level4[1];

	if((i % INPUT_WIDTH) == 0){
#pragma unroll
	    for (int j = 0; j < REDUCTION_ITERATION; j++)
		sum_temp += level5[j];

	    *output_ddr = sum_temp;
	    sum_temp = 0;
	}
    }
}
