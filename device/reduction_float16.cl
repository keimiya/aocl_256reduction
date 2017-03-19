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
 * @param output_ddr 
 */
__kernel void reduction_float16(__global volatile const float16* restrict input_ddr,
				__global  float* restrict output_ddr)
{
    const int REDUCTION_SIZE = 256;
    const int INPUT_WIDTH    = 16; // equal to float16
    const int REDUCTION_ITERATION = REDUCTION_SIZE/INPUT_WIDTH;
    float sum = 0.0e0;

	float16 input_temp;
    float level2[INPUT_WIDTH>>1];
    float level3[INPUT_WIDTH>>2];
    float level4[INPUT_WIDTH>>3];
    float level5[REDUCTION_ITERATION];

for(int j = 0; j < 10000; j++){
    // Each level of reduction takes 3[cycle] since all the adds are run in parallel
    for (int i = 0; i < REDUCTION_ITERATION; i++){
	// read the input value from DDR
	input_temp = input_ddr[i];

	// 1st level: 
	level2[0] = input_temp.s0 + input_temp.s1; // fp32 add takes 3[cycle]
	level2[1] = input_temp.s2 + input_temp.s3;
	level2[2] = input_temp.s4 + input_temp.s5;
	level2[3] = input_temp.s6 + input_temp.s7;
	level2[4] = input_temp.s8 + input_temp.s9;
	level2[5] = input_temp.sA + input_temp.sB;
	level2[6] = input_temp.sC + input_temp.sD;
	level2[7] = input_temp.sE + input_temp.sF;

	// 2nd level
	level3[0] = level2[0] + level2[1];
	level3[1] = level2[2] + level2[3];
	level3[2] = level2[4] + level2[5];
	level3[3] = level2[6] + level2[7];

	// 3rd level
	level4[0] = level3[0] + level3[1];
	level4[1] = level3[2] + level3[3];

	// 4th level
	level5[i + 0] = level4[0] + level4[1];
    }

#pragma unroll
    for (int i = 0; i < REDUCTION_ITERATION; i++)
	sum += level5[i];

    *output_ddr = sum;
}
}

