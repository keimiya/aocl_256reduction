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
    float sum = 0.0e0;

    float8 level2[REDUCTION_SIZE/INPUT_WIDTH];
    float4 level3[REDUCTION_SIZE/INPUT_WIDTH];
    float2 level4[REDUCTION_SIZE/INPUT_WIDTH];
    float level5[REDUCTION_SIZE/INPUT_WIDTH];

    // Each level of reduction takes 3[cycle] since all the adds are run in parallel
    for (int i = 0; i < REDUCTION_SIZE/INPUT_WIDTH; i++){

	// 1st level: directory read the input value from DDR
	level2[i].s0 = input_ddr[i].s0 + input_ddr[i].s1; // fp32 add takes 3[cycle]
	level2[i].s1 = input_ddr[i].s2 + input_ddr[i].s3;
	level2[i].s2 = input_ddr[i].s4 + input_ddr[i].s5;
	level2[i].s3 = input_ddr[i].s6 + input_ddr[i].s7;
	level2[i].s4 = input_ddr[i].s8 + input_ddr[i].s9;
	level2[i].s5 = input_ddr[i].sA + input_ddr[i].sB;
	level2[i].s6 = input_ddr[i].sC + input_ddr[i].sD;
	level2[i].s7 = input_ddr[i].sE + input_ddr[i].sF;

	// 2nd level
	level3[i].s0 = level2[i].s0 + level2[i].s1;
	level3[i].s1 = level2[i].s2 + level2[i].s3;
	level3[i].s2 = level2[i].s4 + level2[i].s5;
	level3[i].s3 = level2[i].s6 + level2[i].s7;

	// 3rd level
	level4[i].s0 = level3[i].s0 + level3[i].s1;
	level4[i].s1 = level3[i].s2 + level3[i].s3;

	// 4th level
	level5[i] = level4[i].s0 + level4[i].s1;
    }

#pragma unroll
    for (int i = 0; i <REDUCTION_SIZE/INPUT_WIDTH; i++)
	sum += level5[i];

    *output_ddr = sum;
}

