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
    const int INPUT_WIDTH    = 16;// equal to float16
    const int ARRAY_INDEX    = REDUCTION_SIZE/INPUT_WIDTH;
    float sum = 0.0e0;

    float level2[8];
    float level3[4];
    float level4[2];
    float level5[16];

    // Each level of reduction takes 3[cycle] since all the adds are run in parallel
    for (int i = 0; i < ARRAY_INDEX; i++){

	// 1st level: directory read the input value from DDR
	level2[0] = input_ddr[i].s0 + input_ddr[i].s1; // fp32 add takes 3[cycle]
	level2[1] = input_ddr[i].s2 + input_ddr[i].s3;
	level3[0] = level2[0] + level2[1]; 	// 2nd level

	level2[2] = input_ddr[i].s4 + input_ddr[i].s5;
	level2[3] = input_ddr[i].s6 + input_ddr[i].s7;
	level3[1] = level2[2] + level2[3];
	
	level4[0] = level3[0] + level3[1]; // 3rd level
	

	level2[4] = input_ddr[i].s8 + input_ddr[i].s9;
	level2[5] = input_ddr[i].sA + input_ddr[i].sB;
	level3[2] = level2[4] + level2[5];

	level2[6] = input_ddr[i].sC + input_ddr[i].sD;
	level2[7] = input_ddr[i].sE + input_ddr[i].sF;
	level3[3] = level2[6] + level2[7];

	level4[1] = level3[2] + level3[3];

	// 4th level
	level5[(i) + 0] = level4[0] + level4[1];
    }

#pragma unroll
    for (int i = 0; i <REDUCTION_SIZE/INPUT_WIDTH; i++)
	sum += level5[i];

    *output_ddr = sum;
}

