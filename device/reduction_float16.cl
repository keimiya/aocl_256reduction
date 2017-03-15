/**
 * @file reduction_float16.cl
 * @brief Device kernel code for 256-elements reduction written in Altera OpenCL.
 * @par   Input width is float16(32bit*16=512bit=64byte)
 * @date  2017/03/14 Copy noda-chan's reduction_single_1k
 * @date  2017/03/15 Start implementing float16 version
 * @ToDo  Understan the number of DDR4's bank
 * @author T.Miyajima
 */

/**
 * AOCL device code
 * @param input_ddr float16 (32bit*16=512bit=64byte) from DDR4-2133
 * @param output_ddr 
 */
__kernel void reduction_float16(__global volatile const float16* restrict input_ddr,
				__global volatile float* restrict outpu_ddr)
{
    const int REDUCTION_SIZE = 256;
    const int INPUT_WIDTH    = 16; // equal to float16
    float sum = 0.0e0;

    float buf1[REDUCTION_SIZE];
    float buf2[REDUCTION_SIZE>>1];
    float buf3[REDUCTION_SIZE>>2];
    float buf4[REDUCTION_SIZE>>3];
    float buf5[REDUCTION_SIZE>>4];
    float buf6[REDUCTION_SIZE>>5];
    float buf7[REDUCTION_SIZE>>6];
    float buf8[REDUCTION_SIZE>>7];
    float buf9;

    // 1st level: directory read the input value from DDR    
    for (int i = 0; i < REDUCTION_SIZE/INPUT_WIDTH; i++){
    	buf1[(i*INPUT_WIDTH)+0] = input_ddr[i].s0 + input_ddr[i].s1;
    	buf1[(i*INPUT_WIDTH)+1] = input_ddr[i].s2 + input_ddr[i].s3;
    	buf1[(i*INPUT_WIDTH)+2] = input_ddr[i].s4 + input_ddr[i].s5;
    	buf1[(i*INPUT_WIDTH)+3] = input_ddr[i].s6 + input_ddr[i].s7;
    	buf1[(i*INPUT_WIDTH)+4] = input_ddr[i].s8 + input_ddr[i].s9;
    	buf1[(i*INPUT_WIDTH)+5] = input_ddr[i].sA + input_ddr[i].sB;
    	buf1[(i*INPUT_WIDTH)+6] = input_ddr[i].sC + input_ddr[i].sD;
    	buf1[(i*INPUT_WIDTH)+7] = input_ddr[i].sE + input_ddr[i].sF;
    }

#pragma unroll 
    for (int i = 0; i < (REDUCTION_SIZE>>1); i++)
	buf2[i] = buf1[i*2 + 0] + buf1[i*2 + 1];

#pragma unroll 
    for (int i = 0; i < (REDUCTION_SIZE>>2); i++)
	buf3[i] = buf2[i*2+0] + buf2[i*2+1];  

#pragma unroll 
    for (int i = 0; i < (REDUCTION_SIZE>>3); i++)
	buf4[i] = buf3[i*2+0] + buf3[i*2+1];  

#pragma unroll
    for (int i = 0; i < (REDUCTION_SIZE>>4); i++)
	buf5[i] = buf4[i*2+0] + buf4[i*2+1];

#pragma unroll
    for (int i = 0; i < (REDUCTION_SIZE>>5); i++)
	buf6[i] = buf5[i*2+0] + buf5[i*2+1];

#pragma unroll
    for (int i = 0; i < (REDUCTION_SIZE>>6); i++)
	buf7[i] = buf6[i*2+0] + buf6[i*2+1];

#pragma unroll
    for (int i = 0; i < (REDUCTION_SIZE>>7); i++)
	buf8[i] = buf7[i*2+0] + buf7[i*2+1];

    buf9 = buf8[0] + buf8[1];
    sum += buf9;
    /* } */
    /* } */
    *outpu_ddr = sum;
}

