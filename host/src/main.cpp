#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <sys/time.h>

using namespace aocl_utils;
 
// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements
scoped_array<cl_mem> input_a_buf; // num_devices elements
//scoped_array<cl_mem> input_b_buf; // num_devices elements
scoped_array<cl_mem> output_buf; // num_devices elements
//scoped_aligned_ptr<cl_mem> output_buf;
// Problem data.
unsigned N = 256; // problem size
scoped_array<scoped_aligned_ptr<float> > input_a; // num_devices elements
//scoped_array<scoped_aligned_ptr<double> > output; // num_devices elements
scoped_array<float> output;
scoped_array<float> ref_output;
//scoped_array<scoped_array<double> > ref_output; // num_devices elements
scoped_array<unsigned> n_per_device; // num_devices elements

// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void cleanup();

// Entry point.
int main(int argc, char **argv) {
    Options options(argc, argv);

    // Optional argument to specify the problem size.
    if(options.has("n")) {
	N = options.get<unsigned>("n");
    }

    // Initialize OpenCL.
    if(!init_opencl()) {
	return -1;
    }

    // Initialize the problem data.
    // Requires the number of devices to be known.
    init_problem();

    // Run the kernel.
    run();

    // Free the resources allocated
    cleanup();

    return 0;
}

/////// HELPER FUNCTIONS ///////
//ランダムなinputデータを生成
// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
    return float(rand()) / float(RAND_MAX) * 20.0e0 - 10.0e0;
}

// Initializes the OpenCL objects.
bool init_opencl() {
    cl_int status;

    printf("Initializing OpenCL\n");

    if(!setCwdToExeDir()) {
	return false;
    }

    // Get the OpenCL platform.
    platform = findPlatform("Altera");
    if(platform == NULL) {
	printf("ERROR: Unable to find Altera OpenCL platform.\n");
	return false;
    }

    // Query the available OpenCL device.
    //
    device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    printf("Platform: %s\n", getPlatformName(platform).c_str());
    printf("Using %d device(s)\n", num_devices);
    for(unsigned i = 0; i < num_devices; ++i) {
	printf("  %s\n", getDeviceName(device[i]).c_str());
    }

    // Create the context.
    //コンテキストの生成
    context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    // Create the program for all device. Use the first device as the
    // representative device (assuming all device are of the same type).
    std::string binary_file = getBoardBinaryFile("reduction_float16", device[0]);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

    // Build the program that was just created.
    //ビルド
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Create per-device objects.
    queue.reset(num_devices);
    kernel.reset(num_devices);
    n_per_device.reset(num_devices);
    input_a_buf.reset(num_devices);
    //  input_b_buf.reset(num_devices);
    output_buf.reset(num_devices);

    for(unsigned i = 0; i < num_devices; ++i) {
	// Command queue.
	//2. コマンドキューの作成
	queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Kernel.
	//カーネルオブジェクトの作成
	const char *kernel_name = "reduction_float16";
	kernel[i] = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel");

	// Determine the number of elements processed by this device.
	n_per_device[i] = N / num_devices; // number of elements handled by this device

	// Spread out the remainder of the elements over the first
	// N % num_devices.
	if(i < (N % num_devices)) {
	    n_per_device[i]++;
	}

	// Input buffers.
	//3. メモリオブジェクトの作成
	input_a_buf[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, 
					n_per_device[i] * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for input A");

	/*  input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
	    n_per_device[i] * sizeof(double), NULL, &status);
	    checkError(status, "Failed to create buffer for input B");
	*/
	// Output buffer.
	output_buf[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, 
				       sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for output");
    }

    return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
    if(num_devices == 0) {
	checkError(-1, "No devices");
    }

    input_a.reset(num_devices);
    //  input_b.reset(num_devices);
    output.reset(num_devices);
    ref_output.reset(num_devices);

    // Generate input vectors A and B and the reference output consisting
    // of a total of N elements.
    // We create separate arrays for each device so that each device has an
    // aligned buffer. 
    for(unsigned i = 0; i < num_devices; ++i) {
	input_a[i].reset(n_per_device[i]);
	// output[i].reset(n_per_device[i]);
	//ref_output[i].reset(n_per_device[i]);
	ref_output[i] = 0.0e0;
	output[i] = 0.0e0;
	for(unsigned j = 0; j < n_per_device[i]; ++j) 
	    input_a[i][j] = rand_float();
	//      ref_output[i] += input_a[i][j];
	const double start_cpu_time = getCurrentTimestamp();
	for(unsigned j = 0; j < n_per_device[i]; ++j) {
	    ref_output[i] += input_a[i][j];
	}
	float sum = ref_output[i];
	const double end_cpu_time = getCurrentTimestamp();
	printf("\ncpu time: %f ms\n", (end_cpu_time - start_cpu_time) * 1e3);
    }
}

void run() {
    cl_int status;

    //struct timeval test2_s, test2_e;
    //gettimeofday(&test2_s, NULL);
    cl_ulong time_ns = 0.0e0;
    const double start_time = getCurrentTimestamp();

    // Launch the problem for each device.
    scoped_array<cl_event> kernel_event(num_devices);
    scoped_array<cl_event> finish_event(num_devices);
    for(unsigned i = 0; i < num_devices; ++i) {
	//入力を各デバイスに転送する
	// Transfer inputs to each device. Each of the host buffers supplied to
	// clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
	// for the host-to-device transfer.
	cl_event write_event[2];
	//input_aの転
        float *input_ptr = (float*)clEnqueueMapBuffer(queue[i], input_a_buf[i], CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, n_per_device[i] * sizeof(float), 0, NULL, &write_event[0], &status);
	checkError(status, "Failed to map input");
	for (unsigned k = 0; k < n_per_device[i]; k++)
	    input_ptr[k] = input_a[0][k];

        float *output_ptr = (float*)clEnqueueMapBuffer(queue[i], output_buf[i], CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float), 0, NULL, &write_event[1], &status);
	checkError(status, "Failed to map output");

	/*status = clEnqueueWriteBuffer(queue[i], input_a_buf[i], CL_FALSE,
	  0, n_per_device[i] * sizeof(float), input_a[i], 0, NULL, &write_event[0]);
	  checkError(status, "Failed to transfer input A");
	*/	
	/*
	//outputの転送
	status = clEnqueueWriteBuffer(queue[i], output_buf[i], CL_FALSE,
	0, sizeof(double), output, 0, NULL, &write_event[1]);
	checkError(status, "Failed to transfer output");
	*/
	// Set kernel arguments.
	unsigned argi = 0;
	//カーネル引数の設定
	status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_a_buf[i]);
	checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &output_buf[i]);
	checkError(status, "Failed to set argument %d", argi - 1);

	// Enqueue kernel.
	// Use a global work size corresponding to the number of elements to add
	// for this device.
	// 
	// We don't specify a local work size and let the runtime choose
	// (it'll choose to use one work-group with the same size as the global
	// work-size).
	//
	// Events are used to ensure that the kernel is not launched until
	// the writes to the input buffers have completed.
	const size_t global_work_size = n_per_device[i];
	// printf("Launching for device %d (%d elements)\n", i, global_work_size);
	//カーネル実行

	//	const double start_k_time = getCurrentTimestamp();
	status = clEnqueueTask(queue[i], kernel[i], 1, write_event, &kernel_event[i]);
	/* status = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL,
	   &global_work_size, NULL, 1, write_event, &kernel_event[i]);*/
	checkError(status, "Failed to launch kernel");
	//	const double end_k_time = getCurrentTimestamp();
	
	//	printf("\nKernel time(getCurrentTimestamp) %f ms\n", (end_k_time - start_k_time) * 1e3);
	clWaitForEvents(num_devices, kernel_event);
	output[i] = *output_ptr;


	// Read the result. This the final operation.
	//
	/*	status = clEnqueueReadBuffer(queue[i], output_buf[i], CL_FALSE,
		0, sizeof(float), output, 1, &kernel_event[i], &finish_event[i]);
	*/
	// Release local events.
	clReleaseEvent(write_event[0]);
	clReleaseEvent(write_event[1]);
	status = clEnqueueUnmapMemObject(queue[i], input_a_buf[i], input_ptr, 0, NULL, NULL);
	checkError(status, "Failed to unmap input");
	status = clEnqueueUnmapMemObject(queue[i], output_buf[i], output_ptr, 0, NULL, NULL);
	checkError(status, "Failed to unmap output");
    }

    // Wait for all devices to finish.
    clWaitForEvents(num_devices, finish_event);
	
    time_ns = getStartEndTime(kernel_event[0]);

    const double end_time = getCurrentTimestamp();
    //gettimeofday(&test2_e, NULL);
    // Wall-clock time taken.
    printf("\nArria 10 SoC\nTurn_around_Time: %f ms\n", (end_time - start_time) * 1e3);

    // printf("\ntest_a10soc_time = %lf ms\n", ((test2_e.tv_sec - test2_s.tv_sec) + (test2_e.tv_usec - test2_s.tv_usec)*1.0E-6)*1000/1000);
    // Get kernel times using the OpenCL event profiling API.
    for(unsigned i = 0; i < num_devices; ++i) {
	//cl_ulong time_ns = getStartEndTime(kernel_event[i]);
	printf("Kernel time (device %d)(getStartEndTime): %f ms\n", i, double(time_ns) * 1e-6);
    }
    //printf("\nKernel time(getCurrentTimestamp) %f ms\n", (end_k_time - start_k_time) * 1e3);

    // Release all events.
    for(unsigned i = 0; i < num_devices; ++i) {
	clReleaseEvent(kernel_event[i]);
	clReleaseEvent(finish_event[i]);
    }
  
    // Verify results.
    bool pass = true;
    for(unsigned i = 0; i < num_devices && pass; ++i) {
	if(fabsf(output[i] - ref_output[i]) > 1.0e-10) {
	    printf("Failed verification @ device %d,\nOutput: %f\nReference: %f\n",
		   i, output[i], ref_output[i]);
	    pass = false;
	} else {
	    printf("\nOutput: %f\nReference: %f\n", output[i], ref_output[i] );
	}
    }

    printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");

  
}

// Free the resources allocated during initialization
void cleanup() {
    for(unsigned i = 0; i < num_devices; ++i) {
	if(kernel && kernel[i]) {
	    clReleaseKernel(kernel[i]);
	}
	if(queue && queue[i]) {
	    clReleaseCommandQueue(queue[i]);
	}
	if(input_a_buf && input_a_buf[i]) {
	    clReleaseMemObject(input_a_buf[i]);
	}/*
	   if(input_b_buf && input_b_buf[i]) {
	   clReleaseMemObject(input_b_buf[i]);
	   }*/
	if(output_buf && output_buf[i]) {
	    clReleaseMemObject(output_buf[i]);
	}
    }

    if(program) {
	clReleaseProgram(program);
    }
    if(context) {
	clReleaseContext(context);
    }
}

