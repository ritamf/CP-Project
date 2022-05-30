
// Based on CUDA SDK template from NVIDIA

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>

// adicionado para copia de arrays
#include <algorithm>
#include <vector>
#include <iostream>

// includes, project
#include <helper_cuda.h>
#include <helper_image.h>

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

#define MAX_BRIGHTNESS 255

// pixel base type
// Use int instead `unsigned char' so that we can
// store negative values.
typedef int pixel_t;

__global__ void reduce1(pixel_t *h_idata, pixel_t *h_odata, int *size)
{
    // extern __shared__ pixel_t sdata[];

    // unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int size2 = size[0];
    // unsigned int write_global_id = blockDim.x * blockIdx.x;

    /// sdata[tid] = h_idata[id];

    //__syncthreads();

    // for(unsigned int s=1; s < blockDim.x; s++)
    //{
    //     sdata[tid]=sdata[tid + s]/4; // to obtain a faded background image

    //    __syncthreads();
    //}

    // write result for this block to global mem
    // if (tid == 0) {
    //    std::copy(sdata, sdata + blockDim.x, h_odata + id);
    //    //g_odata[id] = sdata[0];
    //}

    if (id < size2)
    {
        h_odata[id] = h_idata[id] / 4;
    }
}

__global__ void reduceWithLoop(pixel_t *h_idata, pixel_t *h_odata, int *size)
{

    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int size2 = size[0];
    unsigned int ws = size[1];
    unsigned int w = size[2];
    unsigned int threshold = size[3];

    unsigned int l, k; // indexes in image
    unsigned int Ix, Iy;     // gradient in XX and YY
    unsigned int R;          // R metric
    unsigned int sumIx2, sumIy2, sumIxIy;

    unsigned int j = id/w;
    unsigned int i = id - (j*w);

    if (id < size2)
    {
        sumIx2 = 0;
        sumIy2 = 0;
        sumIxIy = 0;
        for (k = -ws; k <= ws; k++) // height window
        {
            for (l = -ws; l <= ws; l++) // width window
            {
                Ix = ((int)h_idata[(i + k - 1) * w + j + l] - (int)h_idata[(i + k + 1) * w + j + l]) / 32;
                Iy = ((int)h_idata[(i + k) * w + j + l - 1] - (int)h_idata[(i + k) * w + j + l + 1]) / 32;
                sumIx2 += Ix * Ix;
                sumIy2 += Iy * Iy;
                sumIxIy += Ix * Iy;
            }
        }

        R = sumIx2 * sumIy2 - sumIxIy * sumIxIy - 0.05 * (sumIx2 + sumIy2) * (sumIx2 + sumIy2);
        if (R > threshold)
        {
            h_odata[i * w + j] = MAX_BRIGHTNESS;
        }
    }
}

// harris detector code to run on the host
void harrisDetectorHost(const pixel_t *h_idata, const int w, const int h,
                        const int ws,        // window size
                        const int threshold, // threshold value to detect corners
                        pixel_t *reference)
{
    int i, j, k, l; // indexes in image
    int Ix, Iy;     // gradient in XX and YY
    int R;          // R metric
    int sumIx2, sumIy2, sumIxIy;

    for (i = 0; i < h; i++) // height image
    {
        for (j = 0; j < w; j++) // width image
        {
            reference[i * w + j] = h_idata[i * w + j] / 4; // to obtain a faded background image
        }
    }

    for (i = ws + 1; i < h - ws - 1; i++) // height image
    {
        for (j = ws + 1; j < w - ws - 1; j++) // width image
        {
            sumIx2 = 0;
            sumIy2 = 0;
            sumIxIy = 0;
            for (k = -ws; k <= ws; k++) // height window
            {
                for (l = -ws; l <= ws; l++) // width window
                {
                    Ix = ((int)h_idata[(i + k - 1) * w + j + l] - (int)h_idata[(i + k + 1) * w + j + l]) / 32;
                    Iy = ((int)h_idata[(i + k) * w + j + l - 1] - (int)h_idata[(i + k) * w + j + l + 1]) / 32;
                    sumIx2 += Ix * Ix;
                    sumIy2 += Iy * Iy;
                    sumIxIy += Ix * Iy;
                }
            }

            R = sumIx2 * sumIy2 - sumIxIy * sumIxIy - 0.05 * (sumIx2 + sumIy2) * (sumIx2 + sumIy2);
            if (R > threshold)
            {
                reference[i * w + j] = MAX_BRIGHTNESS;
            }
        }
    }
}

// harris detector code to run on the GPU
void harrisDetectorDevice(const pixel_t *h_idata, const int w, const int h,
                          const int ws, const int threshold,
                          pixel_t *h_odata)
{
    // TODO

    // int i, j, k, l; // indexes in image
    // int Ix, Iy;     // gradient in XX and YY
    // int R;          // R metric
    // int sumIx2, sumIy2, sumIxIy;

    int size = h * w;

    int size_arr[5];

    size_arr[0] = size;
    size_arr[1] = ws;
    size_arr[2] = w;
    size_arr[3] = threshold;

    int memsize = size * sizeof(pixel_t);
    int memsize_size_arr = 5 * sizeof(int);

    int threadsPerBlock = 32;

    int blocksPerGrid = size / threadsPerBlock;

    // int MaxRowPerBlock = threadsPerBlock/h;
    // int threadsPerBlock = 32;

    // int blocksPerGrid = (size + threadsPerBlock - 1)/threadsPerBlock;

    // int gaussKernelSize = 7;

    // allocate host memory
    pixel_t *devPtrh_idata;
    pixel_t *devPtrh_odata;
    int *devPtrsize;

    // Allocate device memory
    cudaMalloc((void **)&devPtrh_idata, memsize);
    cudaMalloc((void **)&devPtrh_odata, memsize);
    cudaMalloc((void **)&devPtrsize, memsize_size_arr);

    // Copy data (data to process) from host to device (from CPU to GPU)
    cudaMemcpy(devPtrh_idata, h_idata, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrh_odata, h_odata, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrsize, size_arr, memsize_size_arr, cudaMemcpyHostToDevice);

    // __global__ functions are called:  Func <<< dim grid, dim block >>> (parameter);
    dim3 dimGrid(blocksPerGrid, 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);

    // Execute the Kernel
    reduce1<<<dimGrid, dimBlock>>>(devPtrh_idata, devPtrh_odata, devPtrsize);

    // Copy data from device (results) back to host
    cudaMemcpy(h_odata, devPtrh_odata, memsize, cudaMemcpyDeviceToHost);
    cudaFree(devPtrh_odata);
    
    
    cudaMalloc((void **)&devPtrh_odata, memsize);


    cudaMemcpy(devPtrh_odata, h_odata, memsize, cudaMemcpyHostToDevice);
    reduceWithLoop<<<dimGrid, dimBlock>>>(devPtrh_idata, devPtrh_odata, devPtrsize);
    cudaMemcpy(h_odata, devPtrh_odata, memsize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(devPtrh_idata);
    cudaFree(devPtrh_odata);
    cudaFree(devPtrsize);
}

// print command line format
void usage(char *command)
{
    printf("Usage: %s [-h] [-d device] [-i inputfile] [-o outputfile] [-r referenceFile] [-w windowsize] [-t threshold]\n", command);
}

// main
int main(int argc, char **argv)
{

    // default command line options
    int deviceId = 0;
    char *fileIn = (char *)"chess.pgm",
         *fileOut = (char *)"resultCuda.pgm",
         *referenceOut = (char *)"referenceCuda.pgm";
    unsigned int ws = 1, threshold = 500;

    // parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "d:i:o:r:w:t:h")) != -1)
    {
        switch (opt)
        {

        case 'd':
            if (sscanf(optarg, "%d", &deviceId) != 1)
            {
                usage(argv[0]);
                exit(1);
            }
            break;

        case 'i':
            if (strlen(optarg) == 0)
            {
                usage(argv[0]);
                exit(1);
            }

            fileIn = strdup(optarg);
            break;
        case 'o':
            if (strlen(optarg) == 0)
            {
                usage(argv[0]);
                exit(1);
            }
            fileOut = strdup(optarg);
            break;
        case 'r':
            if (strlen(optarg) == 0)
            {
                usage(argv[0]);
                exit(1);
            }
            referenceOut = strdup(optarg);
            break;
        case 'w':
            if (strlen(optarg) == 0 || sscanf(optarg, "%d", &ws) != 1)
            {
                usage(argv[0]);
                exit(1);
            }
            break;
        case 't':
            if (strlen(optarg) == 0 || sscanf(optarg, "%d", &threshold) != 1)
            {
                usage(argv[0]);
                exit(1);
            }
            break;
        case 'h':
            usage(argv[0]);
            exit(0);
            break;
        }
    }

    // select cuda device
    checkCudaErrors(cudaSetDevice(deviceId));

    // create events to measure host harris detector time and device harris detector time

    cudaEvent_t startH, stopH, startD, stopD;
    checkCudaErrors(cudaEventCreate(&startH));
    checkCudaErrors(cudaEventCreate(&stopH));
    checkCudaErrors(cudaEventCreate(&startD));
    checkCudaErrors(cudaEventCreate(&stopD));

    // allocate host memory
    pixel_t *h_idata = NULL;
    unsigned int h, w;

    // load pgm
    if (sdkLoadPGM<pixel_t>(fileIn, &h_idata, &w, &h) != true)
    {
        printf("Failed to load image file: %s\n", fileIn);
        exit(1);
    }

    // allocate mem for the result on host side
    pixel_t *h_odata = (pixel_t *)malloc(h * w * sizeof(pixel_t));
    pixel_t *reference = (pixel_t *)malloc(h * w * sizeof(pixel_t));

    // detect corners at host

    checkCudaErrors(cudaEventRecord(startH, 0));
    harrisDetectorHost(h_idata, w, h, ws, threshold, reference);
    checkCudaErrors(cudaEventRecord(stopH, 0));
    checkCudaErrors(cudaEventSynchronize(stopH));

    // detect corners at GPU
    checkCudaErrors(cudaEventRecord(startD, 0));
    harrisDetectorDevice(h_idata, w, h, ws, threshold, h_odata);
    checkCudaErrors(cudaEventRecord(stopD, 0));
    checkCudaErrors(cudaEventSynchronize(stopD));

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    float timeH, timeD;
    checkCudaErrors(cudaEventElapsedTime(&timeH, startH, stopH));
    printf("Host processing time: %f (ms)\n", timeH);
    checkCudaErrors(cudaEventElapsedTime(&timeD, startD, stopD));
    printf("Device processing time: %f (ms)\n", timeD);

    // save output images
    if (sdkSavePGM<pixel_t>(referenceOut, reference, w, h) != true)
    {
        printf("Failed to save image file: %s\n", referenceOut);
        exit(1);
    }
    if (sdkSavePGM<pixel_t>(fileOut, h_odata, w, h) != true)
    {
        printf("Failed to save image file: %s\n", fileOut);
        exit(1);
    }

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(reference);

    checkCudaErrors(cudaDeviceReset());
}
