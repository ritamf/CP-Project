
// Based on CUDA SDK template from NVIDIA

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>
#include <sys/time.h>

// includes, project
#include <helper_cuda.h>
#include <helper_image.h>
#include <omp.h>

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#define MAX_BRIGHTNESS 255
 
// pixel base type
// Use int instead `unsigned char' so that we can
// store negative values.
typedef int pixel_t;


// harris detector code to run on the host
void harrisDetectorHost(const pixel_t *h_idata, const int w, const int h, 
                const int ws,               // window size
                const int threshold,        // threshold value to detect corners
                pixel_t * reference)
{
    int i,j,k,l;  // indexes in image
    int Ix, Iy;   // gradient in XX and YY
    int R;        // R metric
    int sumIx2, sumIy2, sumIxIy;

    for(i=0; i<h; i++) //height image
    {
        for(j=0; j<w; j++) //width image
        {
            reference[i*w+j]=h_idata[i*w+j]/4; // to obtain a faded background image
        }
    }

    for(i=ws+1; i<h-ws-1; i++) //height image
    {
        for(j=ws+1; j<w-ws-1; j++) //width image
        {
           sumIx2=0;sumIy2=0;sumIxIy=0;
           for(k=-ws; k<=ws; k++) //height window
              {
                  for(l=-ws; l<=ws; l++) //width window
                  {
                        Ix = ((int)h_idata[(i+k-1)*w + j+l] - (int)h_idata[(i+k+1)*w + j+l])/32;         
                        Iy = ((int)h_idata[(i+k)*w + j+l-1] - (int)h_idata[(i+k)*w + j+l+1])/32;         
                        sumIx2 += Ix*Ix;
                        sumIy2 += Iy*Iy;
                        sumIxIy += Ix*Iy;
                  }
              }

              R = sumIx2*sumIy2-sumIxIy*sumIxIy-0.05*(sumIx2+sumIy2)*(sumIx2+sumIy2);
              if(R > threshold) {
                   reference[i*w+j]=MAX_BRIGHTNESS; 
              }
        }
    }
}   

// harris detector code using OpenMP
void harrisDetectorOpenMP(const pixel_t *h_idata, const int w, const int h, 
                  const int ws, const int threshold, 
                  pixel_t * h_odata)
{
    //TODO
    int n_threads = 2;

    omp_set_num_threads(n_threads);

    int i,j,k,l;  // indexes in image
    int Ix, Iy;   // gradient in XX and YY
    int R;        // R metric
    int sumIx2, sumIy2, sumIxIy;

    #pragma omp parallel for shared(h_odata) private(i, j) firstprivate(h_idata, w, h)
    for(i=0; i<h; i++) //height image
    {
        for(j=0; j<w; j++) //width image
        {   
            //#pragma omp critical
            h_odata[i*w+j]=h_idata[i*w+j]/4; // to obtain a faded background image
        }
    }


    for(i=ws+1; i<h-ws-1; i++) //height image
    {
        for(j=ws+1; j<w-ws-1; j++) //width image
        {
           sumIx2=0;sumIy2=0;sumIxIy=0;
           for(k=-ws; k<=ws; k++) //height window
              {
                  for(l=-ws; l<=ws; l++) //width window
                  {
                        Ix = ((int)h_idata[(i+k-1)*w + j+l] - (int)h_idata[(i+k+1)*w + j+l])/32;         
                        Iy = ((int)h_idata[(i+k)*w + j+l-1] - (int)h_idata[(i+k)*w + j+l+1])/32;         
                        sumIx2 += Ix*Ix;
                        sumIy2 += Iy*Iy;
                        sumIxIy += Ix*Iy;
                  }
              }

              R = sumIx2*sumIy2-sumIxIy*sumIxIy-0.05*(sumIx2+sumIy2)*(sumIx2+sumIy2);
              if(R > threshold) {
                   h_odata[i*w+j]=MAX_BRIGHTNESS; 
              }
        }
    }

}

// print command line format
void usage(char *command) 
{
    printf("Usage: %s [-h] [-i inputfile] [-o outputfile] [-r referenceFile] [-w windowsize] [-t threshold]\n",command);
}

// main
int main( int argc, char** argv) 
{

    // default command line options
    int deviceId = 0;
    char *fileIn        = (char *)"chess.pgm",
         *fileOut       = (char *)"resultOpenMP.pgm",
         *referenceOut  = (char *)"referenceOpenMP.pgm";
    unsigned int ws = 1, threshold = 500;

    // parse command line arguments
    int opt;
    while( (opt = getopt(argc,argv,"i:o:r:w:t:h")) !=-1)
    {
        switch(opt)
        {

            case 'i':
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }

                fileIn = strdup(optarg);
                break;
            case 'o':
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                fileOut = strdup(optarg);
                break;
            case 'r':
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                referenceOut = strdup(optarg);
                break;
            case 'w':
                if(strlen(optarg)==0 || sscanf(optarg,"%d",&ws)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
            case 't':
                if(strlen(optarg)==0 || sscanf(optarg,"%d",&threshold)!=1)
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

    // allocate host memory
    pixel_t * h_idata=NULL;
    unsigned int h,w;

    //load pgm
    if (sdkLoadPGM<pixel_t>(fileIn, &h_idata, &w, &h) != true) {
        printf("Failed to load image file: %s\n", fileIn);
        exit(1);
    }

    // allocate mem for the result on host side
    pixel_t * h_odata   = (pixel_t *) malloc( h*w*sizeof(pixel_t));
    pixel_t * reference = (pixel_t *) malloc( h*w*sizeof(pixel_t));
 
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // detect corners
    harrisDetectorHost(h_idata, w, h, ws, threshold, reference);   

    gettimeofday(&end, NULL);

    struct timeval startMP, endMP;
    gettimeofday(&startMP, NULL);

    // detect corners with OpenMP
    harrisDetectorOpenMP(h_idata, w, h, ws, threshold, h_odata);   

    gettimeofday(&endMP, NULL);
    
    printf( "Host processing time: %f (ms)\n", (end.tv_sec-start.tv_sec)*1000.0 + ((double)(end.tv_usec - start.tv_usec))/1000.0);
    printf( "OpenMP processing time: %f (ms)\n", (endMP.tv_sec-startMP.tv_sec)*1000.0 + ((double)(endMP.tv_usec - startMP.tv_usec))/1000.0);

    // save output images
    if (sdkSavePGM<pixel_t>(referenceOut, reference, w, h) != true) {
        printf("Failed to save image file: %s\n", referenceOut);
        exit(1);
    }
    if (sdkSavePGM<pixel_t>(fileOut, h_odata, w, h) != true) {
        printf("Failed to save image file: %s\n", fileOut);
        exit(1);
    }

    // cleanup memory
    free( h_idata );
    free( h_odata );
    free( reference );
}
