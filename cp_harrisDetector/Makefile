
.PHONY: all clean

TARGETS = harrisDetectorCuda harrisDetectorCudaShared harrisDetectorOpenMP testDiffs

all: $(TARGETS)

harrisDetectorCuda: harrisDetectorCuda.cu
	nvcc -arch=sm_30 -O3 -ICommon $^ -o $@

harrisDetectorCudaShared: harrisDetectorCudaShared.cu
	nvcc -arch=sm_30 -O3 -ICommon $^ -o $@

harrisDetectorOpenMP: harrisDetectorOpenMP.c
	g++ -fopenmp -O3 -ICommon $^ -o $@

testDiffs: testDiffs.c
	g++ -O3 -ICommon $^ -o $@

clean:
	rm -f $(TARGETS) 
	rm -f referenceCuda.pgm resultCuda.pgm referenceOpenMP.pgm resultOpenMP.pgm
