# system dependent flags for CUDA sdk path
CUDA_SDK?=/usr/local/cuda/sdk
CUDA_PATH?=/usr/local/cuda


ifeq ($(points), 1)
NVCC = nvcc
NVCCFLAGS = -ccbin /usr/bin/gcc \
	    -I$(CUDA_SDK)/common/inc \
	    -I$(CUDA_PATH)/include -arch=sm_20 -m64 -g -lineinfo -O0 --compiler-options -Wall -DPOINTS
LD = gcc
LFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lm
else
NVCC = nvcc
NVCCFLAGS = -ccbin /usr/bin/gcc \
	    -I$(CUDA_SDK)/common/inc \
	    -I$(CUDA_PATH)/include -arch=sm_20 -m64 -O2 --compiler-options -O2
LD = gcc
LFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lm
endif

PROJ = nevilleInterpolation
.PHONY: clean

# only one file in project, so no dependency :-)
all:
	$(NVCC) $(NVCCFLAGS) -o $(PROJ).o -c $(PROJ).cu
	$(LD) $(LFLAGS) -o $(PROJ) $(PROJ).o

clean:
	rm -f *.o $(PROJ)
	