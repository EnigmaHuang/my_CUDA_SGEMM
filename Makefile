EXENAME = sgemm_cuda 

CC      = g++
CFLAGS  = -O2 -fopenmp

CCSRCS  = $(wildcard *.cc)
CUSRCS  = $(wildcard *.cu)
OBJS    = $(CCSRCS:.cc=.o)
OBJS   += $(CUSRCS:.cu=.o)

CUDA_PATH  = /usr/local/cuda-8.0
NVCC       = $(CUDA_PATH)/bin/nvcc
NFLAGS     = -O3 -arch=sm_35 -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -Wno-deprecated-gpu-targets #-DPROFILING -lineinfo
LDFLAGS    = -O3 -arch=sm_35 -L$(CUDA_PATH)/lib64 -lcudart -lcublas -Wno-deprecated-gpu-targets #-lineinfo

build : $(EXENAME)

run : build
	export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64 && \
	./$(EXENAME) 

all : run

$(EXENAME): $(OBJS) 
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $(EXENAME) $(OBJS)

%.o : %.cu 
	$(NVCC) $(NFLAGS) -c $^ 

%.o: %.cc 
	$(CC) $(CFLAGS) -c $^

clean:
	$(RM) *.o $(EXENAME)
