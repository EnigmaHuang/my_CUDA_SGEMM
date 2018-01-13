EXENAME = my_sgemm_cuda

CC      = g++
CFLAGS  = -O3 -fopenmp

CCSRCS  = $(wildcard *.cc)
CUSRCS  = $(wildcard *.cu)
OBJS    = $(CCSRCS:.cc=.o)
OBJS   += $(CUSRCS:.cu=.o)

CUDA_PATH  = /usr/local/cuda-8.0
NVCC       = $(CUDA_PATH)/bin/nvcc
NVFLAGS    = -O3 -std=c++11 -arch=sm_61 -Xcompiler -fopenmp -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -Wno-deprecated-gpu-targets
LDFLAGS    = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -Wno-deprecated-gpu-targets 

build : $(EXENAME)

run : build
	export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64 && \
	./$(EXENAME) 

all : run

$(EXENAME): $(OBJS) 
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $(EXENAME) $(OBJS)

%.o : %.cu  
	$(NVCC) $(NVFLAGS) -c $^ 

%.o: %.cc 
	$(CC) $(CFLAGS) -c $^

clean:
	$(RM) *.o $(EXENAME)
