My (re)implementation of CUDA SGEMM on Pascal platform.

Reference:

1. [5KK73 CUDA GEMM](http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmcuda) (kernel 1~3)
2. [OpenCL SGEMM Tutorial](https://cnugteren.github.io/tutorial/pages/page1.html) (kernel 4~8)


Tested on Ubuntu 16.04 LTS + GCC 5.1.0 + CUDA 8 + EVGA GTX 1070 (OC 2002MHz).

Different matrix size:

| n=m=k       |   1024 |   2048 |   3072 |   4096 |   5120 |   6144 |   7168 | 8192   |
| ----------- | -----: | -----: | -----: | -----: | -----: | -----: | -----: | ------ |
| CUBLAS      |   4276 |   5459 |   7016 |   7201 |   7118 |   7010 |   6861 | 7044   |
| My Kernel 8 |    925 |   2546 |   3246 |   4100 |   4282 |   4509 |   4606 | 4807   |
| Ratio       | 21.63% | 46.64% | 46.26% | 56.94% | 60.16% | 64.32% | 67.28% | 68.24% |

Different kernels (n=m=k=8K)

| n=m=k=8K | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | CUBLAS |
| -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ |
| GFlops   | 186  | 202  | 1001 | 1801 | 2375 | 3102 | 3280 | 4807 | 7044   |

For more information, please refer to [this page](http://enigmahuang.github.io/2017/07/06/my-CUDA-SGEMM/).