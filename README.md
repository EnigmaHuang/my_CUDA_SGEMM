My (re)implementation of CUDA SGEMM for Kepler platform.

Reference:

1. [5KK73 CUDA GEMM](http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmcuda) (kernel 1~3, *my_sgemm_kernels.cuh*)
2. [OpenCL SGEMM Tutorial](https://cnugteren.github.io/tutorial/pages/page1.html) (kernel 6~9, *my_sgemm_cm_kernels.cuh*)


Tested on CentOS 7.3 + GCC 4.8.5 + CUDA 8 + Tesla K40c.

Different matrix size:

| n=m=k       | 512      | 1024     | 1536     | 2048     | 2560     | 3072     | 3584     | 4096     | 4608     | 5120     |
| ----------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| CUBLAS      | 1630.23  | 2416.58  | 2744.67  | 3291.79  | 3193.53  | 3196.34  | 3208.96  | 3354.68  | 3208.57  | 3111.38  |
| my_kernel_9 | 753.73   | 1195.91  | 1440.04  | 1515.59  | 1508.38  | 1526.26  | 1541.62  | 1571.39  | 1570.65  | 1584.53  |
| ratio       | 0.462346 | 0.494877 | 0.524668 | 0.460415 | 0.472324 | 0.477502 | 0.480411 | 0.468417 | 0.489517 | 0.509269 |

Different kernels (n=m=k=4K)

| n=m=k=4K | 1     | 2      | 3      | 4      | 5      | 6      | 7       | 8       | 9       | CUBLAS  |
| -------- | ----- | ------ | ------ | ------ | ------ | ------ | ------- | ------- | ------- | ------- |
| my_sgemm | 161.2 | 154.41 | 432.71 | 623.35 | 575.36 | 841.98 | 1105.96 | 1605.77 | 1571.39 | 3354.68 |

For more information, please refer to [this page](http://enigmahuang.github.io/2017/07/06/my-CUDA-SGEMM/).