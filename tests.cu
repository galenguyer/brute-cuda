#include <cuda.h>
#include <stdint.h>
#include "md5.cu"

__device__ int best_prefix = 6;
__device__ int best_suffix = 6;

__device__ int pow(int x, int y) {
    int res = 1;
    for(int i = 0; i < y; i++) {
        res *= x;
    }
    return res;
}

__device__ char* itoa(int i) {
    int len = 1, temp = i;
    while ((temp /= 10) > 0) {
        len++;
    }
    char* a = (char*)malloc(sizeof(char)*(32 + 1));
    for(int idx = len-1; idx >= 0; idx--) {
        temp = i / pow(10, idx) % 10;
        a[len-idx-1] = temp + '0';
    }
    a[len] = '\0';
    return a;
}

__device__ int strlen(char* str) {
    int len;
    for(len = 0; str[len] != '\0'; len++) { }
    return len;
}

__device__ void hexify(const unsigned char* input, char* output) {
    const char* map = "0123456789abcdef";

    for(int i = 0; i < 16; i++) {
        output[i*2] = map[(input[i] & 0xF0) >> 4];
        output[i*2+1] = map[(input[i] & 0x0F)];
    }
    output[32] = '\0';
}

__global__ void brute(int threads, int* randNums) {
    int seed = randNums[(blockIdx.x*threads)+threadIdx.x];
    char* seed_buffer = itoa(seed);
    //printf("[%d->%d] starting with seed %s\n", blockIdx.x, threadIdx.x, seed_buffer);

    unsigned char seed_md5[16];
    md5(seed_buffer, strlen(seed_buffer), seed_md5);
    
    char buffer[33];
    hexify(seed_md5, buffer);

    while (1) {
        char* old_buffer = (char*)malloc(sizeof(char)*33);
        memcpy(old_buffer, buffer, 33);

        md5(buffer, strlen(buffer), seed_md5);
        hexify(seed_md5, buffer);
        for (int i = 0; i < 32; i++){
            if (old_buffer[i] != buffer[i]) {
                if (i > best_prefix) {
                    printf("[%d->%d] new best prefix match: %d characters\n", blockIdx.x, threadIdx.x, i);
                    printf("[%d->%d] %s -> %s\n", blockIdx.x, threadIdx.x, old_buffer, buffer);
                    best_prefix = i;
                }
                break;
            }
        }
        for (int i = 32-1; i > 0; i--){
            if (old_buffer[i] != buffer[i]) {
                if (32-i > best_suffix) {
                    printf("[%d->%d] new best suffix match: %d characters\n", blockIdx.x, threadIdx.x, 32-i-1);
                    printf("[%d->%d] %s -> %s\n", blockIdx.x, threadIdx.x, old_buffer, buffer);
                    best_suffix = i;
                }
                break;
            }
        }
        free(old_buffer);
    }
}

__global__ void bench(int blocks, int threads, int* randNums) {
    int seed = randNums[(blockIdx.x*threads)+threadIdx.x];
    char* seed_buffer = itoa(seed);
    //printf("[%d->%d] starting with seed %s\n", blockIdx.x, threadIdx.x, seed_buffer);

    unsigned char seed_md5[16];
    md5(seed_buffer, strlen(seed_buffer), seed_md5);
    
    char buffer[33];
    hexify(seed_md5, buffer);

    for(int i = 0; i < (1000000/(blocks * threads)); i++) {
        char* old_buffer = (char*)malloc(sizeof(char)*33);
        memcpy(old_buffer, buffer, 33);

        md5(buffer, strlen(buffer), seed_md5);
        hexify(seed_md5, buffer);

        for (int i = 0; i < 32; i++){
            if (old_buffer[i] != buffer[i]) {
                break;
            }
        }
        for (int i = 32-1; i > 0; i--){
            if (old_buffer[i] != buffer[i]) {
                break;
            }
        }

        free(old_buffer);
    }
}

int run_test(const char* name, const char* result, const char* expected) {
    if (strcmp(expected, result) == 0) {
        printf("TEST PASSED: %s: expected %s, got %s\n", name, expected, result);
        return 1;
    } else {
        printf("TEST FAILED: %s: expected %s, got %s\n", name, expected, result);
        return 0;
    }
}

int main() {
    int h_blocks = 128;
    int h_threads = 256;
    int* h_randNums = (int*)malloc(sizeof(int) * h_blocks * h_threads);
    srand(time(0));
    for (int i = 0; i < h_blocks * h_threads; i++) {
        h_randNums[i] = rand();
    }
    int* d_randNums;
    cudaMalloc((void**)&d_randNums, sizeof(int)*h_blocks*h_threads);
    cudaMemcpy(d_randNums, h_randNums, sizeof(int)*h_blocks*h_threads, cudaMemcpyHostToDevice);

    // float elapsedTime;
    // cudaEvent_t bench_start, bench_stop;
    // cudaEventCreate(&bench_start); 
    // cudaEventRecord(bench_start, 0);
    // printf("Running 1,000,000 iterations...\n");
    // bench<<<h_blocks, h_threads>>>(h_blocks, h_threads, d_randNums);
    // cudaEventCreate(&bench_stop);
    // cudaEventRecord(bench_stop, 0); 
    // cudaEventSynchronize(bench_stop); 
    // cudaEventElapsedTime(&elapsedTime, bench_start, bench_stop);
    // printf("Elapsed time: %fms\n\n", elapsedTime);

    cudaMalloc((void**)&d_randNums, sizeof(int)*h_blocks*h_threads);
    cudaMemcpy(d_randNums, h_randNums, sizeof(int)*h_blocks*h_threads, cudaMemcpyHostToDevice);
    brute<<<h_blocks, h_threads>>>(h_threads, d_randNums);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}