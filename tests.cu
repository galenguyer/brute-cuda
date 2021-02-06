#include <cuda.h>
#include <stdint.h>
#include "md5.cu"

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
    char* a = (char*)malloc(sizeof(char)*(len + 1));
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

__device__ char* hexify(unsigned char* input) {
    char* output = (char*)malloc(sizeof(char)*(MD5_HASH_SIZE*2+1));
    const char* map = "0123456789abcdef";

    for(int i = 0; i < MD5_HASH_SIZE; i++) {
        output[i*2] = map[(input[i] & 0xF0) >> 4];
        output[i*2+1] = map[(input[i] & 0x0F)];
    }
    output[MD5_HASH_SIZE*2] = '\0';
    return output;
}

__global__ void brute(int threads, int* randNums) {
    int seed = randNums[(blockIdx.x*threads)+threadIdx.x];
    char* buffer = itoa(seed);
    printf("starting thread %d with seed %s\n", (blockIdx.x * threads) + threadIdx.x, buffer);
    buffer = hexify(md5(buffer, strlen(buffer)));
    while (1) {
        //char* old_buffer = buffer;
        buffer = hexify(md5(buffer, strlen(buffer)));
        /*for (int i = 32-1; i > 0; i--){
            printf("%s, %s\n", old_buffer, buffer);
            if (old_buffer[i] != buffer[i]) {
                if (32-i > 4) {
                    printf("new best suffix match: %d characters\n", 32-i-1);
                    printf("%s -> %s\n", old_buffer, buffer);
                }
                break;
            }
        }
        free(old_buffer);*/
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
    int blocks = 16;
    int threads = 16;
    int* h_randNums = (int*)malloc(sizeof(int) * blocks * threads);
    for (int i = 0; i < blocks * threads; i++) {
        h_randNums[i] = rand();
    }
    int* d_randNums;
    cudaSetDevice(0);
    cudaMalloc((void**)&d_randNums, sizeof(int)*blocks*threads);
    cudaMemcpy(d_randNums, h_randNums, sizeof(int)*blocks*threads, cudaMemcpyHostToDevice);
    brute<<<blocks, threads>>>(threads, d_randNums);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}