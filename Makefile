default:
	nvcc -O2 brute.cu -o brute

bench:
	nvcc -O2 bench.cu -o bench

debug:
	nvcc -g -G brute.cu