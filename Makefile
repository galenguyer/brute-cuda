default:
	nvcc -O2 brute.cu -o brute

debug:
	nvcc -g -G brute.cu