line_calc: ray_tree.cu
	nvcc -o ray_tree -O3 -std=c++11 ray_tree.cu

clean:
	rm ray_tree
