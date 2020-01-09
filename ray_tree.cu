#include <iostream>
#include <math.h>
#include <random>
#include <fstream>

__device__ float square(float x) {
    return x*x;
}

struct Tree_Info {
    int n;
    double corner[3];
    double size; // assume cube
};

// Just build the tree in serial
// DOES NOT WORK
void buildtree(int n, float *r, float *h, float *coldens, int *tree_children, Tree_Info *tree_info) {
    // find max/min size
    int ii,kk;
    float *r_i;

    for ( kk=0 ; kk<3 ; kk++) tree_info->corner[kk] = r[kk];
    tree_info->size = 0.;

    for ( ii=1 ; ii<n ; ii++ ) {
        r_i = &(r[ii*3]);
        
        for ( kk=0 ; kk<3 ; kk++ ) {
            if ( r_i[kk]<tree_info->corner[kk] ) {
                tree_info->size+=tree_info->corner[kk]-r_i[kk];
                tree_info->corner[kk]-=r_i[kk];
            }
            if ( r_i[kk]>tree_info->corner[kk] ) {
                tree_info->size+=r_i[kk]-tree_info->corner[kk];
            }
        }
    }
    
    std::cout << "Corner, size: " << tree_info->corner[0] << " " << tree_info->corner[1] << " " << tree_info->corner[2] << " " << tree_info->size << std::endl;
}

// CUDA Kernel function function to calc number of collisions per particle
__global__ void calcray(int n, float *r, float *h, float *coldens, Tree_Info *tree_info) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
  
  int i,j,k;

  float crossp;
  
  float norm2,h2,dot;
  float *r_i,*r_j;
  float dr[3];

  i = index;
  coldens[i] = 0.;

  r_i = &(r[i*3]);
  norm2 = 0.;
  for (k=0 ; k<3 ; k++) norm2+=square(r_i[k]);

  for (j=0 ; j<n ; j++) {
    r_j = &(r[j*3]);
    // check if particle is in-between origin and target
    dot = 0.;
    for (k=0 ; k<3 ; k++) dot+=r_i[k]*r_j[k];

    if ( dot>0. && dot<norm2 ) {
        // check if ray intersects particle
        for (k=0 ; k<3 ; k++) dr[k] = r_i[k] - r_j[k];

        crossp = square( r_j[1]*dr[2]-r_j[2]*dr[1]);
        crossp+= square(-r_j[0]*dr[2]+r_j[2]*dr[0]);
        crossp+= square( r_j[0]*dr[1]-r_j[1]*dr[0]);
  
        h2 = square(h[j]);

        if ( crossp<=h2*norm2 ) {
          coldens[index]+=1.;
        }
    }
  }
}

int main(void) {
//   int N = 690286; // for comparison with tree code
  int N = 10000; // for quick tests

  float *r,*h,*coldens;
  float *d_coldens;
  
  int *tree_children;
  
  struct Tree_Info *tree_info;
  
  int tree_memory_factor = 8;

  cudaMallocManaged(&r, N*sizeof(float)*3);
  cudaMallocManaged(&h, N*sizeof(float));
  cudaMalloc(&d_coldens, N*sizeof(float));

  cudaMallocManaged(&tree_children, N*sizeof(int)*tree_memory_factor);
  cudaMallocManaged(&tree_info, sizeof(tree_info));

  coldens = new float[N];
  
  std::default_random_engine generator;
  std::uniform_real_distribution<float> locDistribution(-1.0,1.0);
//   std::uniform_real_distribution<float> hDistribution(0.005,0.02);
  std::uniform_real_distribution<float> hDistribution(0.05,0.1);

  
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    r[i*3] = locDistribution(generator);
    r[i*3+1] = locDistribution(generator);
    r[i*3+2] = locDistribution(generator);
    h[i] = hDistribution(generator);
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  buildtree(N,r,h,d_coldens,tree_children,tree_info);

  calcray<<<numBlocks, blockSize>>>(N,r,h,d_coldens,tree_info);
  cudaMemcpy(coldens, d_coldens, N*sizeof(float), cudaMemcpyDeviceToHost);
// 
  std::cout << "coldens0  " << coldens[0] << std::endl;
  
  // Free memory
  cudaFree(r);
  cudaFree(h);
  cudaFree(coldens);
  
  cudaFree(d_coldens);
  cudaFree(tree_children);
  cudaFree(tree_info);

  return 0;
}
