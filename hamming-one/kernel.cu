#include "global.cuh"
#include <cstring>

// gpu sequences storage
//unsigned char** gpu_sequences;
//unsigned char** gpu_distances;
unsigned char* gpu_sequences;
unsigned char* gpu_distances;

// upper-triangle array of discances beetwen sequences
// distance may be either 0 or 1 because higher values are rejected
//unsigned char** distances;

//
void allocate()
{
	gpuErrchk(cudaMalloc((void**)&gpu_sequences, N * LL * sizeof(unsigned char)));
	//for (int i = 0; i < N; ++i)
	//{
	//	gpuErrchk(cudaMalloc((void**)&gpu_sequences[i], LL * sizeof(unsigned char)));
	//	//checkCudaErrors(cudaMalloc((void**)&distances[i], (N - i)*sizeof(unsigned char)));
	//}

	//gpu_distances = new unsigned char*[H];
	gpuErrchk(cudaMalloc((void**)&gpu_distances, H * N * sizeof(unsigned char)));
	/*for (int i = 0; i < H; ++i)
	{
		gpuErrchk(cudaMalloc((void**)&gpu_distances, N * sizeof(unsigned char)));
	}*/
}

void input_data()
{
	//unsigned char* the_same = new unsigned char[LL];
	//for (int j = 0; j < LL; ++j)
	//	the_same[j] = (unsigned char)rand();
	//the_same[0] = 0;

	//for (int i = 0; i < N; ++i)
	//{
	//	/*unsigned char* cpu_sample = new unsigned char[LL];
	//	for (int j = 0; j < LL; ++j)
	//	{
	//		if (i < 3)
	//		{
	//			cpu_sample[j] = the_same[j];
	//		}
	//		else
	//		{
	//			cpu_sample[j] = (unsigned char)rand();
	//		}
	//	}
	//	the_same[0]++;*/
	//	if (i == 2)
	//		the_same[0] = 1;
	//	//memcpy(gpu_sequences[i], the_same, LL);
	//	gpuErrchk(cudaMemcpy(gpu_sequences + i * LL, the_same, LL, cudaMemcpyHostToDevice));
	//}
	unsigned char* temp = new unsigned char[LL*N];
	for (int j = 0; j < LL * N; ++j)
		temp[j] = (unsigned char)rand();
	gpuErrchk(cudaMemcpy(gpu_sequences, temp, LL*N, cudaMemcpyHostToDevice));

	unsigned char* temp2 = new unsigned char[H * N];
	memset(temp2, 0, H * N);
	gpuErrchk(cudaMemcpy(gpu_distances, temp2, H * N, cudaMemcpyHostToDevice));
}

__device__ unsigned char sum_bits(unsigned char* seqs, int i, int j, int start/*, int slice*/)
{
	//extern __shared__ int shared_dist[];
	/*int i = blockIdx.x;
	int j = blockIdx.y;
	int k = threadIdx.x;

	if (j > i)
	{
		unsigned char c1 = (seqs[i][k / BYTE] >> k % BYTE) && 1;
		unsigned char c2 = (seqs[j][k / BYTE] >> k % BYTE) && 1;
		distances[i][j] += (c1 == c2);
	}*/
	/*unsigned char c1 = (seqs[i][k / BYTE] >> (7 - k % BYTE)) && 0x01;
	unsigned char c2 = (seqs[j][k / BYTE] >> (7 - k % BYTE)) && 0x01;
	return (c1 == c2);*/
	//printf("Sum bits start = %d, slice = %d\n", start, slice);
	int sum = 0;
	/*for (int it = slice * start; it < min(slice * start + slice, LL); ++it)
	{*/
	unsigned char c = seqs[i * N + start] ^ seqs[j * N + start];
	for (int k = 0; k < 8; ++k)
		sum += (c >> k) & 0x0001;
	//}
	return sum;
}

__device__ unsigned char sum_all_bits(unsigned char* seqs, int i, int j)
{
	int sum = 0;
	for (int it = 0; it < LL; ++it)
	{
		unsigned char c = seqs[i * N + it] ^ seqs[j * N + it];
		for (int k = 0; k < 8; ++k)
			sum += (c >> k) && 0x0001;
		if (sum >= 2)
			return 2;
	}
	return sum;
}

__global__ void mem_init(unsigned char* seqs, unsigned char* distances)
{
	//extern __shared__ int shared_dist[];

	// set zeros in shared mem
	// in algorithm it is to set them as 0 for [H] first places
	/*for (int i = 0; i < (SH_SIZE) / 4 ; i += BLOCK_TH)
	{
		int realIndex = i + threadIdx.x;
		if (realIndex < (SH_SIZE) / 4)
		{
			shared_dist[i] = 0x00000000;
		}
	}
	__syncthreads();*/

	// upper triangle matrix of distances
	// distances[i][j] refs dist(xi,xj) if i <= j
	// distances[i] has N - i elements;
	//unsigned char** distances;

	int block_id = blockIdx.x;
	int th_id = threadIdx.x;
	int warp_id = th_id % 32;
	if (block_id == 164)
	{
		printf("164 point one\n");
	}
	//int word_slice = LL / BLOCK_TH + 1;

	//if (th_id == 0)
	//{
	//	printf("START, block_id = %d\n", block_id);
	//}
	/*if (th_id < H)
	{
		distances[th_id] = (unsigned char*)&shared_dist[th_id * N/sizeof(int)];
	}*/
	/*for (int p = 0; p < N; p += BLOCK_TH)
	{
		int index = p * BLOCK_TH + th_id;
		int pos = (N - index + 1 + N) * index / 2;
		distances[p * BLOCK_TH + th_id] = (unsigned char*)&shared_dist[pos];
	}*/

	__syncthreads();


	//__shared__ unsigned char sum;
	//sum = 0;
	for (int h = 0; h < H; ++h)
	{
		/*if (th_id == 0)
		{
			distances[min(h, H - 1) * N + block_id] = 0;
		}*/
		//if (th_id == 0)
		//{
		//	sum = 0;
		//	//printf("Sum for h = %d\n", h);
		//}
		//printf("Here block %d, thread %d, h = %d\n", block_id, th_id, h);
		//__syncthreads();
		/*if (block_id == 164)
		{
			printf("164 point two, h %d, th %d\n", h, th_id);
		}*/
		if (block_id > min(h, H - 1))
		{
			//printf("DIFF th %d, bl %d, slice %d\n", th_id, block_id, word_slice);

			//sum += sum_bits(seqs, h, block_id, th_id/*, word_slice*/);
			if (block_id == 164)
			{
				printf("164 point three, h %d, \n");
			}
			unsigned char c = seqs[min(h, H - 1) * N + th_id] ^ seqs[block_id * N + th_id];

			for (int k = 0; k < 8; ++k)
			{
				if (block_id == 164)
				{
					printf("164 point four, h %d, th %d, k %d, c %d\n", h, th_id, k, c);
				}
				distances[min(h, H - 1) * N + block_id] += ((c >> k) & 0x0001);
			}
		}
		/*__syncthreads();
		if (th_id == 1)
		{
			printf("Just before h = %d\n", h);
			distances[h*N + block_id] = sum;
			printf("Here th %d, bl %d, h %d\n", th_id, block_id, h);
		}*/
		__syncthreads();
		if (block_id == 164)
		{
			printf("164 point three\n");
		}
	}
	//printf("DONE\n");
	__syncthreads();
	if (th_id == 0)
	{
		printf("DONE %d\n", block_id);
	}
}
__global__ void hamming_one(unsigned char* seqs, unsigned char* distances)
{
	int block_id = blockIdx.x;
	int th_id = threadIdx.x;
	int warp_id = th_id % 32;
	//int word_slice = LL / BLOCK_TH + 1;

	__syncthreads();

	if (block_id >= H)
	{
		for (int t = block_id + 1; t < N; t += BLOCK_TH)
		{
			int j = t + th_id;
			if (j < N)
			{
				bool valid = true;
				for (int d = 0; d < H; ++d)
				{
					if (abs(distances[d * N + block_id] - distances[d * N + j]) > 1)
					{
						valid = false;
						break;
					}
				}
				if (valid)
				{
					int onc_sum = 0;
					for (int it = 0; it < LL; ++it)
					{
						unsigned char c = seqs[block_id * N + it] ^ seqs[j * N + it];
						for (int k = 0; k < 8; ++k)
							onc_sum += (c >> k) & 0x0001;
						if (onc_sum >= 2)
							break;
					}

					if (/*sum_all_bits(seqs,block_id, j)*/ onc_sum == 1)
					{
						printf("(%d, %d) ", block_id, j);
					}
				}
			}
		}
		__syncthreads();
	}
	__syncthreads();
}



//__global__ void test_kernel()
//{
//	int id = threadIdx.x;
//	int t = 0;
//	if (id == 0)
//	{
//		for (int i = 0; i < 1000000000; ++i) {
//			t++; 
//			if (i % 100000000 == 0) {
//				printf("%d\n", i); //fflush(stdout);
//			}
//		}
//	}
//
//	//__syncthreads();
//	printf("Thread here %d, t = %d\n", id, t);
//	//fflush(stdout);
//}


int main()
{
	gpuErrchk(cudaSetDevice(0));

	allocate();
	input_data();
	//cudaDeviceProp* prop = new cudaDeviceProp();
	//cudaGetDeviceProperties(prop, 0);
	//printf("Shared mem per block: %d\n", prop->sharedMemPerBlock);
	//dim3 blocks = dim3(N, N);
	//int size = (N * (N + 1) * sizeof(unsigned char)) / 2;
	//int size = SH_SIZE / sizeof(int);
	mem_init << <N, 128 >> > (gpu_sequences, gpu_distances);
	//gpuErrchk(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	hamming_one << <N, BLOCK_TH >> > (gpu_sequences, gpu_distances);
	checkCudaErrors(cudaDeviceSynchronize());
	//test_kernel << <1, 64 >> > ();
}


//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}

//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
