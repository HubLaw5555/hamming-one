#include "global.cuh"
#include <cstring>

// gpu sequences storage
//unsigned char** gpu_sequences;
//unsigned char** gpu_distances;
unsigned char* gpu_sequences;
unsigned char* gpu_distances;
unsigned char** cpu_sequences;
__device__ unsigned int hamming_pairs = 0;



cudaEvent_t start_kernels, stop_kernels;
cudaEvent_t start_cpu_gpu, stop_cpu_gpu;

float time_kernels = .0f, time_cpu_gpu = .0f;

// upper-triangle array of discances beetwen sequences
// distance may be either 0 or 1 because higher values are rejected
//unsigned char** distances;

void allocate()
{
	if (CPU_CALL)
	{
		cpu_sequences = new unsigned char* [N];
		for (int i = 0; i < N; ++i)
		{
			cpu_sequences[i] = new unsigned char[LL];
		}
	}
	else
	{
		gpuErrchk(cudaMalloc((void**)&gpu_sequences, N * LL * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&gpu_distances, H * N * sizeof(unsigned char)));
	}
}

void disp_char(unsigned char c)
{
	printf("#");
	for (int i = 0; i < BYTE; ++i)
	{
		printf("%d", ((c >> (BYTE - i - 1)) & 1));
	}
	printf(" ");
}

// all the same but first two chars are 0,0,1,1,1,1,1,1,1...
void input_test_1()
{
	unsigned char temp[LL];
	for (int j = 0; j < LL; ++j)
		temp[j] = (unsigned char)rand();
	temp[0] = 0;

	cudaEventRecord(start_cpu_gpu, 0);
	for (int i = 0; i < N; i++)
	{
		if (i == 2)
			temp[0] = 1;

		gpuErrchk(cudaMemcpy(gpu_sequences + i * LL, temp, LL, cudaMemcpyHostToDevice));
	}
	cudaEventRecord(stop_cpu_gpu, 0);
	cudaEventSynchronize(stop_cpu_gpu);
	cudaEventElapsedTime(&time_cpu_gpu, start_cpu_gpu, stop_cpu_gpu);
}


// first bytes are 0,1,2,3,4,5...
void input_test_2()
{
	unsigned char temp[LL];
	for (int j = 0; j < LL; ++j)
		temp[j] = (unsigned char)rand();

	cudaEventRecord(start_cpu_gpu, 0);
	for (int i = 0; i < N; i++)
	{
		temp[0] = i;
		gpuErrchk(cudaMemcpy(gpu_sequences + i * LL, temp, LL, cudaMemcpyHostToDevice));
	}
	cudaEventRecord(stop_cpu_gpu, 0);
	cudaEventSynchronize(stop_cpu_gpu);
	cudaEventElapsedTime(&time_cpu_gpu, start_cpu_gpu, stop_cpu_gpu);
}


void random_data()
{

	if (CPU_CALL)
	{
		unsigned char* t_dat = new unsigned char[LL];
		for (int i = 0; i < LL; ++i)
			t_dat[i] = (unsigned char)rand();

		for (int i = 0; i < N; ++i)
		{
			memcpy(cpu_sequences[i], t_dat, LL);
		}
	}
	else
	{
		unsigned char* temp = new unsigned char[LL * N];
		for (int j = 0; j < LL * N; ++j)
			temp[j] = (unsigned char)rand();

		cudaEventRecord(start_cpu_gpu, 0);
		gpuErrchk(cudaMemcpy(gpu_sequences, temp, LL * N, cudaMemcpyHostToDevice));
		cudaEventRecord(stop_cpu_gpu, 0);
		cudaEventSynchronize(stop_cpu_gpu);
		cudaEventElapsedTime(&time_cpu_gpu, start_cpu_gpu, stop_cpu_gpu);
	}
}

void cpu_hamming_one()
{
	unsigned char** dists = new unsigned char* [H];
	for (int i = 0; i < H; ++i)
	{
		dists[i] = new unsigned char[N];
		memset(dists[i], 0, N);
	}
	for (int h = 0; h < H; ++h)
	{
		for (int j = h + 1; j < N; ++j)
		{
			int sum = 0;
			for (int i = 0; i < LL; ++i)
			{
				unsigned char c = cpu_sequences[h][i] ^ cpu_sequences[j][i];
				for (int k = 0; k < BYTE; ++k)
				{
					sum += ((c >> k) & 0x0001);
				}
				if (sum >= 2) break;
			}
			if (sum == 1)
			{
				printf("(%d, %d) ", h, j);
			}
		}
	}
	for (int i = H; i < N; ++i)
	{
		for (int j = i + 1; j < N; ++j)
		{
			bool isGood = true;
			for (int d = 0; d < H; ++d)
			{
				if (abs(dists[d][i] - dists[d][j]) > 1)
				{
					isGood = false;
					break;
				}
			}
			if (isGood)
			{
				int sum = 0;
				for (int p = 0; p < LL; ++p)
				{
					unsigned char c = cpu_sequences[i][p] ^ cpu_sequences[j][p];
					for (int k = 0; k < BYTE; ++k)
					{
						sum += ((c >> k) & 0x0001);
					}
					if (sum >= 2) break;
				}
				if (sum == 1)
				{
					printf("(%d, %d) ", i, j);
				}

			}
		}
	}
}


__global__ void mem_init(unsigned char* seqs, unsigned char* distances)
{
	int block_id = blockIdx.x;
	int th_id = threadIdx.x;
	int warp_id = th_id % 32;

	__shared__ unsigned int sum;
	__syncthreads();

	for (int h = 0; h < H; ++h)
	{
		sum = 0;
		if (block_id > h)
		{
			// inside 128 threads of block block_id
			unsigned char c = seqs[h * LL + th_id] ^ seqs[block_id * LL + th_id];

			for (int k = 0; k < BYTE; ++k)
			{
				atomicAdd(&sum, ((c >> k) & 0x0001));
			}
		}

		__syncthreads();
		if (th_id == 0)
		{
			distances[h * N + block_id] = (unsigned char)min(255, sum);
			if (distances[h * N + block_id] == 1)
			{
				printf("(%d, %d) ", h, block_id);

				atomicAdd(&hamming_pairs, 1);
			}
		}
		__syncthreads();
	}
	__syncthreads();
}
__global__ void hamming_one(unsigned char* seqs, unsigned char* distances)
{
	int block_id = blockIdx.x;
	int th_id = threadIdx.x;
	int warp_id = th_id % 32;

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
						unsigned char c = seqs[block_id * LL + it] ^ seqs[j * LL + it];
						for (int k = 0; k < 8; ++k)
							onc_sum += (c >> k) & 0x0001;
						if (onc_sum >= 2)
							break;
					}

					if (onc_sum == 1)
					{
						printf("(%d, %d) ", block_id, j);
						atomicAdd(&hamming_pairs, 1);
					}
				}
			}
		}
		__syncthreads();
	}
	__syncthreads();
}
__global__ void disp() { printf("\n\nTotal hamming pairs: %d\n", hamming_pairs); }

int main()
{
	gpuErrchk(cudaSetDevice(0));

	cudaEventCreate(&start_kernels);
	cudaEventCreate(&stop_kernels);
	cudaEventCreate(&start_cpu_gpu);
	cudaEventCreate(&stop_cpu_gpu);

	allocate();
	input_test_1();
	printf("Kernel start!\n");
	if (!CPU_CALL)
	{
		cudaEventRecord(start_kernels, 0);

		mem_init << <N, LL >> > (gpu_sequences, gpu_distances);
		checkCudaErrors(cudaDeviceSynchronize());
		hamming_one << <N, BLOCK_TH >> > (gpu_sequences, gpu_distances);
		checkCudaErrors(cudaDeviceSynchronize());
		disp << <1, 1 >> > ();
		checkCudaErrors(cudaDeviceSynchronize());

		cudaEventRecord(stop_kernels, 0);
		cudaEventSynchronize(stop_kernels);
		cudaEventElapsedTime(&time_kernels, start_kernels, stop_kernels);

		printf("Kernels execution time: %3f ms\n", time_kernels);
		printf("CPU -> GPU copy time: %3f ms\n", time_cpu_gpu);

		cudaEventDestroy(start_kernels);
		cudaEventDestroy(stop_kernels);
		cudaEventDestroy(start_cpu_gpu);
		cudaEventDestroy(stop_cpu_gpu);
	}
	else
	{
		cpu_hamming_one();
	}
}
