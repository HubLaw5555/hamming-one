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

void allocate()
{
	gpuErrchk(cudaMalloc((void**)&gpu_sequences, N * LL * sizeof(unsigned char)));
	gpuErrchk(cudaMalloc((void**)&gpu_distances, H * N * sizeof(unsigned char)));
}
void zero_distances()
{
	unsigned char* temp2 = new unsigned char[H * N];
	memset(temp2, 0, H * N);
	gpuErrchk(cudaMemcpy(gpu_distances, temp2, H * N, cudaMemcpyHostToDevice));
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

void input_test_1()
{
	unsigned char temp[LL];
	for (int j = 0; j < LL; ++j)
		temp[j] = (unsigned char)rand();
	temp[0] = 0;

	for (int i = 0; i < N; i++)
	{
		if (i == 2)
			temp[0] = 1;

		/*disp_char(temp[0]);
		disp_char(temp[1]);
		disp_char(temp[2]);
		disp_char(temp[3]);
		disp_char(temp[4]);
		printf("\n");*/

		gpuErrchk(cudaMemcpy(gpu_sequences + i*LL, temp, LL, cudaMemcpyHostToDevice));
	}
	zero_distances();
}

void input_data()
{
	unsigned char* temp = new unsigned char[LL*N];
	for (int j = 0; j < LL * N; ++j)
		temp[j] = (unsigned char)rand();
	gpuErrchk(cudaMemcpy(gpu_sequences, temp, LL*N, cudaMemcpyHostToDevice));
	zero_distances();
}


__global__ void mem_init(unsigned char* seqs, unsigned char* distances)
{
	int block_id = blockIdx.x;
	int th_id = threadIdx.x;
	int warp_id = th_id % 32;

	__shared__ unsigned int sum;
	sum = 0;

	__syncthreads();

	for (int h = 0; h < H; ++h)
	{
		if (block_id > h)
		{
			// inside 128 threads of block block_id
			unsigned char c = seqs[h * LL + th_id] ^ seqs[block_id * LL + th_id];

			for (int k = 0; k < 8; ++k)
			{
				sum += ((c >> k) & 0x0001);
			}
		}
		__syncthreads();
		if (th_id == 0)
		{
			distances[h * N + block_id] = sum;
		}
	}
	__syncthreads();
	if (block_id == 0 && th_id == 0)
	{
		printf("(0,1) = %d, (1,2) = %d\n", distances[1], distances[N + 1]);
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
					}
				}
			}
		}
		__syncthreads();
	}
	__syncthreads();
}

int main()
{
	gpuErrchk(cudaSetDevice(0));

	allocate();
	input_test_1();
	printf("Kernel start!\n");
	mem_init << <N, 128 >> > (gpu_sequences, gpu_distances);
	checkCudaErrors(cudaDeviceSynchronize());
	hamming_one << <N, BLOCK_TH >> > (gpu_sequences, gpu_distances);
	checkCudaErrors(cudaDeviceSynchronize());
}
