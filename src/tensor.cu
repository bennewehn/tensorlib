#include <stdio.h>
#include "tensor.h"
#include "utils.h"
#include "math.h"

static Tensor *createTensor(const int *shape, int num_dims)
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

    if (tensor == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for Tensor.\n");
        return NULL;
    }

    // Allocate memory for shape array
    int *t_shape = (int *)malloc(sizeof(int) * num_dims);
    if (t_shape == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for shape array.\n");
        free(tensor); // Free previously allocated memory
        return NULL;
    }

    // Allocate memory for isOnGpu property
    bool *t_isOnGPU = (bool *)malloc(sizeof(bool));
    if (t_isOnGPU == NULL)
    {
        fprintf(stderr, "Failed to allocate memory.\n");
        free(t_shape); // Free previously allocated memory
        free(tensor);
        return NULL;
    }

    // Initialize elementsCount and copy shape dimensions
    int elementsCount = 1;
    for (int i = 0; i < num_dims; i++)
    {
        t_shape[i] = shape[i];
        elementsCount *= shape[i];
    }

    // Assign values to tensor structure
    tensor->num_dims = num_dims;
    tensor->num_elements = elementsCount;
    tensor->shape = t_shape;
    tensor->start_idx = 0;
    tensor->base = NULL;
    tensor->isOnGpu = t_isOnGPU;

    return tensor;
}

Tensor *createTensorCPU(const int *shape, int num_dims)
{
    Tensor *tensor = createTensor(shape, num_dims);

    // Check if tensor creation failed
    if (tensor == NULL)
        return NULL;

    // Allocate memory on the CPU
    float *data = (float *)malloc(tensor->num_elements * sizeof(float));
    if (data == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for tensor data on CPU.\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    tensor->data = data;
    *tensor->isOnGpu = false;

    return tensor;
}

Tensor *createTensorCPU(float *data, const int *shape, int num_dims)
{
    Tensor *t = createTensorCPU(shape, num_dims);
    if (t == NULL)
        return NULL;
    memcpy(t->data, data, t->num_elements * sizeof(float));
    return t;
}

Tensor *createTensorGPU(const int *shape, int num_dims)
{
    Tensor *tensor = createTensor(shape, num_dims);

    // Check if tensor creation failed
    if (tensor == NULL)
        return NULL;

    // Allocate memory on the GPU
    cudaError_t err = cudaMalloc(&(tensor->data), tensor->num_elements * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    *tensor->isOnGpu = true;

    return tensor;
}

Tensor *createTensorGPU(float *data, const int *shape, int num_dims)
{
    Tensor *t = createTensorGPU(shape, num_dims);
    if (t == NULL)
        return NULL;
    cudaMemcpy(t->data, data, t->num_elements * sizeof(float), cudaMemcpyHostToDevice);
    return t;
}

void freeTensor(Tensor *tensor)
{
    if (tensor)
    {
        // if tensor is not referencing other tensor
        if (tensor->base == NULL)
        {
            if (*tensor->isOnGpu)
            {
                cudaFree(tensor->data);
            }
            else
            {
                free(tensor->data);
            }
            free(tensor->isOnGpu);
        }
        free(tensor->shape);
        free(tensor);
    }
}

int moveTensorToGPU(Tensor *tensor)
{
    if (tensor->base != NULL)
    {
        fprintf(stderr, "Cannot move a reference tensor.");
        return -3;
    }

    if (*tensor->isOnGpu)
    {
        fprintf(stderr, "Tensor is already on GPU.\n");
        return -1;
    }

    int size = tensor->num_elements * sizeof(float);

    float *d_ptr;
    // Allocate memory on the GPU
    cudaError_t err = cudaMalloc(&(d_ptr), size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -2;
    }

    // Copy data from host to device
    cudaMemcpy(d_ptr, tensor->data, size, cudaMemcpyHostToDevice);

    // free host memory
    free(tensor->data);

    // Update tensor pointer
    tensor->data = d_ptr;

    *tensor->isOnGpu = true;

    return 0;
}

int moveTensorToCPU(Tensor *tensor)
{
    if (tensor->base != NULL)
    {
        fprintf(stderr, "Cannot move a reference tensor.");
        return -3;
    }

    if (!*tensor->isOnGpu)
    {
        fprintf(stderr, "Tensor is already on CPU.\n");
        return -1;
    }

    int size = tensor->num_elements * sizeof(float);

    // Allocate memory on the CPU
    float *h_data = (float *)malloc(size);
    if (h_data == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for tensor data on CPU.\n");
        return -2;
    }

    // Copy data from host to device
    cudaMemcpy(h_data, tensor->data, size, cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(tensor->data);

    // Update tensor pointer
    tensor->data = h_data;

    *tensor->isOnGpu = false;

    return 0;
}

// Function to copy data from GPU to CPU
void copyTensorFromGPU(const Tensor *tensor, float **hostData)
{
    int size = tensor->num_elements * sizeof(float);

    *hostData = (float *)malloc(size);

    cudaMemcpy(*hostData, &(tensor->data[tensor->start_idx]), size, cudaMemcpyDeviceToHost);
}

Tensor *getTensorByIdx(Tensor *t, const int *idx, int idx_len)
{
    // Indexer
    // 1. check if the given index is compatible with the tensor shape
    // a). condition: dims of index <= dims of tensor
    bool shape_incompatible = false;
    if (idx_len > t->num_dims)
        shape_incompatible = true;
    // b). condition: nth index has to be < nth shape
    for (int i = 0; i < idx_len; i++)
    {
        if (idx[i] >= t->shape[i])
        {
            shape_incompatible = true;
            break;
        }
    }

    if (shape_incompatible)
    {
        char *index = array_to_string(idx, idx_len);
        char *shape = array_to_string(t->shape, t->num_dims);
        fprintf(stderr, "Given index (%s) is incompatible with tensor shape (%s).\n", index, shape);
        free(index);
        free(shape);
        return NULL;
    }

    // Calculate start index
    // Ex. index = (1, 2)    shape = (2, 3, 4)
    // start_index = 1 * (3, 4)  +  2 * (4)
    // end_index = 1 * (3, 4)  +  (2+1) * (4)
    int start_idx = 0;
    for (int i = 0; i < idx_len; i++)
    {
        int product = 1;
        for (int j = i + 1; j < t->num_dims; j++)
        {
            product *= t->shape[j];
        }
        start_idx += idx[i] * product;
    }

    int num_dims = t->num_dims - idx_len;
    int *shape = &(t->shape[idx_len]);
    // if you index a single element
    if (idx_len == t->num_dims)
    {
        num_dims = 1;
        shape = (int *)malloc(sizeof(int));
        if (shape == NULL)
        {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }
        *shape = 1;
    }

    Tensor *res = createTensor(shape, num_dims);
    if (!res)
        return NULL;

    res->start_idx = start_idx;
    res->data = t->data;

    free(res->isOnGpu);
    res->isOnGpu = t->isOnGpu;

    res->base = t;

    return res;
}

void printTensorRecursive(const float *data, int mem_offset, const int *shape, int len_shape, int indent)
{
    printIndent(indent);

    printf("[");
    if (len_shape == 1)
    {
        for (int i = 0; i < shape[0]; ++i)
        {
            if (i > 0)
            {
                printf(", ");
            }
            printf("%.6f", data[i + mem_offset]);
        }
    }
    else
    {
        printf("\n");
        int elements_in_current_dim = 1;
        for (int i = 1; i < len_shape; ++i)
        {
            elements_in_current_dim *= shape[i];
        }
        for (int i = 0; i < shape[0]; ++i)
        {
            printTensorRecursive(data + i * elements_in_current_dim, mem_offset, shape + 1, len_shape - 1, indent + 1);
            if (i < shape[0] - 1)
            {
                printf(",\n"); // Line break and comma after each dimension except last element
            }
            else
            {
                printf("\n"); // Line break after the last dimension
            }
        }
        printIndent(indent);
    }
    printf("]");
}

void print(const Tensor *tensor)
{
    float *data = tensor->data;
    // Copy to CPU
    if (*(tensor->isOnGpu))
    {
        copyTensorFromGPU(tensor, &data);
        printTensorRecursive(data, 0, tensor->shape, tensor->num_dims, 0);
    }
    else
    {
        printTensorRecursive(data, tensor->start_idx, tensor->shape, tensor->num_dims, 0);
    }

    printf("\n");
}

void printShape(const Tensor *tensor)
{
    printf("(%s)\n", array_to_string(tensor->shape, tensor->num_dims));
}

__global__ void setTensorAtKernel(float *data, int index, float value)
{
    data[index] = value;
}

void setTensorAt(Tensor *tensor, const int *index, int len_index, float value)
{

    if (len_index != tensor->num_dims)
    {
        fprintf(stderr, "Index must have the same dimension as the tensor.\n");
        return;
    }

    int my_idx = 0;
    for (int i = 0; i < len_index; i++)
    {
        int product = 1;
        for (int j = i + 1; j < tensor->num_dims; j++)
        {
            product *= tensor->shape[j];
        }
        my_idx += index[i] * product;
    }

    if (*tensor->isOnGpu)
    {
        setTensorAtKernel<<<1, 1>>>(tensor->data, my_idx + tensor->start_idx, value);
        cudaDeviceSynchronize();
    }
    else
    {
        tensor->data[my_idx + tensor->start_idx] = value;
    }
}

__global__ void setTensorKernel(float *data, int num_elements, int mem_offset, float value)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        data[tid] = value;
    }
}

void setTensor(Tensor *tensor, float value)
{
    if (*tensor->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, setTensorKernel, 0, 0);

        int gridSize = (tensor->num_elements + blockSize - 1) / blockSize;

        setTensorKernel<<<gridSize, blockSize>>>(tensor->data, tensor->num_elements, tensor->start_idx, value);
        cudaDeviceSynchronize();
    }
    else
    {
        // serial way
        for (int i = 0; i < tensor->num_elements; i++)
        {
            tensor->data[i + tensor->start_idx] = value;
        }
    }
}

bool haveSameShape(const Tensor *t1, const Tensor *t2)
{
    if (t1->num_dims != t2->num_dims)
        return false;

    for (int i = 0; i < t1->num_dims; i++)
    {
        if (t1->shape[i] != t2->shape[i])
        {
            return false;
        }
    }
    return true;
}

bool ensureSameDevice(const Tensor *t1, const Tensor *t2)
{
    if (*t1->isOnGpu != *t2->isOnGpu)
    {
        fprintf(stderr, "Both tensors have to be on the same device.\n");
        return false;
    }
    return true;
}

bool ensureSameShape(const Tensor *t1, const Tensor *t2)
{
    if (!haveSameShape(t1, t2))
    {
        fprintf(stderr, "Both tensors must have the same shape.\n");
        return false;
    }
    return true;
}

__global__ void powTensorInPlaceKernel(float *data, int num_elements, int mem_offset, float exponent)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        data[tid + mem_offset] = powf(data[tid + mem_offset], exponent);
    }
}

void powTensorInPlace(Tensor *tensor, float exponent)
{
    if (*tensor->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, powTensorInPlaceKernel, 0, 0);

        int gridSize = (tensor->num_elements + blockSize - 1) / blockSize;

        powTensorInPlaceKernel<<<gridSize, blockSize>>>(tensor->data, tensor->num_elements, tensor->start_idx, exponent);
        cudaDeviceSynchronize();
    }
    else
    {
        // serial way
        for (int i = 0; i < tensor->num_elements; i++)
        {
            tensor->data[i + tensor->start_idx] = powf(tensor->data[i + tensor->start_idx], exponent);
        }
    }
}

__global__ void powTensorKernel(float *d_data, float *d_out, int exponent, int offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_out[tid] = powf(d_data[tid + offset], exponent); 
    }
}


Tensor *powTensor(const Tensor *t1, float exponent)
{

    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, powTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        Tensor *t = createTensorGPU(t1->shape, t1->num_dims);
        if(!t) return NULL;

        powTensorKernel<<<gridSize, blockSize>>>(t1->data, t->data, exponent, t1->start_idx, t1->num_elements);
        cudaDeviceSynchronize();

        return t;
    }
    else
    {
        Tensor *t = createTensorCPU(t1->shape, t1->num_dims);
        if(!t) return NULL;

        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t->data[i] = powf(t1->data[i + t1->start_idx], exponent);
        }

        return t;
    }
}


__global__ void addInPlaceTensorKernel(float *d_data1, float *d_data2, int t1_offset, int t2_offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_data1[tid + t1_offset] += d_data2[tid + t2_offset];
    }
}

// Adds t2 to t1.
void addTensorInPlace(Tensor *t1, Tensor *t2)
{

    if (!(ensureSameDevice(t1, t2), ensureSameShape(t1, t2)))
        return;

    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, addInPlaceTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        addInPlaceTensorKernel<<<gridSize, blockSize>>>(t1->data, t2->data, t1->start_idx, t2->start_idx, t1->num_elements);
        cudaDeviceSynchronize();
    }
    else
    {
        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t1->data[i + t1->start_idx] += t2->data[i + t2->start_idx];
        }
    }
}

__global__ void addTensorKernel(float *d_data1, float *d_data2, float *d_out, int t1_offset, int t2_offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_out[tid] = d_data1[tid + t1_offset] + d_data2[tid + t2_offset];
    }
}


Tensor *addTensor(const Tensor *t1, const Tensor *t2)
{
    if (!(ensureSameDevice(t1, t2), ensureSameShape(t1, t2)))
        return NULL;


    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, addTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        Tensor *t = createTensorGPU(t1->shape, t1->num_dims);
        if(!t) return NULL;

        addTensorKernel<<<gridSize, blockSize>>>(t1->data, t2->data, t->data, t1->start_idx, t2->start_idx, t1->num_elements);
        cudaDeviceSynchronize();

        return t;
    }
    else
    {
        Tensor *t = createTensorCPU(t1->shape, t1->num_dims);
        if(!t) return NULL;

        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t->data[i] = t1->data[i + t1->start_idx] + t2->data[i + t2->start_idx];
        }

        return t;
    }
}

__global__ void multiplyInPlaceTensorKernel(float *d_data1, float *d_data2, int t1_offset, int t2_offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_data1[tid + t1_offset] *= d_data2[tid + t2_offset];
    }
}

// Mulitiplies t2 to t1.
void multiplyTensorInPlace(Tensor *t1, Tensor *t2)
{

    if (!(ensureSameDevice(t1, t2), ensureSameShape(t1, t2)))
        return;

    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiplyInPlaceTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        multiplyInPlaceTensorKernel<<<gridSize, blockSize>>>(t1->data, t2->data, t1->start_idx, t2->start_idx, t1->num_elements);
        cudaDeviceSynchronize();
    }
    else
    {
        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t1->data[i + t1->start_idx] *= t2->data[i + t2->start_idx];
        }
    }
}

__global__ void multiplyTensorKernel(float *d_data1, float *d_data2, float *d_out, int t1_offset, int t2_offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_out[tid] = d_data1[tid + t1_offset] * d_data2[tid + t2_offset];
    }
}

Tensor *multiplyTensor(const Tensor *t1, const Tensor *t2)
{
    if (!(ensureSameDevice(t1, t2), ensureSameShape(t1, t2)))
        return NULL;


    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiplyTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        Tensor *t = createTensorGPU(t1->shape, t1->num_dims);
        if(!t) return NULL;

        multiplyTensorKernel<<<gridSize, blockSize>>>(t1->data, t2->data, t->data, t1->start_idx, t2->start_idx, t1->num_elements);
        cudaDeviceSynchronize();

        return t;
    }
    else
    {
        Tensor *t = createTensorCPU(t1->shape, t1->num_dims);
        if(!t) return NULL;

        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t->data[i] = t1->data[i + t1->start_idx] * t2->data[i + t2->start_idx];
        }

        return t;
    }
}

__global__ void divideInPlaceTensorKernel(float *d_data1, float *d_data2, int t1_offset, int t2_offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_data1[tid + t1_offset] /= d_data2[tid + t2_offset];
    }
}

// Divides t1 by t2. t1 /= t2
void divideTensorInPlace(Tensor *t1, Tensor *t2)
{

    if (!(ensureSameDevice(t1, t2), ensureSameShape(t1, t2)))
        return;

    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, divideInPlaceTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        divideInPlaceTensorKernel<<<gridSize, blockSize>>>(t1->data, t2->data, t1->start_idx, t2->start_idx, t1->num_elements);
        cudaDeviceSynchronize();
    }
    else
    {
        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t1->data[i + t1->start_idx] /= t2->data[i + t2->start_idx];
        }
    }
}

__global__ void divideTensorKernel(float *d_data1, float *d_data2, float *d_out, int t1_offset, int t2_offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_out[tid] = d_data1[tid + t1_offset] / d_data2[tid + t2_offset];
    }
}

Tensor *divideTensor(const Tensor *t1, const Tensor *t2)
{
    if (!(ensureSameDevice(t1, t2), ensureSameShape(t1, t2)))
        return NULL;


    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, divideTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        Tensor *t = createTensorGPU(t1->shape, t1->num_dims);
        if(!t) return NULL;

        divideTensorKernel<<<gridSize, blockSize>>>(t1->data, t2->data, t->data, t1->start_idx, t2->start_idx, t1->num_elements);
        cudaDeviceSynchronize();

        return t;
    }
    else
    {
        Tensor *t = createTensorCPU(t1->shape, t1->num_dims);
        if(!t) return NULL;

        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t->data[i] = t1->data[i + t1->start_idx] / t2->data[i + t2->start_idx];
        }

        return t;
    }
}

__global__ void subInPlaceTensorKernel(float *d_data1, float *d_data2, int t1_offset, int t2_offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_data1[tid + t1_offset] -= d_data2[tid + t2_offset];
    }
}

// Subtract t1 by t2. t1 /= t2
void subTensorInPlace(Tensor *t1, Tensor *t2)
{

    if (!(ensureSameDevice(t1, t2), ensureSameShape(t1, t2)))
        return;

    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, subInPlaceTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        subInPlaceTensorKernel<<<gridSize, blockSize>>>(t1->data, t2->data, t1->start_idx, t2->start_idx, t1->num_elements);
        cudaDeviceSynchronize();
    }
    else
    {
        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t1->data[i + t1->start_idx] -= t2->data[i + t2->start_idx];
        }
    }
}

__global__ void subTensorKernel(float *d_data1, float *d_data2, float *d_out, int t1_offset, int t2_offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_out[tid] = d_data1[tid + t1_offset] - d_data2[tid + t2_offset];
    }
}

Tensor *subTensor(const Tensor *t1, const Tensor *t2)
{
    if (!(ensureSameDevice(t1, t2), ensureSameShape(t1, t2)))
        return NULL;


    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, subTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        Tensor *t = createTensorGPU(t1->shape, t1->num_dims);
        if(!t) return NULL;

        subTensorKernel<<<gridSize, blockSize>>>(t1->data, t2->data, t->data, t1->start_idx, t2->start_idx, t1->num_elements);
        cudaDeviceSynchronize();

        return t;
    }
    else
    {
        Tensor *t = createTensorCPU(t1->shape, t1->num_dims);
        if(!t) return NULL;

        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t->data[i] = t1->data[i + t1->start_idx] - t2->data[i + t2->start_idx];
        }

        return t;
    }
}

__global__ void expInPlaceTensorKernel(float *d_data, int offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_data[tid + offset] = expf(d_data[tid + offset]);
    }
}

// Exponential in place
void expTensorInPlace(Tensor *t1)
{
    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, expInPlaceTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        expInPlaceTensorKernel<<<gridSize, blockSize>>>(t1->data, t1->start_idx, t1->num_elements);
        cudaDeviceSynchronize();
    }
    else
    {
        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t1->data[i + t1->start_idx] = expf(t1->data[i + t1->start_idx]);
        }
    }
}

__global__ void expTensorKernel(float *d_data, float *d_out, int offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_out[tid] = expf(d_data[tid + offset]);
    }
}

Tensor *expTensor(const Tensor *tensor)
{
    if (*tensor->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, expTensorKernel, 0, 0);

        int gridSize = (tensor->num_elements + blockSize - 1) / blockSize;

        Tensor *t = createTensorGPU(tensor->shape, tensor->num_dims);
        if(!t) return NULL;

        expTensorKernel<<<gridSize, blockSize>>>(tensor->data, t->data, tensor->start_idx, tensor->num_elements);
        cudaDeviceSynchronize();

        return t;
    }
    else
    {
        Tensor *t = createTensorCPU(tensor->shape, tensor->num_dims);
        if(!t) return NULL;

        // serial way
        for (int i = 0; i < tensor->num_elements; i++)
        {
            t->data[i] = expf(tensor->data[i + tensor->start_idx]);
        }

        return t;
    }
}

__global__ void logInPlaceTensorKernel(float *d_data, int offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_data[tid + offset] = logf(d_data[tid + offset]);
    }
}

// Log in place
void logTensorInPlace(Tensor *t1)
{
    if (*t1->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, logInPlaceTensorKernel, 0, 0);

        int gridSize = (t1->num_elements + blockSize - 1) / blockSize;

        logInPlaceTensorKernel<<<gridSize, blockSize>>>(t1->data, t1->start_idx, t1->num_elements);
        cudaDeviceSynchronize();
    }
    else
    {
        // serial way
        for (int i = 0; i < t1->num_elements; i++)
        {
            t1->data[i + t1->start_idx] = logf(t1->data[i + t1->start_idx]);
        }
    }
}

__global__ void logTensorKernel(float *d_data, float *d_out, int offset, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        d_out[tid] = logf(d_data[tid + offset]);
    }
}

Tensor *logTensor(const Tensor *tensor)
{
    if (*tensor->isOnGpu)
    {
        int minGridSize;
        int blockSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, logTensorKernel, 0, 0);

        int gridSize = (tensor->num_elements + blockSize - 1) / blockSize;

        Tensor *t = createTensorGPU(tensor->shape, tensor->num_dims);
        if(!t) return NULL;

        logTensorKernel<<<gridSize, blockSize>>>(tensor->data, t->data, tensor->start_idx, tensor->num_elements);
        cudaDeviceSynchronize();

        return t;
    }
    else
    {
        Tensor *t = createTensorCPU(tensor->shape, tensor->num_dims);
        if(!t) return NULL;

        // serial way
        for (int i = 0; i < tensor->num_elements; i++)
        {
            t->data[i] = logf(tensor->data[i + tensor->start_idx]);
        }

        return t;
    }
}