#ifndef TENSOR_H
#define TENSOR_H

typedef struct Tensor Tensor;

struct Tensor{
    int *shape;
    int num_dims;
    int num_elements;
    // Which section of the data to consider.
    int start_idx;
    float *data;
    bool* isOnGpu;
    Tensor* base;
};


Tensor* createTensorGPU(const int *shape, int num_dims);
Tensor* createTensorGPU(float *data, const int *shape, int num_dims);
Tensor* createTensorCPU(const int *shape, int num_dims);
Tensor *createTensorCPU(float *data, const int *shape, int num_dims);
int moveTensorToGPU(Tensor* tensor);
int moveTensorToCPU(Tensor* tensor);
void printTensor(const Tensor *t);
Tensor *getTensorByIdx(Tensor *t, const int *idx, int idx_len);
void freeTensor(Tensor *t);
void setTensorAt(Tensor *tensor, const int *index, int len_index, float value);

void setTensor(Tensor *tensor, float value);

bool haveSameShape(const Tensor *t1, const Tensor *t2);

void printShape(const Tensor *tensor);

void addTensorInPlace(Tensor *t1, Tensor *t2);
Tensor *addTensor(const Tensor *t1, const Tensor *t2);

void multiplyTensorInPlace(Tensor *t1, Tensor *t2);
Tensor *multiplyTensor(const Tensor *t1, const Tensor *t2);


#endif