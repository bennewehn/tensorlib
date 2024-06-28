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
void print(const Tensor *t);
Tensor *getTensorByIdx(Tensor *t, const int *idx, int idx_len);
void freeTensor(Tensor *t);
void setTensorAt(Tensor *tensor, const int *index, int len_index, float value);

void setTensor(Tensor *tensor, float value);

bool haveSameShape(const Tensor *t1, const Tensor *t2);

void printShape(const Tensor *tensor);


// Binary Operators:

void addTensorInPlace(Tensor *t1, Tensor *t2);
Tensor *addTensor(const Tensor *t1, const Tensor *t2);

void multiplyTensorInPlace(Tensor *t1, Tensor *t2);
Tensor *multiplyTensor(const Tensor *t1, const Tensor *t2);

void divideTensorInPlace(Tensor *t1, Tensor *t2);
Tensor *divideTensor(const Tensor *t1, const Tensor *t2);

void subTensorInPlace(Tensor *t1, Tensor *t2);
Tensor *subTensor(const Tensor *t1, const Tensor *t2);

void powTensorInPlace(Tensor *tensor, float exponent);
Tensor *powTensor(const Tensor *t1, float exponent);


// Unary Operators:

void expTensorInPlace(Tensor *tensor);
Tensor *expTensor(const Tensor *tensor);

void logTensorInPlace(Tensor *tensor);
Tensor *logTensor(const Tensor *tensor);

Tensor *reluTensor(const Tensor *tensor);
void reluTensorInPlace(Tensor *t1);

#endif