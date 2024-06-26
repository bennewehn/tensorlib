#include <stdio.h>
#include "tensor.h"

int main(){
    // Create Tensor on GPU with init values
    int shape2[] = {2, 3, 2};
    float arr2[2][3][2] = {
        {
            {1, 2},
            {3, 4},
            {5, 6}
        },
        {
            {7, 8},
            {9, 10},
            {11, 12}
        }
    };
    Tensor* t2 = createTensorCPU(&arr2[0][0][0], shape2, 3);

    int idx[] = {1};
    Tensor* ref = getTensorByIdx(t2, idx, 1);

    int myidx[] = {1, 1};
    setTensorAt(ref, myidx, 2, 200);

    printTensor(t2);

    Tensor *sum = addTensor(ref, ref);

    return 0;
}