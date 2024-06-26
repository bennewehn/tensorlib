#include <stdio.h>

static void printCUDADeviceProperties(cudaDeviceProp *device_properties){
    // Print device properties
    printf("Device Name: %s\n", device_properties->name);
    printf("Total Global Memory: %zu bytes\n", device_properties->totalGlobalMem);
    printf("Shared Memory per Block: %zu bytes\n", device_properties->sharedMemPerBlock);
    printf("Registers per Block: %d\n", device_properties->regsPerBlock);
    printf("Warp Size: %d\n", device_properties->warpSize);
    printf("Memory Pitch: %zu bytes\n", device_properties->memPitch);
    printf("Max Threads per Block: %d\n", device_properties->maxThreadsPerBlock);
    printf("Max Threads Dimension: [%d, %d, %d]\n", device_properties->maxThreadsDim[0], device_properties->maxThreadsDim[1], device_properties->maxThreadsDim[2]);
    printf("Max Grid Size: [%d, %d, %d]\n", device_properties->maxGridSize[0], device_properties->maxGridSize[1], device_properties->maxGridSize[2]);
    printf("Clock Rate: %d kHz\n", device_properties->clockRate);
    printf("Total Constant Memory: %zu bytes\n", device_properties->totalConstMem);
    printf("Compute Capability: %d->%d\n", device_properties->major, device_properties->minor);
    printf("Texture Alignment: %zu bytes\n", device_properties->textureAlignment);
    printf("Device Overlap: %d\n", device_properties->deviceOverlap);
    printf("Multiprocessor Count: %d\n", device_properties->multiProcessorCount);
    printf("Kernel Execution Timeout: %d\n", device_properties->kernelExecTimeoutEnabled);
    printf("Integrated: %d\n", device_properties->integrated);
    printf("Can Map Host Memory: %d\n", device_properties->canMapHostMemory);
    printf("Compute Mode: %d\n", device_properties->computeMode);
    printf("Concurrent Kernels: %d\n", device_properties->concurrentKernels);
    printf("ECC Enabled: %d\n", device_properties->ECCEnabled);
    printf("PCI Bus ID: %d\n", device_properties->pciBusID);
    printf("PCI Device ID: %d\n", device_properties->pciDeviceID);
    printf("PCI Domain ID: %d\n", device_properties->pciDomainID);
    printf("TCC Driver: %d\n", device_properties->tccDriver);
    printf("Async Engine Count: %d\n", device_properties->asyncEngineCount);
    printf("Unified Addressing: %d\n", device_properties->unifiedAddressing);
    printf("Memory Clock Rate: %d kHz\n", device_properties->memoryClockRate);
    printf("Memory Bus Width: %d bits\n", device_properties->memoryBusWidth);
    printf("L2 Cache Size: %d bytes\n", device_properties->l2CacheSize);
    printf("Max Threads per Multiprocessor: %d\n", device_properties->maxThreadsPerMultiProcessor);
    printf("Stream Priorities Supported: %d\n", device_properties->streamPrioritiesSupported);
    printf("Global L1 Cache Supported: %d\n", device_properties->globalL1CacheSupported);
    printf("Local L1 Cache Supported: %d\n", device_properties->localL1CacheSupported);
    printf("Max Shared Memory per Multiprocessor: %zu bytes\n", device_properties->sharedMemPerMultiprocessor);
    printf("Max Registers per Multiprocessor: %d\n", device_properties->regsPerMultiprocessor);
    printf("Managed Memory: %d\n", device_properties->managedMemory);
    printf("Is Multi-GPU Board: %d\n", device_properties->isMultiGpuBoard);
    printf("Multi-GPU Board Group ID: %d\n", device_properties->multiGpuBoardGroupID);
}

void printCurrentDeviceInformation(){
    // Get the current device
    int current_device;
    cudaGetDevice(&current_device);

    // Get device properties
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, current_device);

    printCUDADeviceProperties(&device_properties);
}