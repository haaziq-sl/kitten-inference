#pragma once
#include "backend/backend.hpp"
#include <cuda_runtime.h>

struct CudaAllocator : Allocator {
    MemSpace mem_space() const override { return MemSpace::CUDA; }
    void* alloc(size_t n) override;
    void  free(void* p)  override;
    void  copy_h2d(void* dst, const void* src, size_t n) override;
    void  copy_d2h(void* dst, const void* src, size_t n) override;
    void  copy_d2d(void* dst, const void* src, size_t n) override;
    void  sync() override;
    void  zero_memory(void* ptr, size_t n) override;
};
