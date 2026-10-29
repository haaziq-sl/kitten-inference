// src/backend/cpu/cpu_backend.hpp — CpuAllocator definition, shared by cpu_backend.cpp and model.cpp
#pragma once
#include "backend/backend.hpp"
#include <cstdlib>
#include <cstring>

struct CpuAllocator : Allocator {
    MemSpace mem_space()  const override { return MemSpace::Host; }
    void* alloc(size_t n) override { return ::malloc(n); }
    void  free (void* p)  override { ::free(p); }
    void  copy_h2d(void* d, const void* s, size_t n) override { ::memcpy(d, s, n); }
    void  copy_d2h(void* d, const void* s, size_t n) override { ::memcpy(d, s, n); }
    void  copy_d2d(void* d, const void* s, size_t n) override { ::memcpy(d, s, n); }
    void  sync() override {}
};
