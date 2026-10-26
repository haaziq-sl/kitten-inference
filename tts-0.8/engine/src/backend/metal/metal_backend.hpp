// src/backend/metal/metal_backend.hpp
// Pure C++ header — no ObjC, no Metal headers.
#pragma once
#include "backend/backend.hpp"

#include <functional>

struct MetalAllocator : Allocator {
    MemSpace mem_space() const override;
    void* alloc(size_t n) override;
    void  free(void* p)  override;
    void  copy_h2d(void* dst, const void* src, size_t n) override;
    void  copy_d2h(void* dst, const void* src, size_t n) override;
    void  copy_d2d(void* dst, const void* src, size_t n) override;
    void  sync() override;
};

// Runs fn inside an Objective-C autorelease pool, which is emptied when fn returns or throws (the exception is passed
// on). Metal work makes autoreleased objects on every call: command buffers, encoders, descriptors and names. A
// command buffer keeps every buffer it used alive, so without a pool that is emptied, a thread that never ends (such
// as Python's main thread) keeps all of them, and memory grows with every call.
void metal_autorelease_scope(const std::function<void()>& fn);
