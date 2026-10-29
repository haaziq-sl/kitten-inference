// src/engine/buffer.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <functional>

// ─────────────────────────────────────────────────────────────────
// MemSpace: which address space does a pointer belong to?
// ─────────────────────────────────────────────────────────────────
enum class MemSpace : uint8_t {
    Host  = 0,   // CPU RAM — malloc / std::vector::data()
    CUDA  = 1,   // cudaMalloc device pointer
    HIP   = 2,   // hipMalloc device pointer
    Metal = 3,   // MTL::Buffer* cast to void* (shared or private storage)
};

// ─────────────────────────────────────────────────────────────────
// BufferView: non-owning view of any buffer.
// Replaces the raw void* passed to ILayer::forward().
// ─────────────────────────────────────────────────────────────────
struct BufferView {
    void*    ptr   = nullptr;
    MemSpace where = MemSpace::Host;
    size_t   bytes = 0;

    bool   valid()   const { return ptr != nullptr; }
    bool   is_host() const { return where == MemSpace::Host; }
    bool   is_gpu()  const { return where != MemSpace::Host; }

    template<typename T>       T* as()       { return static_cast<T*>(ptr); }
    template<typename T> const T* as() const { return static_cast<const T*>(ptr); }
};

// ─────────────────────────────────────────────────────────────────
// Buffer: owning buffer in any memory space.
// The deleter is injected at construction, so Buffer knows
// nothing about the GPU API — it just calls the right free() on
// destruction. Moveable, non-copyable.
// ─────────────────────────────────────────────────────────────────
struct Buffer {
    BufferView                    view;
    std::function<void(void*)> deleter;

    Buffer() = default;

    Buffer(void* p, size_t bytes, MemSpace where,
                 std::function<void(void*)> del)
        : view{p, where, bytes}, deleter(std::move(del)) {}

    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& o) noexcept
        : view(o.view), deleter(std::move(o.deleter)) {
        o.view = {};
    }
    Buffer& operator=(Buffer&& o) noexcept {
        if (this != &o) { release(); view = o.view; deleter = std::move(o.deleter); o.view = {}; }
        return *this;
    }

    ~Buffer() { release(); }

    void*    ptr()   const { return view.ptr; }
    MemSpace where() const { return view.where; }
    size_t   bytes() const { return view.bytes; }
    bool     valid() const { return view.ptr != nullptr; }

    template<typename T>       T* as()       { return view.as<T>(); }
    template<typename T> const T* as() const { return view.as<T>(); }

    // Manually release before destructor (useful to resize a slot)
    void release() {
        if (view.ptr && deleter) { deleter(view.ptr); view = {}; }
    }
};
