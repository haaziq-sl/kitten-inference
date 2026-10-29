// src/backend/backend.cpp
#include "backend.hpp"
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <cstdio>

static std::vector<BackendOps*>& registry() {
    static std::vector<BackendOps*> r;
    return r;
}

static BackendOps* s_active = nullptr;

void register_backend(BackendOps* ops) {
    registry().push_back(ops);
    // Always keep the highest-priority backend active.
    std::stable_sort(registry().begin(), registry().end(),
                     [](const BackendOps* a, const BackendOps* b){
                         return a->priority > b->priority;
                     });
    s_active = registry().front();
}

BackendOps* active_backend() {
    if (!s_active)
        throw std::runtime_error("No backend registered. "
            "Did you link cpu_backend.cpp (or another backend)?");
    return s_active;
}

BackendOps* backend_by_name(const char* name) {
    for (auto* b : registry())
        if (strcmp(b->name, name) == 0) return b;
    return nullptr;
}

bool set_active_backend(const char* name) {
    BackendOps* b = backend_by_name(name);
    if (!b) return false;
    s_active = b;
    return true;
}

void require_active_backend(const char* name) {
    if (!set_active_backend(name)) {
        fprintf(stderr, "Fatal: backend '%s' is not available "
                        "(not compiled in or not registered).\n", name);
        exit(1);
    }
    fprintf(stderr, "[backend] active: %s\n", name);
}
