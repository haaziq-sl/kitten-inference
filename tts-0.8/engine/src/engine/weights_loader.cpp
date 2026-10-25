#include "weights_loader.hpp"
#include "weight_cache.hpp"
#include "utils/file_open.hpp"
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <memory>

static void read_or_die(FILE* f, void* dst, size_t n, const char* what) {
    if (fread(dst, 1, n, f) != n)
        throw std::runtime_error(std::string("read error at ") + what);
}

void load_weights(const std::string& weights_path,
                  std::vector<std::unique_ptr<ILayer>>& layers,
                  WeightLoadMemoryStats* memory_stats)
{
    clear_packed_weight_cache();

    auto file_deleter = [](FILE* fp){ if (fp) fclose(fp); };
    std::unique_ptr<FILE, decltype(file_deleter)> f_owner(
        fopen_utf8(weights_path, "rb"), file_deleter);
    FILE* f = f_owner.get();
    if (!f)
        throw std::runtime_error("Cannot open weights: " + weights_path);

    char magic[4];
    read_or_die(f, magic, 4, "magic");
    if (memcmp(magic, "W1X1", 4) != 0)
        throw std::runtime_error("Bad magic in " + weights_path + " (expected W1X1)");

    uint32_t version, num_entries;
    read_or_die(f, &version,     4, "version");
    read_or_die(f, &num_entries, 4, "num_entries");

    struct IndexEntry {
        char     name[64];
        uint64_t offset;
        uint32_t nbytes;
        uint32_t _pad;
    };
    std::unordered_map<std::string, IndexEntry> index;
    index.reserve(num_entries);
    for (uint32_t i = 0; i < num_entries; ++i) {
        IndexEntry e;
        read_or_die(f, &e, sizeof(e), "index_entry");
        e.name[63] = '\0';
        index[std::string(e.name)] = e;
    }

    // read_tensor captures f by reference — must be called synchronously within
    // this scope; layers must not store the lambda for deferred use.
    ReadTensorFn read_tensor = [&](const std::string& tensor_name,
                                    std::vector<int32_t>& shape_out,
                                    std::vector<uint8_t>& data_out) -> bool {
        auto it = index.find(tensor_name);
        if (it == index.end()) return false;
        const IndexEntry& e = it->second;
        if (fseek(f, (long)e.offset, SEEK_SET) != 0) return false;
        uint8_t dtype, ordering; uint16_t ndim;
        read_or_die(f, &dtype,    1, "tensor_dtype");
        read_or_die(f, &ordering, 1, "tensor_ordering");
        read_or_die(f, &ndim,     2, "tensor_ndim");
        shape_out.resize(ndim);
        if (ndim > 0)
            read_or_die(f, shape_out.data(), (size_t)ndim * 4, "tensor_shape");
        size_t data_bytes = (size_t)e.nbytes - 4 - (size_t)ndim * 4;
        data_out.resize(data_bytes);
        if (memory_stats && data_bytes > memory_stats->peak_tensor_bytes)
            memory_stats->peak_tensor_bytes = data_bytes;
        read_or_die(f, data_out.data(), data_bytes, "tensor_data");
        return true;
    };

    for (auto& layer : layers)
        layer->load_weights(read_tensor);
    clear_packed_weight_cache();

    // Explicitly close and check return value (fclose can fail on network filesystems).
    FILE* raw_f = f_owner.release();
    if (fclose(raw_f) != 0)
        fprintf(stderr, "Warning: fclose failed for %s: %s\n",
                weights_path.c_str(), strerror(errno));

    // printf("Loaded W1X1 weights from %s  (%u tensors)\n",
    //        weights_path.c_str(), num_entries);
}
