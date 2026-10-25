// src/utils/download.hpp — header-only weight download utility
//
// ensure_file(local_path, url):
//   If local_path does not exist, download it from url using curl.
//   Returns true on success, false on failure.
//   Creates parent directories as needed.

#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <filesystem>

inline bool validate_w1x1_file(const std::string& path, std::string* reason = nullptr)
{
    namespace fs = std::filesystem;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        if (reason) *reason = "cannot open file";
        return false;
    }

    auto close_and_return = [&](bool ok, const std::string& why = "") {
        fclose(f);
        if (!ok && reason) *reason = why;
        return ok;
    };

    char magic[4];
    if (fread(magic, 1, 4, f) != 4)
        return close_and_return(false, "file is too small");

    // Not every downloaded .bin is a W1X1 weight archive; leave raw assets alone.
    if (memcmp(magic, "W1X1", 4) != 0)
        return close_and_return(true);

    uint32_t version = 0, num_entries = 0;
    if (fread(&version, 4, 1, f) != 1 || fread(&num_entries, 4, 1, f) != 1)
        return close_and_return(false, "truncated W1X1 header");

    const uint64_t file_size = (uint64_t)fs::file_size(path);
    uint64_t max_end = 0;
    for (uint32_t i = 0; i < num_entries; ++i) {
        char name[64];
        uint64_t offset = 0;
        uint32_t nbytes = 0, pad = 0;
        if (fread(name, 1, sizeof(name), f) != sizeof(name) ||
            fread(&offset, sizeof(offset), 1, f) != 1 ||
            fread(&nbytes, sizeof(nbytes), 1, f) != 1 ||
            fread(&pad, sizeof(pad), 1, f) != 1) {
            return close_and_return(false, "truncated W1X1 index");
        }
        uint64_t end = offset + (uint64_t)nbytes;
        if (end < offset)
            return close_and_return(false, "invalid W1X1 tensor range");
        if (end > max_end) max_end = end;
    }

    if (max_end > file_size)
        return close_and_return(false, "truncated W1X1 tensor data");
    return close_and_return(true);
}

inline bool ensure_file(const std::string& local_path, const std::string& url)
{
    namespace fs = std::filesystem;
    if (fs::exists(local_path)) {
        std::string reason;
        if (validate_w1x1_file(local_path, &reason))
            return true;
        fprintf(stderr, "  [download] removing invalid %s (%s)\n",
                local_path.c_str(), reason.c_str());
        fs::remove(local_path);
    }

    if (fs::path p(local_path); p.has_parent_path())
        fs::create_directories(p.parent_path());

    printf("  [download] %s\n            <- %s\n", local_path.c_str(), url.c_str());
    fflush(stdout);

    // -L: follow redirects (HuggingFace LFS uses them)
    // -f: fail on HTTP error (4xx/5xx)
    // --progress-bar: single-line progress
    std::string cmd = "curl -L -f --progress-bar -o \""
                    + local_path + "\" \"" + url + "\"";
    int ret = system(cmd.c_str());
    if (ret != 0) {
        fs::remove(local_path);   // remove partial file
        fprintf(stderr, "\n  [download FAILED] exit=%d  %s\n", ret, url.c_str());
        return false;
    }
    std::string reason;
    if (!validate_w1x1_file(local_path, &reason)) {
        fs::remove(local_path);
        fprintf(stderr, "\n  [download FAILED] invalid file: %s  %s\n",
                reason.c_str(), url.c_str());
        return false;
    }
    printf("  [download OK] %s\n", local_path.c_str());
    return true;
}
