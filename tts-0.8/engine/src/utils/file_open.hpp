// src/utils/file_open.hpp — fopen for UTF-8 paths.
//
// Paths reach the engine as UTF-8 (pybind11 converts a Python str to UTF-8). On Windows, narrow fopen()
// reads a path in the ANSI code page instead, so any non-ASCII character (for example a user folder named
// José) fails to open. There the path is converted to UTF-16 and opened with _wfopen(). Elsewhere it is plain fopen().
#pragma once

#include <cstdio>
#include <string>

#ifdef _WIN32
#include <exception>
#include <filesystem>
#endif

inline FILE* fopen_utf8(const std::string& path, const char* mode) {
#ifdef _WIN32
    try {
        const std::filesystem::path wide(std::u8string(path.begin(), path.end()));
        const std::wstring wide_mode(mode, mode + std::char_traits<char>::length(mode));
        return _wfopen(wide.c_str(), wide_mode.c_str());
    } catch (const std::exception&) {
        return fopen(path.c_str(), mode);  // not valid UTF-8: open it as the C runtime would
    }
#else
    return fopen(path.c_str(), mode);
#endif
}
