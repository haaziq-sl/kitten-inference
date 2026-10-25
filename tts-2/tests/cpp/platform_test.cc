// kt2n::platform (include/kt2n/platform.h): the read-only file mapping behind every model loader, and the clock
// helpers, on Linux and Windows.
#include "kt2n/platform.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <random>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

fs::path temp_dir() {
  const fs::path d = fs::temp_directory_path() / ("kt2n_platform_test_" + std::to_string(std::random_device{}()));
  fs::create_directories(d);
  return d;
}

TEST(Platform, MapsAFileReadOnly) {
  const fs::path d = temp_dir();
  const fs::path f = d / "data.bin";
  std::vector<char> bytes(70000);
  for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = char(i * 131 + 7);
  std::ofstream(f, std::ios::binary).write(bytes.data(), std::streamsize(bytes.size()));
  const uint8_t* data = nullptr;
  uint64_t size = 0;
  std::string why;
  ASSERT_EQ(kt2n::platform::map_file(f.string(), &data, &size, &why), kt2n::platform::MapResult::kOk) << why;
  ASSERT_EQ(size, bytes.size());
  EXPECT_EQ(std::memcmp(data, bytes.data(), bytes.size()), 0);
  kt2n::platform::unmap_file(data, size);
  fs::remove_all(d);
}

TEST(Platform, EmptyFileMapsToNothing) {
  const fs::path d = temp_dir();
  const fs::path f = d / "empty.bin";
  std::ofstream(f, std::ios::binary).flush();
  const uint8_t* data = reinterpret_cast<const uint8_t*>(&d);  // must be overwritten
  uint64_t size = 1;
  EXPECT_EQ(kt2n::platform::map_file(f.string(), &data, &size), kt2n::platform::MapResult::kOk);
  EXPECT_EQ(data, nullptr);
  EXPECT_EQ(size, 0u);
  kt2n::platform::unmap_file(data, size);  // no-op
  fs::remove_all(d);
}

TEST(Platform, MissingFileAndDirectory) {
  const fs::path d = temp_dir();
  const uint8_t* data = nullptr;
  uint64_t size = 0;
  std::string why;
  EXPECT_EQ(kt2n::platform::map_file((d / "missing.bin").string(), &data, &size, &why),
            kt2n::platform::MapResult::kOpenFailed);
  EXPECT_EQ(why, "No such file or directory");  // the same text on Linux and Windows
  EXPECT_EQ(data, nullptr);
  EXPECT_EQ(kt2n::platform::map_file(d.string(), &data, &size, &why), kt2n::platform::MapResult::kNotRegular);
  fs::remove_all(d);
}

TEST(Platform, PageSizeAndShortSleep) {
  const size_t page = kt2n::platform::page_size();
  EXPECT_GE(page, 4096u);
  EXPECT_EQ(page & (page - 1), 0u);  // a power of two
  kt2n::platform::TimerResolution timer;
  const auto t0 = std::chrono::steady_clock::now();
  kt2n::platform::sleep_us(300);
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
  EXPECT_GE(us, 290);
  EXPECT_LT(us, 50000);  // a short pause, not a scheduler tick
}

}  // namespace
