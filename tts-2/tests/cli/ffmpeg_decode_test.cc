// kitten-tts-native: reference audio decode with the system's ffmpeg program (cli/ffmpeg_decode.h).
#include "ffmpeg_decode.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using kt2n::cli::ffmpeg_decode;
using kt2n::cli::ffmpeg_decode_args;
using kt2n::cli::FfmpegMissing;

TEST(FfmpegDecode, ArgumentsMirrorTheReferenceDecode) {
  const std::vector<std::string> want = {"-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i",
                                         "my file.wav", "-map_metadata", "-1", "-vn", "-sn", "-dn", "-ac", "1",
                                         "-ar", "24000", "-t", "30.100", "-f", "f32le", "pipe:1"};
  EXPECT_EQ(ffmpeg_decode_args("my file.wav"), want);
}

TEST(FfmpegDecode, MissingProgramNamesTheInstallCommand) {
  try {
    ffmpeg_decode("unused.wav", 24000, 30.1, "kt2n-no-such-ffmpeg-program");
    FAIL() << "expected FfmpegMissing";
  } catch (const FfmpegMissing& e) {
    const std::string m = e.what();
    EXPECT_NE(m.find("ffmpeg was not found"), std::string::npos) << m;
#if defined(_WIN32)
    EXPECT_NE(m.find("winget install"), std::string::npos) << m;
#else
    EXPECT_NE(m.find("apt-get install -y ffmpeg"), std::string::npos) << m;
#endif
    EXPECT_EQ(m.find('\n'), std::string::npos) << "one line";
  }
}

bool have_ffmpeg() {
#if defined(_WIN32)
  return std::system("ffmpeg -version > NUL 2>&1") == 0;
#else
  return std::system("ffmpeg -version > /dev/null 2>&1") == 0;
#endif
}

void put32(std::ofstream& f, uint32_t v) {
  for (int i = 0; i < 4; ++i) f.put(static_cast<char>((v >> (8 * i)) & 0xff));
}
void put16(std::ofstream& f, uint16_t v) {
  f.put(static_cast<char>(v & 0xff));
  f.put(static_cast<char>(v >> 8));
}

// A 16-bit mono WAV at 24 kHz needs no resampling: ffmpeg's s16 -> float conversion is x / 32768, exactly.
TEST(FfmpegDecode, DecodesA24kHzWavExactly) {
  if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg is not on PATH";
  const fs::path path = fs::temp_directory_path() / ("kt2n-ffmpeg-test-" + std::to_string(std::rand()) + " a.wav");
  std::vector<int16_t> pcm(24000 * 2);
  for (size_t i = 0; i < pcm.size(); ++i)
    pcm[i] = static_cast<int16_t>(std::lround(12000.0 * std::sin(2.0 * 3.14159265358979 * 220.0 * double(i) / 24000.0)));
  {
    std::ofstream f(path, std::ios::binary);
    const uint32_t data = static_cast<uint32_t>(pcm.size() * 2);
    f.write("RIFF", 4);
    put32(f, 36 + data);
    f.write("WAVEfmt ", 8);
    put32(f, 16);
    put16(f, 1);
    put16(f, 1);
    put32(f, 24000);
    put32(f, 48000);
    put16(f, 2);
    put16(f, 16);
    f.write("data", 4);
    put32(f, data);
    f.write(reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(data));
  }
  const std::vector<float> got = ffmpeg_decode(path.string());
  fs::remove(path);
  ASSERT_EQ(got.size(), pcm.size());
  for (size_t i = 0; i < pcm.size(); ++i) ASSERT_EQ(got[i], static_cast<float>(pcm[i]) / 32768.0f) << i;
}

TEST(FfmpegDecode, ABadFileIsAnError) {
  if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg is not on PATH";
  const fs::path path = fs::temp_directory_path() / ("kt2n-ffmpeg-test-" + std::to_string(std::rand()) + ".wav");
  {
    std::ofstream f(path, std::ios::binary);
    f << "this is not audio";
  }
  EXPECT_THROW(ffmpeg_decode(path.string()), std::runtime_error);
  fs::remove(path);
}

}  // namespace
