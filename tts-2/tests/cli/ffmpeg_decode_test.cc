// kitten-tts-native: reference audio decode with the system's ffmpeg program (cli/ffmpeg_decode.h).
#include "ffmpeg_decode.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using kt2n::cli::ffmpeg_decode;
using kt2n::cli::ffmpeg_decode_args;
using kt2n::cli::FfmpegMissing;

TEST(FfmpegDecode, ArgumentsMirrorTheReferenceDecode) {
  const std::vector<std::string> want = {"-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i",
                                         "pipe:0", "-map_metadata", "-1", "-vn", "-sn", "-dn", "-ac", "1",
                                         "-ar", "24000", "-t", "30.100", "-f", "f32le", "pipe:1"};
  EXPECT_EQ(ffmpeg_decode_args("pipe:0"), want);
  EXPECT_EQ(ffmpeg_decode_args("my file.wav")[7], "my file.wav");
}

TEST(FfmpegDecode, MissingProgramNamesTheInstallCommand) {
  try {
    ffmpeg_decode("unused.wav", std::vector<char>(100, 'x'), 24000, 30.1, "kt2n-no-such-ffmpeg-program");
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

std::vector<char> read_file(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

fs::path temp_name(const std::string& suffix) {
  return fs::temp_directory_path() / ("kt2n-ffmpeg-test-" + std::to_string(std::rand()) + suffix);
}

// Two seconds of a 220 Hz tone, 16-bit mono at 24 kHz.
std::vector<int16_t> write_tone_wav(const fs::path& path) {
  std::vector<int16_t> pcm(24000 * 2);
  for (size_t i = 0; i < pcm.size(); ++i)
    pcm[i] = static_cast<int16_t>(std::lround(12000.0 * std::sin(2.0 * 3.14159265358979 * 220.0 * double(i) / 24000.0)));
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
  return pcm;
}

// A 16-bit mono WAV at 24 kHz needs no resampling: ffmpeg's s16 -> float conversion is x / 32768, exactly.
TEST(FfmpegDecode, DecodesA24kHzWavExactly) {
  if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg is not on PATH";
  const fs::path path = temp_name(" a.wav");
  const std::vector<int16_t> pcm = write_tone_wav(path);
  const std::vector<float> got = ffmpeg_decode(path.string(), read_file(path));
  fs::remove(path);
  ASSERT_EQ(got.size(), pcm.size());
  for (size_t i = 0; i < pcm.size(); ++i) ASSERT_EQ(got[i], static_cast<float>(pcm[i]) / 32768.0f) << i;
}

// The file name as kitten-tts-native passes it: UTF-8 (on Windows wmain converts the wide arguments to UTF-8).
std::string utf8_name(const fs::path& path) {
  const std::u8string s = path.u8string();
  return std::string(s.begin(), s.end());
}

// An M4A file with its index (moov) after the audio cannot be read from a pipe: ffmpeg then reads it by name, so the
// name goes on ffmpeg's command line. This one has spaces and characters outside any 8-bit code page, which on Windows
// must survive the UTF-8 to UTF-16 conversion and the command-line quoting for CreateProcessW.
TEST(FfmpegDecode, ReadsAnM4aWithItsIndexAtTheEnd) {
  if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg is not on PATH";
  const fs::path wav = temp_name(".wav"), m4a = temp_name(".m4a");
  write_tone_wav(wav);
  const std::string cmd = "ffmpeg -nostdin -hide_banner -loglevel error -y -i \"" + wav.string() +
                          "\" -c:a aac -b:a 96k \"" + m4a.string() + "\"";
  ASSERT_EQ(std::system(cmd.c_str()), 0) << cmd;
  // Made under an ASCII name (std::system takes the ANSI code page on Windows), then renamed with the wide-character
  // file API.
  const fs::path named =
      m4a.parent_path() / fs::path(u8"kt2n ffmpeg test \u00e9\u00df \u4f60\u597d \u0436 my voice.m4a");
  fs::rename(m4a, named);
  const std::vector<float> got = ffmpeg_decode(utf8_name(named), read_file(named));
  fs::remove(wav);
  fs::remove(named);
  EXPECT_GT(got.size(), size_t(24000 * 19 / 10));  // about 2 s at 24 kHz
}

// The same name for the first attempt's file (its bytes go on standard input) and a missing file for the fallback: a
// file that ffmpeg cannot decode from the pipe and that is gone by name fails with ffmpeg's error, not a hang.
TEST(FfmpegDecode, AnUndecodableFileWithANonAsciiNameIsAnError) {
  if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg is not on PATH";
  const fs::path gone = fs::temp_directory_path() / fs::path(u8"kt2n missing \u00e9 \u4f60.m4a");
  EXPECT_THROW(ffmpeg_decode(utf8_name(gone), std::vector<char>(4096, '\0')), std::runtime_error);
}

// Changes the current folder for one test, and back.
struct InFolder {
  fs::path old = fs::current_path();
  explicit InFolder(const fs::path& dir) { fs::current_path(dir); }
  ~InFolder() { fs::current_path(old); }
};

// The by-name read, forced: the bytes on standard input are not audio, the file is. With the name as the CLI gets it
// (absolute, UTF-8, spaces and characters outside any 8-bit code page; on Windows with its drive letter).
TEST(FfmpegDecode, ReadsByNameWhenThePipeHasNoAudio) {
  if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg is not on PATH";
  const fs::path dir = temp_name("-byname");
  fs::create_directories(dir);
  const fs::path ascii = dir / "tone.wav";
  const std::vector<int16_t> pcm = write_tone_wav(ascii);
  const fs::path named = dir / fs::path(u8"kt2n by name \u00e9\u00df \u4f60\u597d.wav");
  fs::rename(ascii, named);
  const std::vector<float> got = ffmpeg_decode(utf8_name(fs::absolute(named)), std::vector<char>(4096, '\0'));
  fs::remove_all(dir);
  ASSERT_EQ(got.size(), pcm.size());
  for (size_t i = 0; i < pcm.size(); ++i) ASSERT_EQ(got[i], static_cast<float>(pcm[i]) / 32768.0f) << i;
}

#if !defined(_WIN32)
// A bare name read by name: without the "file:" prefix ffmpeg reads "2026-10-25T10:30.wav" as protocol
// "2026-10-25T10" (recorders name files like this). Windows file names cannot contain ':'.
TEST(FfmpegDecode, ReadsABareFileNameWithAColon) {
  if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg is not on PATH";
  const fs::path dir = temp_name("-colon");
  fs::create_directories(dir);
  const std::vector<int16_t> pcm = write_tone_wav(dir / "2026-10-25T10:30.wav");
  std::vector<float> got;
  {
    InFolder here(dir);
    got = ffmpeg_decode("2026-10-25T10:30.wav", std::vector<char>(4096, '\0'));  // not audio: read by name
  }
  fs::remove_all(dir);
  ASSERT_EQ(got.size(), pcm.size());
}
#endif

// An ffmpeg in the current folder is never run (Windows looks there before PATH; elsewhere an empty PATH entry means
// the current folder): the one on PATH decodes, or, without one, the missing-ffmpeg message comes.
TEST(FfmpegDecode, NeverRunsAnFfmpegFromTheCurrentFolder) {
  const bool on_path = have_ffmpeg();  // before the planted one is there
  const fs::path dir = temp_name("-planted");
  fs::create_directories(dir);
  const fs::path wav = dir / "tone.wav";
  const std::vector<int16_t> pcm = write_tone_wav(wav);
#if defined(_WIN32)
  { std::ofstream(dir / "ffmpeg.exe", std::ios::binary) << "not a program"; }
  { std::ofstream(dir / "ffmpeg.bat") << "@exit /b 3\r\n"; }
#else
  { std::ofstream(dir / "ffmpeg") << "#!/bin/sh\nexit 3\n"; }
  fs::permissions(dir / "ffmpeg", fs::perms::owner_all);
  const std::string old_path = std::getenv("PATH") ? std::getenv("PATH") : "";
  setenv("PATH", (std::string(":.:") + old_path).c_str(), 1);
#endif
  std::vector<float> got;
  std::string error;
  {
    InFolder here(dir);
    try {
      got = ffmpeg_decode("tone.wav", read_file(wav));
    } catch (const std::exception& e) {
      error = e.what();
    }
  }
#if !defined(_WIN32)
  setenv("PATH", old_path.c_str(), 1);
#endif
  fs::remove_all(dir);
  if (on_path) {
    ASSERT_EQ(error, "");
    ASSERT_EQ(got.size(), pcm.size());
  } else {
    EXPECT_NE(error.find("ffmpeg was not found"), std::string::npos) << error;
    EXPECT_EQ(error.find("Windows error"), std::string::npos) << error;
  }
}

TEST(FfmpegDecode, ABadFileIsAnError) {
  if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg is not on PATH";
  const fs::path path = temp_name(".wav");
  {
    std::ofstream f(path, std::ios::binary);
    f << "this is not audio";
  }
  EXPECT_THROW(ffmpeg_decode(path.string(), read_file(path)), std::runtime_error);
  fs::remove(path);
}

// A "recording" that is a playlist or manifest would make ffmpeg's by-name run open the local files it names (HLS and
// DASH segments); such files are refused before ffmpeg runs. Real audio is not.
TEST(FfmpegDecode, PlaylistsAndManifestsAreRefused) {
  using kt2n::cli::is_playlist;
  auto bytes = [](const std::string& s) { return std::vector<char>(s.begin(), s.end()); };
  EXPECT_TRUE(is_playlist(bytes("#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nfile:/srv/audio/other.wav\n")));
  EXPECT_TRUE(is_playlist(bytes("\xEF\xBB\xBF\r\n  #EXTM3U\n#EXTINF:1,\nhttp://example.com/a.ts\n")));
  EXPECT_TRUE(is_playlist(bytes("<?xml version=\"1.0\"?>\n<MPD profiles=\"urn:mpeg:dash:profile:isoff-on-demand:2011\">")));
  // FFmpeg finds DASH anywhere in what it probes, in any case; SDP by a c= line.
  EXPECT_TRUE(is_playlist(bytes(std::string(5000, 'x') + "<mPd profiles=\"urn:mpeg:DASH:Profile:full:2011\">")));
  EXPECT_TRUE(is_playlist(bytes("v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=x\r\nc=IN IP4 127.0.0.1\r\nm=audio 5004 RTP/AVP 0\r\n")));
  EXPECT_TRUE(is_playlist(bytes("<CompositionPlaylist xmlns=\"http://www.smpte-ra.org/schemas/2067-3/2016\">")));
  EXPECT_TRUE(is_playlist(bytes("ffconcat version 1.0\nfile /srv/audio/other.wav\n")));
  EXPECT_FALSE(is_playlist(bytes("RIFF....WAVEfmt ")));
  EXPECT_FALSE(is_playlist(bytes(std::string("ID3\x04") + std::string(6, '\0') + " <MPD in a tag")));  // binary audio
  EXPECT_FALSE(is_playlist({}));
  const fs::path path = temp_name(".wav");
  {
    std::ofstream f(path, std::ios::binary);
    f << "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nfile:" << path.string() << "\n#EXT-X-ENDLIST\n";
  }
  try {
    ffmpeg_decode(path.string(), read_file(path), 24000, 30.1, "kt2n-no-such-ffmpeg-program");
    FAIL() << "expected an error";
  } catch (const FfmpegMissing&) {
    FAIL() << "ffmpeg was to be started for a playlist";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("playlist or stream manifest"), std::string::npos) << e.what();
  }
  fs::remove(path);
}

}  // namespace
