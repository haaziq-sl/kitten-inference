// kitten-tts-native: decoding the reference audio for voice cloning with the system's ffmpeg program, for engine
// libraries built without the in-process decode (kt2n_build_info() lists KT2N_CAP_VOICE_PCM_ONLY). CPU only.
//
// The command mirrors the reference pipeline's upload decode, which the in-process decode (src/voice/decode.h) also
// mirrors, reading the file by name instead of from a pipe:
//   ffmpeg -nostdin -hide_banner -loglevel error -threads 1 -i FILE -map_metadata -1 -vn -sn -dn -ac 1 -ar 24000
//          -t 30.100 -f f32le pipe:1
// ffmpeg is started directly (no shell), with standard input closed, and its standard error goes to the caller's.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace kt2n::cli {

// ffmpeg is not on PATH (or cannot be started). what() is one line that names the install command.
class FfmpegMissing : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// The arguments after the program name, for `path` at `rate` Hz, cut at `max_seconds` (30.1 s like the reference).
std::vector<std::string> ffmpeg_decode_args(const std::string& path, int rate = 24000, double max_seconds = 30.1);

// The one-line error for a missing ffmpeg.
std::string ffmpeg_missing_message();

// Runs `program` (searched on PATH) with ffmpeg_decode_args() and returns its mono float32 samples. Throws
// FfmpegMissing when it cannot be started, std::runtime_error when it fails or returns no audio.
std::vector<float> ffmpeg_decode(const std::string& path, int rate = 24000, double max_seconds = 30.1,
                                 const std::string& program = "ffmpeg");

}  // namespace kt2n::cli
