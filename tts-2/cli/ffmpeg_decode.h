// kitten-tts-native: decoding the reference audio for voice cloning with the system's ffmpeg program, for engine
// libraries built without the in-process decode (kt2n_build_info() lists KT2N_CAP_VOICE_PCM_ONLY). CPU only.
//
// The command is the reference pipeline's upload decode, which the in-process decode (src/voice/decode.h) mirrors:
//   ffmpeg -nostdin -hide_banner -loglevel error -threads 1 -i pipe:0 -map_metadata -1 -vn -sn -dn -ac 1 -ar 24000
//          -t 30.100 -f f32le pipe:1
// with the file's bytes on standard input, so the samples are the same as the default build's with the same FFmpeg
// version, and also with some others (measured bit for bit on WAV, MP3, FLAC, Ogg Vorbis and M4A: FFmpeg 6.1.1 and
// 8.1.3 programs against the default build's FFmpeg 6.1.1 libraries; FFmpeg 9.0.2 decodes M4A/AAC 142 samples shorter,
// the other formats the same). Some files cannot be read from a pipe (an MP4/M4A file whose index is at the end): when
// that gives no audio, ffmpeg reads the file by name instead (`-i FILE`), where the in-process decode has no audio at
// all. ffmpeg is started directly (no shell).
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

// The arguments after the program name, reading `input` ("pipe:0" or a file name) at `rate` Hz, cut at `max_seconds`
// (30.1 s like the reference).
std::vector<std::string> ffmpeg_decode_args(const std::string& input, int rate = 24000, double max_seconds = 30.1);

// The one-line error for a missing ffmpeg.
std::string ffmpeg_missing_message();

// Decodes the reference audio file `path`, whose bytes are `data`, with `program` (searched on PATH) and returns mono
// float32 samples at `rate` Hz. Throws FfmpegMissing when it cannot be started, std::runtime_error when it fails or
// returns no audio (ffmpeg's own message is printed on standard error).
std::vector<float> ffmpeg_decode(const std::string& path, const std::vector<char>& data, int rate = 24000,
                                 double max_seconds = 30.1, const std::string& program = "ffmpeg");

}  // namespace kt2n::cli
