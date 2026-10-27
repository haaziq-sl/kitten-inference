// Voice cloning: the in-process reference decode (src/voice/decode.cc) reads only the bytes it is given. A
// "recording" that is really a playlist or manifest (HLS, DASH) must not make it open the URLs or files it names.
// CPU only; Linux and macOS (a local TCP listener stands in for a remote server).
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "decode.h"

using kt2n::voice::decode_audio;

static int g_fail = 0;
#define CHECK(c)                                                       \
  do {                                                                 \
    if (!(c)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++g_fail;                                                        \
    }                                                                  \
  } while (0)

// 0.5 s of a 440 Hz tone, 24 kHz mono 16-bit PCM WAV.
static std::vector<uint8_t> wav() {
  const uint32_t n = 12000, data = n * 2;
  std::vector<uint8_t> w;
  auto put = [&](const void* p, size_t k) { w.insert(w.end(), (const uint8_t*)p, (const uint8_t*)p + k); };
  auto u32 = [&](uint32_t v) { put(&v, 4); };
  auto u16 = [&](uint16_t v) { put(&v, 2); };
  put("RIFF", 4); u32(36 + data); put("WAVE", 4);
  put("fmt ", 4); u32(16); u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16);
  put("data", 4); u32(data);
  for (uint32_t i = 0; i < n; ++i) u16((uint16_t)(int16_t)std::lround(8000 * std::sin(2 * M_PI * 440 * i / 24000.0)));
  return w;
}

// Counts connections and closes each one at once, so that a decode that connects gets an error instead of waiting.
struct Listener {
  int fd = -1;
  int port = 0;
  std::atomic<int> count{0};
  std::atomic<bool> stop{false};
  std::thread worker;
  Listener() {
    fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof a;
    if (fd < 0 || bind(fd, (sockaddr*)&a, sizeof a) != 0 || listen(fd, 8) != 0 ||
        getsockname(fd, (sockaddr*)&a, &len) != 0) {
      std::perror("listener");
      std::exit(2);
    }
    port = ntohs(a.sin_port);
    worker = std::thread([this] {
      while (!stop) {
        pollfd p{fd, POLLIN, 0};
        if (poll(&p, 1, 50) > 0) {
          int c = accept(fd, nullptr, nullptr);
          if (c >= 0) {
            ++count;
            close(c);
          }
        }
      }
    });
  }
  ~Listener() {
    stop = true;
    worker.join();
    close(fd);
  }
  // Whether anything connected since the last call.
  bool contacted() {
    usleep(200 * 1000);
    return count.exchange(0) > 0;
  }
};

static bool decodes(const std::string& text, std::string* err) {
  std::vector<float> out;
  return decode_audio((const uint8_t*)text.data(), text.size(), 30.1, out, err) && !out.empty();
}

int main() {
  // A real recording still decodes, sample for sample.
  {
    auto w = wav();
    std::vector<float> out;
    std::string err;
    CHECK(decode_audio(w.data(), w.size(), 30.1, out, &err));
    CHECK(out.size() == 12000);
  }
  Listener server;
  const std::string url = "http://127.0.0.1:" + std::to_string(server.port);
  // A local recording that a playlist or manifest could name.
  char dir[] = "/tmp/kt2n-decode-XXXXXX";
  CHECK(mkdtemp(dir) != nullptr);
  const std::string local = std::string(dir) + "/real.wav";
  {
    auto w = wav();
    std::ofstream(local, std::ios::binary).write((const char*)w.data(), (std::streamsize)w.size());
  }
  const std::string cases[][2] = {
      {"HLS media playlist, http segment",
       "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1.0,\n" + url +
           "/seg.ts\n#EXT-X-ENDLIST\n"},
      {"HLS media playlist, local file",
       "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1.0,\nfile:" + local +
           "\n#EXT-X-ENDLIST\n"},
      {"DASH manifest, http segment",
       "<?xml version=\"1.0\"?>\n<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" profiles=\"urn:mpeg:dash:profile:isoff-"
       "on-demand:2011\" type=\"static\" mediaPresentationDuration=\"PT1S\" minBufferTime=\"PT1S\"><Period><AdaptationSet "
       "mimeType=\"audio/mp4\"><Representation id=\"a\" bandwidth=\"64000\" codecs=\"mp4a.40.2\"><BaseURL>" +
           url + "/audio.mp4</BaseURL></Representation></AdaptationSet></Period></MPD>\n"},
      {"SDP session description (RTP over UDP)",
       "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=x\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=audio " + std::to_string(server.port) +
           " RTP/AVP 0\r\n"},
      {"DASH manifest, local file",
       "<?xml version=\"1.0\"?>\n<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" profiles=\"urn:mpeg:dash:profile:isoff-"
       "on-demand:2011\" type=\"static\" mediaPresentationDuration=\"PT1S\" minBufferTime=\"PT1S\"><Period><AdaptationSet "
       "mimeType=\"audio/wav\"><Representation id=\"a\" bandwidth=\"64000\"><BaseURL>file:" +
           local + "</BaseURL></Representation></AdaptationSet></Period></MPD>\n"},
  };
  for (const auto& c : cases) {
    std::string err;
    const bool ok = decodes(c[1], &err);
    const bool contacted = server.contacted();
    std::printf("%s: %s (%s)%s\n", c[0].c_str(), ok ? "decoded" : "refused", err.c_str(),
                contacted ? ", and the server was contacted" : "");
    CHECK(!ok);
    CHECK(!contacted);
  }
  std::remove(local.c_str());
  rmdir(dir);
  if (g_fail) {
    std::fprintf(stderr, "%d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("voice.decode: ok\n");
  return 0;
}
