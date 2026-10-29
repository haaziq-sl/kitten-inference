#pragma once
// audio.hpp — WAV file read/write utilities.
// Supports: PCM16 (1-2 ch), IEEE float32 (1-2 ch), mono output.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────
// read_wav
//   Returns mono float32 samples in [-1, 1].
//   Supports PCM16 and float32 WAVs, 1 or 2 channels (stereo is
//   downmixed by taking the first channel).
//   If expected_sr > 0 and the file's sample rate doesn't match,
//   a warning is printed but reading continues.
//   If out_sample_rate is not null, the file's actual rate is stored.
// ─────────────────────────────────────────────────────────────────
inline std::vector<float> read_wav(const std::string& path,
                                    int  expected_sr    = 0,
                                    int* out_sample_rate = nullptr)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open WAV: " + path);

    char riff[4], wave[4];
    uint32_t riff_size;
    fread(riff, 1, 4, f); fread(&riff_size, 4, 1, f); fread(wave, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        fclose(f);
        throw std::runtime_error("Not a WAV file: " + path);
    }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    bool found_fmt = false, found_data = false;
    long data_pos = 0;

    while (!feof(f)) {
        char id[4]; uint32_t sz;
        if (fread(id, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1) break;
        if (memcmp(id, "fmt ", 4) == 0) {
            found_fmt = true;
            fread(&audio_format,   2, 1, f);
            fread(&num_channels,   2, 1, f);
            fread(&sample_rate,    4, 1, f);
            uint32_t byte_rate;   fread(&byte_rate,    4, 1, f);
            uint16_t block_align; fread(&block_align,  2, 1, f);
            fread(&bits_per_sample, 2, 1, f);
            if (sz > 16) fseek(f, (long)(sz - 16), SEEK_CUR);
        } else if (memcmp(id, "data", 4) == 0) {
            found_data = true;
            data_size  = sz;
            data_pos   = ftell(f);
            break;
        } else {
            fseek(f, (long)sz, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data) {
        fclose(f);
        throw std::runtime_error("Malformed WAV: " + path);
    }
    if (audio_format != 1 && audio_format != 3) {
        fclose(f);
        throw std::runtime_error("Unsupported WAV format (need PCM16 or float32): " + path);
    }
    if (expected_sr > 0 && sample_rate != (uint32_t)expected_sr)
        fprintf(stderr, "Warning: WAV sample rate %u != expected %d\n",
                sample_rate, expected_sr);

    if (out_sample_rate) *out_sample_rate = (int)sample_rate;

    fseek(f, data_pos, SEEK_SET);
    int bytes_per_sample = bits_per_sample / 8;
    int total_frames     = (int)(data_size / ((uint32_t)bytes_per_sample * num_channels));
    std::vector<float> out(total_frames);

    if (audio_format == 1 && bits_per_sample == 16) {
        for (int i = 0; i < total_frames; ++i) {
            int16_t s = 0;
            fread(&s, 2, 1, f);
            if (num_channels == 2) { int16_t dummy; fread(&dummy, 2, 1, f); }
            out[i] = s / 32768.f;
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        for (int i = 0; i < total_frames; ++i) {
            float s;
            fread(&s, 4, 1, f);
            if (num_channels == 2) { float dummy; fread(&dummy, 4, 1, f); }
            out[i] = s;
        }
    } else {
        fclose(f);
        throw std::runtime_error("Unsupported bit depth in WAV: " + path);
    }

    fclose(f);
    return out;
}

// ─────────────────────────────────────────────────────────────────
// write_wav
//   Writes mono float32 samples as an IEEE float WAV (format 3).
// ─────────────────────────────────────────────────────────────────
inline void write_wav(const std::string& path, const float* data,
                       int n_samples, int sample_rate)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);

    const uint32_t data_bytes  = (uint32_t)n_samples * 4;
    const uint32_t fmt_size    = 18;  // IEEE float fmt chunk with cbSize=0
    const uint32_t riff_size   = 4 + (8 + fmt_size) + (8 + data_bytes);
    const uint32_t sr_u32      = (uint32_t)sample_rate;
    const uint32_t byte_rate   = (uint32_t)sample_rate * 4;
    const uint16_t audio_fmt   = 3;   // WAVE_FORMAT_IEEE_FLOAT
    const uint16_t channels    = 1;
    const uint16_t block_align = 4;
    const uint16_t bps         = 32;
    const uint16_t cb_size     = 0;

    fwrite("RIFF",       1, 4, f);
    fwrite(&riff_size,   4, 1, f);
    fwrite("WAVE",       1, 4, f);
    fwrite("fmt ",       1, 4, f);
    fwrite(&fmt_size,    4, 1, f);
    fwrite(&audio_fmt,   2, 1, f);
    fwrite(&channels,    2, 1, f);
    fwrite(&sr_u32,      4, 1, f);
    fwrite(&byte_rate,   4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bps,         2, 1, f);
    fwrite(&cb_size,     2, 1, f);
    fwrite("data",       1, 4, f);
    fwrite(&data_bytes,  4, 1, f);
    fwrite(data,         4, n_samples, f);
    fclose(f);
}
