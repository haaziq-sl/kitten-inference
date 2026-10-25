#pragma once
#include "engine/layer.hpp"

class EmbeddingFp32 : public ILayer {
public:
    int num_embeddings = 0, embedding_dim = 0;
    std::vector<float> weight;

    void     load_weights(const ReadTensorFn&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    MemSpace required_mem_space() const override { return MemSpace::Host; }
    Shape output_shape(Shape in) const override {
        return {embedding_dim, in.d1, in.d2};
    }
};

class BertEmbeddingsFp32 : public ILayer {
public:
    int emb_size = 0;
    std::vector<float> word_weight;
    std::vector<float> pos_weight;
    std::vector<float> type_weight;
    std::vector<float> ln_weight;
    std::vector<float> ln_bias;

    void     load_weights(const ReadTensorFn&) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    MemSpace required_mem_space() const override { return MemSpace::Host; }
    Shape output_shape(Shape in) const override { return {emb_size, in.d1, in.d2}; }
};
