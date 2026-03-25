# x-transformers Adjustable Parameters

Comprehensive reference of **every** parameter accepted by `Decoder`, `TransformerWrapper`,
and `AutoregressiveWrapper`, extracted directly from the x-transformers source code.

Each parameter is annotated with:
- **Experiment status** from our 247-experiment campaign (current best BPC: **1.2079**)
- **Flash compatibility** — whether it works with `attn_flash=True`
- **Practical recommendations** for our enwik8 training setup

> **Source files** (read-only reference):
> - `x-transformers/x_transformers/x_transformers.py` — core classes
> - `x-transformers/x_transformers/attend.py` — Attend class
> - `x-transformers/x_transformers/autoregressive_wrapper.py` — AR wrapper

---

## Table of Contents

- [Current Best Configuration](#current-best-configuration)
- [1. Core Architecture (Decoder)](#1-core-architecture-decoder)
- [2. Normalization (Decoder)](#2-normalization-decoder)
- [3. Feedforward Network (Decoder, ff\_ prefix)](#3-feedforward-network-decoder-ff_-prefix)
- [4. Attention — Core (Decoder, attn\_ prefix)](#4-attention--core-decoder-attn_-prefix)
- [5. Attention — QK Normalization](#5-attention--qk-normalization)
- [6. Attention — Multi-Query / Grouped-Query / Multi-Latent](#6-attention--multi-query--grouped-query--multi-latent)
- [7. Attention — Gating & Value Processing](#7-attention--gating--value-processing)
- [8. Attention — Sparse & Alternative Attention Functions](#8-attention--sparse--alternative-attention-functions)
- [9. Attention — Positional & Bias Mechanisms](#9-attention--positional--bias-mechanisms)
- [10. Attention — Advanced / Experimental](#10-attention--advanced--experimental)
- [11. Positional Encoding (Decoder)](#11-positional-encoding-decoder)
- [12. Residual Connections (Decoder)](#12-residual-connections-decoder)
- [13. Layer Structure (Decoder)](#13-layer-structure-decoder)
- [14. Multi-Stream / Cross-Layer (Decoder)](#14-multi-stream--cross-layer-decoder)
- [15. TransformerWrapper Options](#15-transformerwrapper-options)
- [16. AutoregressiveWrapper Options](#16-autoregressivewrapper-options)
- [17. Flash Attention Compatibility Matrix](#17-flash-attention-compatibility-matrix)
- [18. Untried Parameters — Prioritized Recommendations](#18-untried-parameters--prioritized-recommendations)
- [19. Training Strategy Ideas (Non-Parameter)](#19-training-strategy-ideas-non-parameter)
- [Recommended Combinations](#recommended-combinations)
- [Model Sizing Guide](#model-sizing-guide)

---

## Current Best Configuration

```python
# train.py as of experiment #247 (val_bpc = 1.2079)
TransformerWrapper(
    num_tokens=256,
    max_seq_len=4096,
    post_emb_norm=True,
    emb_frac_gradient=0.1,
    attn_layers=Decoder(
        dim=448, depth=6, heads=7,
        rotary_pos_emb=True,
        rotary_xpos=True,
        attn_flash=True,
        attn_qk_norm=True,
        attn_laser=True,
        ff_glu=True,
        ff_swish=True,
        ff_glu_mult_bias=True,
        use_rmsnorm=True,
        add_value_residual=True,
        shift_tokens=1,
        softclamp_output=True,
        zero_init_branch_output=True,
    ),
)
# Optimizer: MuonAdamAtan2(lr=1.1e-2, muon_beta1=0.92, muon_rms_factor=0.1,
#   decoupled_wd=True, weight_decay=0.001)
# BATCH_SIZE=24, GRADIENT_ACCUMULATE_EVERY=1, GRAD_CLIP=0.8
# WARMDOWN_RATIO=0.47, FINAL_LR_FRAC=0.02
```

---

## 1. Core Architecture (Decoder)

These are the fundamental parameters that determine model size and capacity.

| Parameter | Type | Default | Description | Experiment Status |
|-----------|------|---------|-------------|-------------------|
| `dim` | int | required | Model hidden dimension. Primary capacity knob. | **Current: 448.** Tried 192-768. Sweet spot is 448 (19.6M params, ~118M tokens). |
| `depth` | int | required | Number of transformer layers. | **Current: 6.** Tried 4-12. depth=6 is optimal -- deeper is too slow (fewer tokens). |
| `heads` | int | 8 | Number of attention heads. Usually `dim // 64`. | **Current: 7** (dim_head=64). Tried 4-12. heads=7 matches dim=448 well. |

---

## 2. Normalization (Decoder)

Only **one** normalization type can be active at a time. Enforced by assertion in source.

| Parameter | Type | Default | Description | Flash OK? | Experiment Status |
|-----------|------|---------|-------------|-----------|-------------------|
| `use_rmsnorm` | bool | False | RMSNorm instead of LayerNorm. No mean centering. Used in LLaMA/Gopher. [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467) | Yes | **KEEP.** In current best config. |
| `use_simple_rmsnorm` | bool | False | `l2norm(x) * sqrt(dim)` with no learned gamma. [Qin et al. 2023](https://arxiv.org/abs/2307.14995) | Yes | **Tried (#193):** 1.409 -- faster but less accurate. Discard. |
| `use_scalenorm` | bool | False | ScaleNorm -- simpler alternative to LayerNorm. [Nguyen & Salazar 2019](https://arxiv.org/abs/1910.05895) | Yes | **Untried.** Similar to simple_rmsnorm; unlikely to beat rmsnorm. Low priority. |
| `use_dynamic_tanh` | bool | False | DynamicTanh normalization ([arxiv 2503.10622](https://arxiv.org/abs/2503.10622)). Norm-free architecture. | Yes | **Tried (#196, #233):** 1.266/1.265 -- slower, worse than RMSNorm. Discard. |
| `use_derf` | bool | False | Derf normalization ([arxiv 2512.10938](https://arxiv.org/abs/2512.10938)). Another norm alternative. | Yes | **Tried (#153, #242):** 1.230/1.246 -- slower, much worse. Discard. |
| `use_adaptive_layernorm` | bool | False | Adaptive LayerNorm (for conditional generation, DiT paper). Requires `dim_condition`. | Yes | **N/A.** For conditional generation, not language modeling. |
| `use_adaptive_rmsnorm` | bool | False | Adaptive RMSNorm. Requires `dim_condition`. | Yes | **N/A.** Same as above. |
| `norm_add_unit_offset` | bool | **True** | Add unit offset to norm gammas so they can safely have weight decay applied. Per Ohad Rubin. | Yes | **Tried removing (#235):** 1.209 -- default True is slightly better. Keep default. |
| `sandwich_norm` | bool | False | Extra layernorm on branch outputs (pre-norm + post-norm on each sublayer). From CogView. [Ding et al. 2021](https://arxiv.org/abs/2105.13290) | Yes | **Tried (#22, #174):** 1.434/1.227 -- overhead not worth it. Discard. |
| `resi_dual` | bool | False | Hybrid pre+post layernorm. Reduces representation collapse while maintaining stability. [Microsoft 2023](https://arxiv.org/abs/2304.14802) | Yes | **Untried.** |
| `resi_dual_scale` | float | 0.1 | Scale factor for prenorm residual in resi_dual (prevents fp16 overflow). | Yes | -- |
| `pre_norm` | bool | True | Use pre-layernorm. Set False for post-layernorm. | Yes | **Default True.** Not worth changing -- pre-norm is standard and stable. |
| `pre_norm_has_final_norm` | bool | True | Whether pre-norm has a final norm layer. | Yes | **Untried.** Removing final norm would be unusual; not recommended. |
| `attn_head_scale` | bool | False | Per-head scaling after attention aggregation (Normformer). [Normformer 2022](https://openreview.net/forum?id=GMYWzWztDx5) | Yes | **Tried (#44, #232):** 1.416/1.215 -- slower, worse. Discard. |
| `ff_post_act_ln` | bool | False | Extra layernorm after feedforward activation (Normformer). [Normformer 2022](https://openreview.net/forum?id=GMYWzWztDx5) | Yes | **Tried (#40, #179):** 1.420/1.249 -- slower, worse. Discard. |

### Normalization sub-params (only relevant when corresponding norm is active)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dynamic_tanh_init_alpha` | float | 1.0 | Initial alpha for DynamicTanh. Only relevant with `use_dynamic_tanh=True`. |
| `dim_condition` | int | None (=dim) | Dimension of conditioning signal. Only for adaptive norms. |
| `adaptive_condition_mlp` | bool | False | Use MLP to process condition before adaptive norm. |
| `adaptive_condition_mlp_expansion` | int | 4 | Expansion factor for condition MLP. |
| `use_adaptive_layerscale` | bool | False | Ada-LN-Zero from DiT paper. Paired with adaptive layernorm. |

---

## 3. Feedforward Network (Decoder, ff_ prefix)

Parameters prefixed with `ff_` in `Decoder(...)` are stripped and passed to `FeedForward.__init__`.

| Parameter | Type | Default | Description | Experiment Status |
|-----------|------|---------|-------------|-------------------|
| `ff_glu` | bool | False | **Gated Linear Unit** in feedforward. "You should always turn this on." [Shazeer 2020](https://arxiv.org/abs/2002.05202) | **KEEP.** In current best config. |
| `ff_swish` | bool | False | SwiGLU activation when combined with `ff_glu=True` (PaLM/LLaMA). [PaLM 2022](https://arxiv.org/abs/2204.02311) | **KEEP.** In current best config. |
| `ff_glu_mult_bias` | bool | False | Learnable multiplicative bias in GLU gate. Adds `nn.Parameter(ones(inner_dim))`. | **KEEP.** In current best since exp #65 (1.381 -> improved). |
| `ff_relu_squared` | bool | False | ReLU^2 from Primer NAS. Note: if using GLU, GELU is default; relu_squared is for non-GLU. Mutually exclusive with `ff_solu`. [So et al. 2021](https://arxiv.org/abs/2109.08668) | **Was in baseline.** Superseded by SwiGLU. N/A when ff_glu+ff_swish active. |
| `ff_solu` | bool | False | **SoLU** (Softmax Linear Unit): `layernorm(x * softmax(x))`. From Anthropic interpretability work. Mutually exclusive with `ff_relu_squared`. | **UNTRIED.** Interesting activation -- adds a LayerNorm inside FF. May hurt throughput. Medium priority. |
| `ff_custom_activation` | Module | None | Arbitrary custom activation module (deepcopied). Overrides all activation choices. | **N/A.** For custom experimentation outside train.py. |
| `ff_mult` | int | 4 | FFN expansion factor. Inner dim = `dim * ff_mult`. With GLU, effective expansion is `ff_mult * 2/3`. | **Tried 2,3,5,8:** ff_mult=4 (default) is best. 2 too few params; 5,8 too slow. |
| `ff_no_bias` | bool | False | Remove bias from FF linear layers. PaLM style. [PaLM 2022](https://arxiv.org/abs/2204.02311) | **Tried (#14, #182):** 1.436/1.238 -- bias terms help for char-level. Discard. |
| `ff_dropout` | float | 0.0 | Dropout in feedforward. | **Not tried explicitly.** Short training = no regularization benefit. Not recommended. |
| `ff_sublayer_dropout` | float | 0.0 | Dropout after the entire feedforward output projection (distinct from ff_dropout which is inside). | **UNTRIED.** Short training, unlikely to help. Low priority. |
| `ff_zero_init_output` | bool | False | Zero-init output projection. Note: already controlled by `zero_init_branch_output` at Decoder level. | **Active via `zero_init_branch_output=True`** which sets both ff and attn zero_init. |

---

## 4. Attention -- Core (Decoder, attn_ prefix)

Parameters prefixed with `attn_` are stripped and passed to `Attention.__init__`.

| Parameter | Type | Default | Description | Flash OK? | Experiment Status |
|-----------|------|---------|-------------|-----------|-------------------|
| `attn_flash` | bool | False | Use PyTorch `scaled_dot_product_attention`. Faster + less memory. **Always turn on.** [Dao et al. 2022](https://arxiv.org/abs/2205.14135) | -- | **KEEP.** In current best. |
| `attn_dim_head` | int | 64 | Dimension per attention head. Usually leave as default. | Yes | **Default 64.** Not explicitly tuned. dim=448/heads=7 = 64. |
| `attn_dropout` | float | 0.0 | Dropout on attention weights post-softmax. | Yes | **Not tried.** Short training, not recommended. |
| `attn_sublayer_dropout` | float | 0.0 | Dropout after the attention output projection (different from attn_dropout). | Yes | **UNTRIED.** Low priority for same reason. |
| `attn_on_attn` | bool | False | Gate attention output with queries via GLU. [Huang et al. 2019](https://arxiv.org/abs/1908.06954) | N/A | **Tried (#67):** Crash -- conflicts with `zero_init_branch_output` (Sequential has no .weight). Incompatible with current config. |
| `attn_zero_init_output` | bool | False | Zero-init output projection. Already controlled by `zero_init_branch_output`. | Yes | **Active** via parent flag. |
| `attn_max_attend_past` | int | None | Limit how far back attention can look (sliding window). | Yes | **UNTRIED.** Could be interesting for very long sequences. We use seq=4096, and full attention has been good. Low priority. |
| `attn_num_mem_kv` | int | 0 | Learned persistent memory key/value pairs prepended to attention. [Sukhbaatar et al. 2019](https://arxiv.org/abs/1907.01470) | **No (compile)** | **Tried (#12, #84, #168):** 1.427/1.384/crash. Hurts throughput significantly; torch.compile shape error at seq=4096. Discard. |

---

## 5. Attention -- QK Normalization

| Parameter | Type | Default | Description | Flash OK? | Experiment Status |
|-----------|------|---------|-------------|-----------|-------------------|
| `attn_qk_norm` | bool | False | L2-normalize Q and K (cosine similarity attention). Critical for stability at high LR. [Henry et al. 2020](https://arxiv.org/abs/2010.04245), [Dehghani et al. 2023](https://arxiv.org/abs/2302.05442) | Yes | **KEEP.** Removing caused catastrophic divergence (1.73 BPC, exp #158). |
| `attn_qk_norm_groups` | int | 1 | Number of groups for grouped QK normalization. dim_head must be divisible. | Yes | **Tried 8 (#120, #239):** 1.289/1.240 -- much worse. Default 1 is best. |
| `attn_qk_norm_scale` | float | 10 | Fixed scale for cosine sim attention. | Yes | **Tried 8,15 (#246, #247):** 1.224/1.211 -- default 10 is sweet spot. |
| `attn_qk_norm_dim_scale` | bool | False | Learned scale per feature dimension (as in Google Brain 22B paper). | Yes | **Tried (#80, #225):** 1.390/1.216 -- slower, worse. Discard. |

---

## 6. Attention -- Multi-Query / Grouped-Query / Multi-Latent

| Parameter | Type | Default | Description | Flash OK? | Experiment Status |
|-----------|------|---------|-------------|-----------|-------------------|
| `attn_one_kv_head` | bool | False | Multi-Query Attention: single KV head. Mutually exclusive with `attn_kv_heads`. [Shazeer 2019](https://arxiv.org/abs/1911.02150) | Yes | **Untried.** Saves memory but likely hurts quality at our small scale. |
| `attn_kv_heads` | int | None (=heads) | Grouped-Query Attention. E.g. heads=8, kv_heads=2 means 4:1 sharing. [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245) | Yes | **Tried (#56):** Crash -- conflicts with `add_value_residual`. Incompatible with current config. |
| `attn_value_dim_head` | int | None (=dim_head) | Separate dimension per value head (can differ from key dim). | Yes | **UNTRIED.** Exotic. Might allow larger value dim without affecting QK compute. Low priority. |
| `attn_dim_out` | int | None (=dim) | Output dimension (can differ from input dim). | Yes | **N/A.** For cross-attention or special architectures. |
| `attn_use_latent_q` | bool | False | Multi-Latent Attention (MLA) -- compress queries through a latent bottleneck. | Yes | **UNTRIED.** DeepSeek-style. Needs `attn_dim_latent_q`. Interesting but complex. |
| `attn_dim_latent_q` | int | None | Dimension for latent queries (MLA). Required if `attn_use_latent_q=True`. | Yes | -- |
| `attn_use_latent_kv` | bool | False | MLA -- compress key-values through a latent bottleneck. Saves KV cache memory. | Yes | **UNTRIED.** Promising for memory savings. See [Untried Recommendations](#18-untried-parameters--prioritized-recommendations). |
| `attn_dim_latent_kv` | int | None | Dimension for latent key-values (MLA). Required if `attn_use_latent_kv=True`. | Yes | -- |
| `attn_latent_rope_subheads` | int | None | Number of subheads for decoupled RoPE in MLA. Some keys bypass the latent path and get RoPE directly. | Yes | -- |

---

## 7. Attention -- Gating & Value Processing

| Parameter | Type | Default | Description | Flash OK? | Experiment Status |
|-----------|------|---------|-------------|-----------|-------------------|
| `attn_gate_values` | bool | False | Gate aggregated values with input (AlphaFold2-style). Adds `nn.Linear(dim, out_dim)` with sigmoid gating. | Yes | **Tried (#30, #72, #130, #240):** Always worse (1.406-1.274). Slower throughput. Discard. |
| `attn_gate_value_heads` | bool | False | Per-head gating of output values ("Attend to nothing" paper). Adds `nn.Linear(dim, heads)`. | Yes | **Tried (#70, #229):** 1.372/1.218 -- slower, slightly worse. Discard. |
| `attn_swiglu_values` | bool | False | Use SiLU instead of sigmoid for value gating (only relevant if `attn_gate_values=True`). | Yes | **N/A.** attn_gate_values was discarded. |
| `attn_value_rmsnorm` | bool | False | RMSNorm on values. From AlphaGenome and ByteDance's GR3 for stability. | Yes | **Tried (#62, #228):** 1.386/1.220 -- marginal or worse, slower. Discard. |
| `attn_laser` | bool | False | **LASER**: exponentiate values for enhanced gradients ([arxiv 2411.03493](https://arxiv.org/abs/2411.03493)). `v = exp(softclamp(v))`, then `out = log(attn @ v)`. | Yes | **KEEP.** In current best since exp #71 (1.361). Removing confirmed harmful (#241, #119). |
| `attn_laser_softclamp_value` | float | 15.0 | Soft clamp value for LASER exponentiation. | Yes | **Tried 30 (#75):** 1.393 -- default 15 much better. Keep default. |

---

## 8. Attention -- Sparse & Alternative Attention Functions

These are **incompatible with flash attention** (enforced by assertions in `Attend.__init__`).

| Parameter | Type | Default | Description | Flash OK? | Experiment Status |
|-----------|------|---------|-------------|-----------|-------------------|
| `attn_sparse_topk` | int | None | Keep only top-k attention values before softmax. [Zhao et al. 2019](https://arxiv.org/abs/1912.11637) | **No** | **Untried.** Flash incompatible. Would need `attn_flash=False` and lose speed. Not recommended. |
| `attn_sparse_topk_straight_through` | bool | False | Straight-through gradients for sparse topk ([arxiv 2505.22074](https://arxiv.org/abs/2505.22074)). | **No** | -- |
| `attn_hard` | bool | False | Extreme case: only propagate single argmax value. | **No** | **Untried.** Unlikely to help. |
| `attn_sigmoid` | bool | False | Sigmoid attention (instead of softmax). Recent research direction. | **No** | **Untried.** Flash incompatible -- would lose a lot of speed. Not recommended unless flash support is added. |
| `attn_l2_distance` | bool | False | L2 distance attention (instead of dot product). | **No** | **Untried.** Flash incompatible. |
| `attn_gumbel_softmax` | bool | False | Gumbel-softmax attention. | N/A | **Untried.** Research curiosity, unlikely to help. |
| `attn_gumbel_softmax_temp` | float | 1.0 | Temperature for Gumbel-softmax. | -- | -- |
| `attn_gumbel_softmax_hard` | bool | True | Hard Gumbel-softmax. | -- | -- |
| `attn_selective` | bool | False | Selective attention (causal only). **Disables KV cache.** | **No** | **Untried.** Flash incompatible and disables KV cache. Not recommended. |
| `attn_cog_signed` | bool | False | CoG attention -- allows negative attention weights for expressiveness. | **No** | **Untried.** Flash incompatible. |
| `attn_custom_attn_fn` | Callable | None | Custom attention function replacing softmax. | Depends | **N/A.** For custom research. |

---

## 9. Attention -- Positional & Bias Mechanisms

| Parameter | Type | Default | Description | Flash OK? | Experiment Status |
|-----------|------|---------|-------------|-----------|-------------------|
| `attn_softclamp_logits` | bool | False | Soft clamp attention logits (Gemma 2 style). `tanh(logits / clamp_value) * clamp_value`. | **No** | **Tried (#230):** Crash -- incompatible with flash attn. |
| `attn_logit_softclamp_value` | float | 50.0 | Value for logit soft clamping. | -- | -- |
| `attn_use_cope` | bool | False | **Contextual Positional Encoding** (CoPE, [arxiv 2405.18719](https://arxiv.org/abs/2405.18719)). Positions are derived from attention gates (cumsum of sigmoid). | **No** | **UNTRIED.** Flash incompatible -- would lose speed. Could be powerful but needs `attn_flash=False`. See recommendations. |
| `attn_cope_max_pos` | int | 16 | Max position for CoPE embeddings. | -- | -- |
| `attn_cope_soft_onehot_pos` | bool | False | Use soft one-hot positions for CoPE (vs interpolation). | -- | -- |
| `attn_cope_talking_heads` | bool | False | Add talking heads variant inside CoPE. | -- | -- |
| `attn_data_dependent_alibi` | bool | False | **Forgetting Transformers** ([arxiv 2412.12847](https://openreview.net/forum?id=q2Lnyegkr8)). Data-dependent positional decay. Mutually exclusive with other pos biases. | **Compile issue** | **Tried (#83):** Crash -- torch.compile shape error. |
| `attn_data_dependent_alibi_per_row` | bool | False | Per-row variant using Q/K projections for forget gates. More expressive than global. | -- | -- |
| `attn_data_dependent_alibi_per_row_dim_head` | int | 8 | Dim per head for per-row forget gate Q/K. | -- | -- |
| `attn_add_zero_kv` | bool | False | Add zero key/value pair ("attention is off by one", Evan Miller). | **Compile issue** | **Tried (#64):** Crash -- torch.compile shape error mid-run. |
| `attn_head_learned_sink` | bool | False | Learned sink token per head. Working solution from gpt-oss project. | **No** | **UNTRIED.** Flash incompatible (assertion in Attend). Would need `attn_flash=False`. |

---

## 10. Attention -- Advanced / Experimental

| Parameter | Type | Default | Description | Flash OK? | Experiment Status |
|-----------|------|---------|-------------|-----------|-------------------|
| `attn_pre_talking_heads` | bool | False | Linear mixing across heads pre-softmax (Talking Heads). [Shazeer et al. 2020](https://arxiv.org/abs/2003.02436) | **No** | **Tried (#74):** Crash -- incompatible with flash attention. |
| `attn_post_talking_heads` | bool | False | Linear mixing across heads post-softmax. | **No** | -- |
| `attn_pre_scale_post_talking_heads` | bool | False | Combine pre-softmax heads, then scale post-softmax. Variant of talking heads. | **No** | **Untried.** Flash incompatible. |
| `attn_orthog_projected_values` | bool | False | **"Belief attention"** -- return orthogonal projected weighted values on original values. ICLR 2026. Doubles output dim of `to_out` (more params). | Yes | **UNTRIED.** Adds compute. Promising research but may hurt throughput. Medium priority. |
| `attn_orthog_projected_values_per_head` | bool | False | Per-head variant of belief attention. Can combine with above. | Yes | **UNTRIED.** Same considerations. |
| `attn_hybrid_module` | Module | None | Hymba-style hybrid attention + SSM module ([arxiv 2411.13676](https://arxiv.org/abs/2411.13676)). Runs another module (e.g. Mamba) alongside attention and mixes outputs. | Yes | **Untried.** Requires external module (Mamba). Complex integration. |
| `attn_hybrid_learned_mix` | bool | False | Learned mixing between attention and hybrid module. | -- | -- |
| `attn_hybrid_mask_kwarg` | str | None | Kwarg name to forward mask to hybrid module. | -- | -- |
| `attn_hybrid_fold_axial_dim` | int | None | Fold axial dimension for hybrid module. | -- | -- |
| `attn_onnxable` | bool | False | ONNX-compatible attention implementation. | -- | **N/A.** For deployment, not training. |
| `attn_attend_sdp_kwargs` | dict | `{enable_flash:True, enable_math:True, enable_mem_efficient:True}` | PyTorch SDP backend kwargs. Could disable specific backends. | -- | **Untried.** Could try forcing only flash backend. Low priority. |
| `attn_flash_pack_seq` | bool | False | Efficient packed sequence masking for variable length sequences. Requires `flash-attn` package and SM80+ GPU. | Flash-attn | **N/A.** For variable-length batches. Our dataset uses fixed-length sequences. |

---

## 11. Positional Encoding (Decoder)

Only **one** of rotary, polar, alibi, rel_pos_bias, or dynamic_pos_bias can be active.

| Parameter | Type | Default | Description | Experiment Status |
|-----------|------|---------|-------------|-------------------|
| `rotary_pos_emb` | bool | False | **RoPE** (Rotary Positional Embeddings). Standard for modern transformers. [Su et al. 2021](https://arxiv.org/abs/2104.09864) | **KEEP.** In current best. |
| `rotary_xpos` | bool | False | Modified RoPE for length extrapolation (adds ALiBi-like decay). [Sun et al. 2022](https://arxiv.org/abs/2212.10554) | **KEEP.** In current best since exp #154 (1.224). |
| `rotary_xpos_scale_base` | int | 512 | Receptive field scale for rotary_xpos. | **Tried 1024, 2048, 4096 (#177, #155, #244):** Default 512 is best. |
| `rotary_emb_dim` | int | None (dim_head//2) | Dimension for rotary embeddings. With dim_head=64, default is 32. | **UNTRIED.** Could try dim_head (=64) for full rotary. Source warns < 32 is bad. See recommendations. |
| `rotary_interpolation_factor` | float | 1.0 | Interpolation factor for extending context (NTK-aware scaling). | **Untried.** Only relevant for context extrapolation beyond training length. |
| `rotary_base_rescale_factor` | float | 1.0 | Base rescale factor for RoPE (NTK-aware). `base *= factor^(dim/(dim-2))`. | **Untried.** Same as above. |
| `rotate_num_heads` | int | None (=heads) | Only apply rotary to this many heads (for decoupled RoPE in MLA). Others get no position info. | **N/A.** Only for MLA. |
| `polar_pos_emb` | bool | False | **PoPE** -- Polar Positional Embedding ([arxiv 2509.10534](https://arxiv.org/abs/2509.10534)). Mutually exclusive with RoPE. | **Tried (#118):** 1.697 -- much worse than RoPE. Discard. |
| `polar_bias_uniform_init` | bool | False | Uniform init for polar pos bias in [-2pi, 0]. | -- | -- |
| `rel_pos_bias` | bool | False | T5-style learned relative position bias. [Raffel et al. 2020](https://arxiv.org/abs/1910.10683) | **No (flash incompatible)** | **Untried.** Incompatible with flash attention. |
| `rel_pos_num_buckets` | int | 32 | Number of buckets for T5 rel pos bias. | -- | -- |
| `rel_pos_max_distance` | int | 128 | Max distance for T5 rel pos bias. | -- | -- |
| `dynamic_pos_bias` | bool | False | Learned MLP position bias. [CrossFormer](https://arxiv.org/abs/2108.00154), [SwinV2](https://arxiv.org/abs/2111.09883) | **No (flash incompatible)** | **Untried.** Incompatible with flash attention. |
| `dynamic_pos_bias_log_distance` | bool | False | Use log distances (linear is better for language). | -- | -- |
| `dynamic_pos_bias_mlp_depth` | int | 2 | MLP depth for dynamic pos bias. | -- | -- |
| `dynamic_pos_bias_norm` | bool | False | Normalization for dynamic pos bias MLP. | -- | -- |
| `alibi_pos_bias` | bool | False | ALiBi: static linear bias. Hinders global attention. [Press et al. 2021](https://ofir.io/train_short_test_long.pdf) | **No (flash incompatible)** | **Untried.** RoPE is clearly superior for our setup. |
| `alibi_num_heads` | int | None (=heads) | Only apply ALiBi to this many heads. | -- | -- |

---

## 12. Residual Connections (Decoder)

| Parameter | Type | Default | Description | Experiment Status |
|-----------|------|---------|-------------|-------------------|
| `gate_residual` | bool | False | GRU-gated residual connections. Mutually exclusive with `num_residual_streams > 1`. [Parisotto et al. 2019](https://arxiv.org/abs/1910.06764) | **Tried (#35, #171):** 1.607/4.234 -- catastrophic. Discard. |
| `scale_residual` | bool | False | Learned residual scaling (Normformer). [Normformer 2022](https://openreview.net/forum?id=GMYWzWztDx5) | **Tried (#38, #173):** 1.401/1.223 -- slight overhead, worse. Discard. |
| `scale_residual_constant` | float | 1.0 | Constant multiplier for residual scaling. | **Untried.** Since scale_residual was discarded, N/A. |
| `add_value_residual` | bool | False | **ResFormer** value residuals -- add first layer's attention values to all subsequent layers. ([arxiv 2410.17897](https://arxiv.org/abs/2410.17897)). | **KEEP.** In current best since exp #10 (1.434). Removing confirmed harmful (#49). |
| `learned_value_residual_mix` | bool | **True** | Per-token learned mixing for value residual (data-dependent). When `add_value_residual=True`, adds `nn.Linear(dim, heads) + Sigmoid` per attention layer. Credit: @Blinkdl. | **Active by default** (True when add_value_residual=True). **UNTRIED setting False.** Could test `learned_value_residual_mix=False` for fixed 0.5 mix -- saves a tiny amount of compute. |
| `residual_attn` | bool | False | Residualize pre-softmax attention scores across layers. [He et al. 2020](https://arxiv.org/abs/2012.11747) | **No (flash incompatible)** | **Not applicable.** Incompatible with flash attention. |
| `cross_residual_attn` | bool | False | Same for cross-attention. | -- | **N/A.** No cross-attention. |
| `softclamp_output` | bool | False | Soft-clamp final hidden states before final norm (Gemma 2 style). `tanh(x/value) * value`. | **KEEP.** In current best since exp #57 (1.385). Removing confirmed harmful (#209). |
| `softclamp_output_value` | float | 30.0 | Soft clamp value. | **Tried 50 (#248):** 1.214 -- default 30 is better. |
| `zero_init_branch_output` | bool | False | Zero-init output projections of both attention and feedforward. GPT-NeoX style. | **KEEP.** In current best since exp #57 (1.385). |
| `reinject_input` | bool | False | DEQ-style input reinjection at every layer. Adds `nn.Linear(dim, dim)` projection. | **Tried (#76):** 1.375 -- slightly worse than baseline at that point. |
| `learned_reinject_input_gate` | bool | False | Learned gate for input reinjection. Adds `nn.Linear(dim, 1)` -> sigmoid. | **Untried standalone.** Was not tested independently of reinject_input. |

---

## 13. Layer Structure (Decoder)

| Parameter | Type | Default | Description | Experiment Status |
|-----------|------|---------|-------------|-------------------|
| `macaron` | bool | False | **Macaron**: FFN-Attn-FFN sandwich structure. Based on dynamical systems POV. [Lu et al. 2019](https://arxiv.org/abs/1906.02762), [Conformer](https://arxiv.org/abs/2005.08100) | **Tried (#15, #167):** 1.451/1.265 -- too slow (nearly doubles FFN params). Discard. |
| `sandwich_coef` | int | None | Sandwich layer reordering: `(attn)*coef + (attn,ff)*(depth-coef) + (ff)*coef`. [Press et al. 2020](https://arxiv.org/abs/1911.03864) | **Tried 2,6 (#121, #243):** 1.289/1.223 -- worse than standard ordering. Discard. |
| `weight_tie_layers` | bool | False | Tie weights across all layers (ALBERT-style). Dramatically fewer params. [Lan et al. 2019](https://arxiv.org/abs/1909.11942) | **Tried (#51, #52):** 1.499/1.501 -- much worse. Too few effective params. Discard. |
| `custom_layers` | tuple[str,...] | None | Custom layer sequence, e.g. `('a', 'f', 'a', 'f')` or `('a', 'f', 'f')` for 2:1 FFN ratio. | **UNTRIED.** Interesting -- could try more FFN layers per attention layer. See recommendations. |
| `layers_execute_order` | tuple[int,...] | None | Custom execution order of layers (0-indexed). Allows weight-sharing patterns. Can also be passed at forward time for depth extrapolation. | **Untried.** Complex. Low priority. |
| `par_ratio` | int | None | PAR (Parallel Attention/FF) ratio. Changes layer interleaving pattern. | **Untried.** Exotic. Low priority. |
| `shift_tokens` | int or tuple | 0 | Shift a subset of feature dimensions by 1 token. **Helps char-level training.** For causal, shifts range `[0, shift_tokens]`. [PENG Bo 2021](https://zhuanlan.zhihu.com/p/191393788) | **KEEP.** shift_tokens=1 in current best. Tried 2 (#193): worse. Removing confirmed harmful (#46, #157). |
| `unet_skips` | bool | False | U-Net style skip connections across layers. Latter-half layers get concatenated with earlier-half. | **Tried (#68, #211):** 1.436/1.280 -- much worse. Discard. |
| `layer_dropout` | float | 0.0 | **Stochastic depth**: randomly drop entire layers during training. | **Tried 0.05 (#48):** 1.434 -- hurts short training. Discard. |
| `use_layerscale` | bool | False | LayerScale (from CaiT/DeiT). Per-layer learned scaling of branch output. | **Tried (#60, #178):** 1.418/1.231 -- conflicts with zero_init_branch_output. Discard. |
| `layerscale_init_value` | float | 0.0 | Init value for LayerScale gamma. | -- | -- |

---

## 14. Multi-Stream / Cross-Layer (Decoder)

These are advanced architectural modifications that add significant complexity.

| Parameter | Type | Default | Description | Experiment Status |
|-----------|------|---------|-------------|-------------------|
| `num_residual_streams` | int | 1 | **Hyper-Connections** ([arxiv 2409.19606](https://arxiv.org/abs/2409.19606)). >1 creates multiple residual streams with learned mixing (Sinkhorn-constrained). Multiplies effective batch size by `num_residual_streams`. | **Tried 2 (#58, #195):** 3.506/2.062 -- catastrophic. Too slow, didn't converge. Discard. |
| `qkv_receive_diff_residuals` | bool | False | Q/K/V receive different residual stream views. Only works with hyper-connections or LIMe. | **N/A.** Requires multi-stream. |
| `hyper_conn_sinkhorn_iters` | int | 5 | Sinkhorn iterations for hyper-connection mixing matrix constraint. | -- | -- |
| `integrate_layers` | bool | False | **LIMe** -- Layer Integrated Memory ([arxiv 2502.09245](https://arxiv.org/abs/2502.09245)). Each layer dynamically weights all past layer hidden states. Adds `nn.Linear(dim, num_views * num_layers)` per layer. | **Tried (#59, #194):** 1.512/OOM -- too slow/OOM. Discard. |
| `layer_integrate_use_softmax` | bool | True | Use softmax (vs ReLU) for LIMe layer weights. | -- | -- |

---

## 15. TransformerWrapper Options

| Parameter | Type | Default | Description | Experiment Status |
|-----------|------|---------|-------------|-------------------|
| `num_tokens` | int | required | Vocabulary size (256 for enwik8). | **Fixed.** |
| `max_seq_len` | int | required | Maximum sequence length. | **Current: 4096.** Tried 512-8192. 4096 is optimal. |
| `use_abs_pos_emb` | bool | True | Absolute positional embeddings. Auto-disabled when RoPE is active (via `disable_abs_pos_emb`). | **Tried removing (#238):** 1.209 -- RoPE only is slightly worse. Keeping default (auto-disabled with RoPE). |
| `scaled_sinu_pos_emb` | bool | False | Scaled sinusoidal positional embeddings (learned scale). Alternative to learned absolute. | **Untried.** RoPE handles positions; this is redundant. |
| `l2norm_embed` | bool | False | L2-normalize embeddings + small init (fixnorm). | **Tried (#16, #245):** 1.425/1.216 -- post_emb_norm is better. Discard. |
| `post_emb_norm` | bool | False | LayerNorm right after embeddings (BLOOM/YaLM-style). | **KEEP.** In current best since exp #39 (1.390). |
| `emb_frac_gradient` | float | 1.0 | Fraction of gradient flowing to embeddings. Set 0.1 for GLM-130B/CogView style. `x = x * frac + x.detach() * (1-frac)`. | **KEEP at 0.1.** In current best since exp #66 (1.369). Tried 0.05, 0.15, 0.5 -- 0.1 is optimal. |
| `emb_dropout` | float | 0.0 | Dropout after embedding. | **Tried 0.1 (#47):** 1.400 -- hurts short training. |
| `emb_dim` | int | None (=dim) | Embedding dimension (can differ from model dim; projects if needed). | **Untried.** Could try larger emb_dim with projection. Low priority. |
| `tie_embedding` | bool | False | Tie input/output embeddings. Output becomes `t @ token_emb.weight.T`. | **UNTRIED.** With vocab=256, embedding is tiny (256 x 448 = 115K params, 0.6% of total). Minimal impact expected. Low priority. |
| `ff_deep_embed` | bool | False | **Deep embeddings** from nanogpt-speedrun / RWKV 8. Learns per-token, per-layer embedding that multiplies FFN output. Adds `nn.Parameter(ones(num_tokens, depth, dim))`. | **Tried (#61):** 1.411 -- extra params didn't help. Discard. |
| `num_memory_tokens` | int | None | Learned memory/register tokens passed through all layers. [Burtsev 2020](https://arxiv.org/abs/2006.11527), [Darcet et al. 2023](https://arxiv.org/abs/2309.16588) | **Tried 4 (#122, #237):** 1.262/1.217 -- slight overhead, worse. Discard. |
| `memory_tokens_interspersed_every` | int | None | Intersperse memory tokens every N positions (decoder only). | **Untried.** Since num_memory_tokens was discarded, N/A. |
| `max_mem_len` | int | 0 | Transformer-XL recurrence memory length. | **Untried.** Complex to set up properly. Could help with context but adds memory overhead. |
| `shift_mem_down` | int | 0 | Enhanced recurrence: route memory of layer N to layer N-1 on next step. [Ding et al. 2021](https://arxiv.org/abs/2012.15688) | **Untried.** Requires max_mem_len > 0. |
| `mixture_of_softmax` | bool | False | **Mixture of Softmax** output head. Multiple softmax components with learned mixing. | **Tried (#114):** Crash -- torch.compile dtype mismatch BF16 vs Float. |
| `mixture_of_softmax_k` | int | 4 | Number of softmax mixtures. | -- | -- |
| `sigsoftmax_logits` | bool | False | Sigmoid-Softmax logits: `softmax(x) * sigmoid(x)`. | **Tried (#63):** 1.404 -- didn't help. Discard. |
| `recycling` | bool | False | AlphaFold2-style recycling -- multiple forward passes reusing output. | **Untried.** Would drastically reduce throughput. Not recommended. |
| `train_max_recycle_steps` | int | 4 | Max recycle steps during training. | -- | -- |
| `add_continuous_pred_head` | bool | False | Add auxiliary continuous embedding prediction head. Predicts next token's embedding alongside logits. | **Untried.** Adds auxiliary loss. Interesting but complex. Low priority. |
| `input_not_include_cache` | bool | False | Whether input includes past tokens from cache. For generation optimization. | **N/A.** For inference. |

---

## 16. AutoregressiveWrapper Options

| Parameter | Type | Default | Description | Experiment Status |
|-----------|------|---------|-------------|-------------------|
| `mask_prob` | float | 0.0 | **Forgetful Causal Masking**: randomly mask tokens during AR training (MLM-style). Paper uses 0.15. [Liu et al. 2022](https://arxiv.org/abs/2210.13432) | **Tried 0.15 (#25, #236):** 1.472/OOM. Slows flash attn significantly; OOM at seq=4096. Discard. |
| `add_attn_z_loss` | bool | False | Add attention z-loss for training stability. Weight from TransformerWrapper `attn_z_loss_weight` (default 1e-4). | **UNTRIED.** Small regularization signal. Might help stability slightly. Low priority. |
| `next_embed_loss_weight` | float | 0.1 | Weight for continuous embedding prediction loss (requires `add_continuous_pred_head=True` on TransformerWrapper). | **N/A.** Requires add_continuous_pred_head. |

---

## 17. Flash Attention Compatibility Matrix

This is critical -- many parameters crash or silently fail with `attn_flash=True`.

### Incompatible with `attn_flash=True` (enforced by assertions)

| Parameter | Error Type | Notes |
|-----------|------------|-------|
| `attn_sigmoid=True` | Assertion | Use softmax with flash |
| `attn_hard=True` | Assertion | |
| `attn_sparse_topk` | Assertion | |
| `attn_pre_talking_heads` | Assertion | |
| `attn_post_talking_heads` | Assertion | |
| `attn_pre_scale_post_talking_heads` | Assertion | |
| `attn_selective=True` | Assertion | |
| `attn_cog_signed=True` | Assertion | |
| `attn_softclamp_logits=True` | Assertion | Gemma 2 logit clamp |
| `attn_head_learned_sink=True` | Assertion | gpt-oss sink tokens |
| `attn_use_cope=True` | Assertion | CoPE positions |
| `residual_attn=True` | Assertion | Needs pre-softmax attention |
| `rel_pos_bias=True` | Assertion | T5 relative position bias |
| `dynamic_pos_bias=True` | Assertion | Dynamic position bias |

### Problematic with `torch.compile(dynamic=False)`

These work with flash attention but crash with torch.compile due to dynamic shapes:

| Parameter | Error | Tested |
|-----------|-------|--------|
| `attn_num_mem_kv > 0` | Shape error at seq=4096 | Exp #168 |
| `attn_add_zero_kv=True` | Shape error mid-run | Exp #64 |
| `attn_data_dependent_alibi=True` | Shape error | Exp #83 |
| `mixture_of_softmax=True` | Dtype mismatch BF16/Float | Exp #114 |
| `attn_on_attn=True` + `zero_init_branch_output` | Sequential has no .weight | Exp #67 |
| `attn_kv_heads` + `add_value_residual` | Value residual conflict | Exp #56 |
| `mask_prob > 0` | OOM/shape mismatch at seq=4096 | Exp #236 |

### Safe with flash attention + torch.compile

Everything in the current best config, plus: `attn_laser`, `attn_orthog_projected_values`, `attn_gate_values`, `attn_gate_value_heads`, `attn_value_rmsnorm`, `attn_head_scale`, `attn_use_latent_q/kv`, all `ff_*` params, all normalization params, all residual params.

---

## 18. Untried Parameters -- Prioritized Recommendations

Based on 247 experiments, here's what's left to try, ordered by expected impact and risk.

### Tier 1: High Potential, Safe, Untried

| # | Idea | Parameters | Rationale | Risk | Expected Impact |
|---|------|-----------|-----------|------|-----------------|
| 1 | **Full rotary dimension** | `rotary_emb_dim=64` (= dim_head) | Default is dim_head//2 = 32. Using full dim means all features get position info. May improve long-range modeling at seq=4096. Zero extra compute. | Very Low | Small positive or neutral |
| 2 | **Belief attention** | `attn_orthog_projected_values=True` | ICLR 2026 paper. Returns orthogonal projection of weighted values onto original values. Doubles output dim of `to_out` (more params). Flash compatible. | Medium | Unknown -- novel technique |
| 3 | **Custom layer pattern (more FFN)** | `custom_layers=('a','f','f') * 4` | Instead of 6x(attn,ff), try 4x(attn,ff,ff). Same depth but 2:1 FFN:attn ratio. FFN is cheaper than attn, so similar throughput. | Low | Small positive or neutral |
| 4 | **Custom layer pattern (more attn)** | `custom_layers=('a','a','f') * 4` | Try 2:1 attn:ffn ratio instead. More attention capacity. | Low | Small positive or neutral |
| 5 | **Disable learned_value_residual_mix** | `learned_value_residual_mix=False` (need to modify source call) | Currently True by default. Saves a tiny `nn.Linear(dim, heads)` per attn layer. Fixed 0.5 mix might be equally good. | Very Low | Slight throughput gain if quality matches |

### Tier 2: Medium Potential, Worth Testing

| # | Idea | Parameters | Rationale | Risk |
|---|------|-----------|-----------|------|
| 6 | **SoLU activation** | `ff_solu=True` (remove ff_swish, ff_relu_squared) | Anthropic's Softmax Linear Unit. Adds LayerNorm inside FFN. Different from SwiGLU. Still compatible with ff_glu. Mutually exclusive with ff_relu_squared (but not with ff_swish behavior since SoLU replaces the activation). Note: with ff_glu=True, need to check interaction. | Medium |
| 7 | **Per-head belief attention** | `attn_orthog_projected_values_per_head=True` | Per-head variant, can be combined with global variant. Even more expressive but adds more params. | Medium |
| 8 | **Transformer-XL recurrence** | TransformerWrapper: `max_mem_len=4096` | Extends effective context by carrying KV cache between segments. Complex but could improve BPC. Requires `rotary_pos_emb=True` (already active). | High (complex) |
| 9 | **Attention sublayer dropout** | `attn_sublayer_dropout=0.05` | Unlike attn_dropout (on weights), this is after the output projection. May provide mild regularization without flash attn conflict. | Low |
| 10 | **Multi-Latent Attention (MLA)** | `attn_use_latent_kv=True, attn_dim_latent_kv=128` | DeepSeek's MLA compresses KV through a bottleneck. Saves memory -> could fit larger batch. But conflicts with `attn_kv_heads` and possibly `add_value_residual`. | High |

### Tier 3: Training Strategy Changes (Non-Parameter)

| # | Idea | Implementation | Rationale |
|---|------|---------------|-----------|
| 11 | **Two-phase training** | Short context high-throughput first, then long context | See more tokens in phase 1, then fine-tune with long context. Requires modifying the training loop. |
| 12 | **Progressive batch size** | Start batch=8, ramp to batch=32 over training | Small batch = more updates early; large batch = lower noise late. |
| 13 | **torch.compile(mode='max-autotune')** | Change compile options | May find faster kernels. Longer compilation time but potentially faster training. |
| 14 | **Gradient accumulation schedule** | accum=2 for first half, accum=1 for second half | Double effective batch early, more frequent updates late. |

### Tier 4: Nuclear Options (if Stuck)

| # | Idea | Parameters | Rationale | Risk |
|---|------|-----------|-----------|------|
| 15 | **Disable flash attention + enable CoPE** | `attn_flash=False, attn_use_cope=True, attn_cope_max_pos=64` | CoPE adds content-dependent position info. May substantially help char-level modeling. But losing flash attention costs ~30-50% throughput. Only try if per-step quality gain outweighs fewer steps. | Very High |
| 16 | **Disable flash + enable head_learned_sink** | `attn_flash=False, attn_head_learned_sink=True` | Sink tokens fix attention outlier problems. | Very High |
| 17 | **Completely different architecture** | dim=640, depth=4, heads=10, ff_mult=3 | Wider, shallower, narrower FFN. Different throughput/quality tradeoff. | High |
| 18 | **AdamW instead of Muon** | Replace optimizer entirely | Muon has been good but maybe a different optimizer landscape exists. | High |

---

## 19. Training Strategy Ideas (Non-Parameter)

These don't require changing x-transformers parameters but can break through plateaus.

### Two-Phase Context Length Schedule

The single highest-impact non-parameter change. Current bottleneck may be total tokens seen.

```python
# Phase 1: Short context, high throughput
MAX_SEQ_LEN = 1024
BATCH_SIZE = 96  # or whatever fits in VRAM

# Phase 2: Long context, fine-tune
MAX_SEQ_LEN = 4096
BATCH_SIZE = 24
```

Estimated improvement: Phase 1 would see ~3x more tokens. The long-context phase then fine-tunes the model to exploit the 4096 context. This requires significant train.py modifications but is the most promising avenue.

### Progressive Model Scaling

Train a small model (dim=256) first, then "grow" it to dim=448. Weight initialization from the smaller model can be done via zero-padding or projection. Very experimental.

### Ensemble Evaluation

Run 2 different models, average their logits at eval time. Not really fair to the spirit of the experiment but technically within the rules.

### Curriculum Learning

Start with easy data (English text portions of enwik8), then train on the full distribution. Hard to implement since enwik8 is XML/HTML with varying difficulty.

---

## Recommended Combinations

Based on the x-transformers README and common practice in modern LLMs:

### Baseline (modern defaults)
```python
Decoder(
    dim=512, depth=6, heads=8,
    rotary_pos_emb=True,
    attn_flash=True,
    attn_qk_norm=True,
    use_rmsnorm=True,
    ff_relu_squared=True,
)
```

### SwiGLU (PaLM/LLaMA-style)
```python
Decoder(
    dim=512, depth=6, heads=8,
    rotary_pos_emb=True,
    attn_flash=True,
    attn_qk_norm=True,
    use_rmsnorm=True,
    ff_glu=True,        # enable gating
    ff_swish=True,       # Swish activation
    ff_no_bias=True,     # no bias (PaLM style)
)
```

### Memory-efficient (GQA)
```python
Decoder(
    dim=768, depth=8, heads=12,
    attn_kv_heads=4,     # grouped-query attention
    rotary_pos_emb=True,
    attn_flash=True,
    attn_qk_norm=True,
    use_rmsnorm=True,
    ff_glu=True,
    ff_swish=True,
)
```

### Kitchen sink (many features)
```python
Decoder(
    dim=640, depth=8, heads=10,
    rotary_pos_emb=True,
    attn_flash=True,
    attn_qk_norm=True,
    use_rmsnorm=True,
    ff_glu=True,
    ff_swish=True,
    macaron=True,            # sandwich FFN
    attn_num_mem_kv=16,      # persistent memory
    shift_tokens=1,          # good for char-level
    sandwich_norm=True,      # extra stability
)
```

---

## Model Sizing Guide

Phil Wang's guideline: **1:5 model-to-data ratio** (tokens seen = 5 x params).

Token throughput depends on your GPU. Rough estimates:

| Params | dim | depth | heads | Tokens needed (5x) | Notes |
|--------|-----|-------|-------|--------------------:|-------|
| ~5M | 256 | 6 | 4 | 25M | Very small, fast iteration |
| ~19M | 512 | 6 | 8 | 95M | Good starting point for 6-8 GB VRAM |
| ~57M | 768 | 8 | 12 | 285M | Needs 12+ GB VRAM |
| ~64M | 640 | 12 | 10 | 320M | Deeper, narrower variant |

Based on our 247 experiments on RTX 4090:

| dim | depth | heads | Params | Tokens seen | VRAM | Best BPC | Notes |
|-----|-------|-------|--------|-------------|------|----------|-------|
| 256 | 8 | 4 | 8.5M | ~155M | 1.4 GB | 1.396 | Good for rapid iteration |
| 384 | 6 | 6 | 14.4M | ~154M | 4.2 GB | 1.265 | |
| **448** | **6** | **7** | **19.6M** | **~118M** | **14.3 GB** | **1.208** | **Current sweet spot** |
| 512 | 6 | 8 | 25.5M | ~112M | ~6 GB | 1.271 | Too big, fewer tokens |
| 640 | 6 | 10 | 40M | ~80M | 5.1 GB | 1.765 | Way too big |
| 768 | 8 | 12 | 57M | ~60M | 4.4 GB | 1.871 | Far too big |

**Key insight**: The optimal model size is determined by the throughput/quality tradeoff. At seq=4096, batch=24, accum=1, the dim=448 model processes ~118M tokens -- roughly 6x its parameter count, close to the 5x Chinchilla ratio.

**VRAM usage** is primarily driven by `dim`, `depth`, `batch_size`, and `max_seq_len`.
Using `attn_kv_heads` (GQA) can significantly reduce attention memory for larger models.
