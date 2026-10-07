"""Pack spec for MiMo-V2.6-Pro-RL-uncensored (dealignai).

Geometry captured from the real checkpoint at
``/nas-1/models/mimo/mimo-v2.6-pro-rl-uncensored`` (see
``mods/exl3-mimo/pro-source/geometry.json``): 6144 hidden, 384 routed experts,
70 layers (69 MoE), 2048 moe-intermediate → 64 slots.
"""

from __future__ import annotations

from exl3pack.spec import Geometry, PackSpec

SPEC = PackSpec(
    slug="mimo-v2.6-pro-rl-uncensored",
    geometry=Geometry(
        architecture="MiMoV2ForCausalLM",
        hidden_size=6144,
        intermediate_size=2048,
        num_experts=384,
        num_slots=64,
        moe_layer_count=69,
        num_hidden_layers=70,
        dense_layers=[0],
    ),
    codebook="mcg",
    bits=3,
    dense_format="bf16",
    ignored_layers=[
        "down_proj",
        "eh_proj",
        "gate",
        "gate_proj",
        "k_proj",
        "lm_head",
        "o_proj",
        "q_proj",
        "qkv_proj",
        "up_proj",
        "v_proj",
    ],
    node_map_key="mimo-v2.6-pro-rl-uncensored",
)
