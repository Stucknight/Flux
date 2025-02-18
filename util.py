import torch

from model import Flux, FluxParams
from modules.autoencoder import AutoEncoder, AutoEncoderParams
from modules.conditioner import HFEmbedder

flux_configs = {
    "params": FluxParams(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=2.0,
        num_heads=4,
        depth=2,
        depth_single_blocks=4,
        axes_dim=[96, 336, 336],
        theta=10_000,
        qkv_bias=True
    )
}

ae_configs = {
    "ae_params": AutoEncoderParams(
        resolution=256,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )}

def load_flow_model():
    with torch.device("meta"):
        model = Flux(flux_configs["params"]).to(torch.bfloat16)
    return model


def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae():
    with torch.device("meta"):
        ae = AutoEncoder(ae_configs["ae_params"])
    return ae