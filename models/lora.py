import torch.nn as nn
import torch
import math

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0,dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank

        #Freeze baseline so weights do not update
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        #define the 2 low rank matrices as trainable parameters
        # lora_A: shape (rank, in_features)
        # lora_B: shape (out_features, rank)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))

        # 3. Optional dropout on input before applying low-rank projection
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        # 4. Initialize the weights!
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization for lora_A (similar to standard nn.Linear)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) #
        # Zero out lora_B so initial adapter output is strictly 0
        nn.init.zeros_(self.lora_B) # guarantees the model starts in the exact state as your pre-trained checkpoint without any glitching

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base frozen forward pass
        base_out = self.base_layer(x)

        # Low-rank forward pass: (x @ A.T) @ B.T * scaling
        # x is shape (B, T, in_features)
        lora_out = self.dropout(x)
        lora_out = torch.matmul(lora_out, self.lora_A.t())   # -> (B, T, rank)
        lora_out = torch.matmul(lora_out, self.lora_B.t())   # -> (B, T, out_features)
        lora_out = lora_out * self.scaling

        #  Combine both paths
        return base_out + lora_out

def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0, target_modules=("qkv", "proj")):

    #Freezes the base model and injects LoRALinear into the attention layers of every DITBlock.

    #Freeze every parameter in the entire base model
    for param in model.parameters():
        param.requires_grad = False

    #walk through all blocks and replace targeted linear layers
    injected_count = 0
    for i, block in enumerate(model.blocks):
        if hasattr(block, "attn"):
            if "qkv" in target_modules and hasattr(block.attn, "qkv"):
                block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
                injected_count += 1
            if "proj" in target_modules and hasattr(block.attn, "proj"):
                block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha, dropout=dropout)
                injected_count += 1

    #calculate and display parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = (trainable_params / total_params) * 100

    print(f"[LoRA Injection] Injected into {injected_count} linear layers across {len(model.blocks)} blocks.")
    print(f"[LoRA Injection] Total Parameters: {total_params:,}")
    print(f"[LoRA Injection] Trainable Parameters: {trainable_params:,} ({percent:.2f}%)")

    return model


def save_lora_weights(model: nn.Module, save_path: str):

    #Extracts and saves only the trainable LoRA parameters (lora_A and lora_B).

    lora_state_dict = {
        name: param.cpu()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    torch.save(lora_state_dict, save_path)
    import os
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[LoRA Saved] Saved {len(lora_state_dict)} tensors to '{save_path}' ({size_mb:.2f} MB)")


def load_lora_weights(model: nn.Module, load_path: str, device="cpu"):

    #Loads saved LoRA weights into an injected model.

    lora_state_dict = torch.load(load_path, map_location=device, weights_only=True)
    # strict=False allows loading into the full model where only lora keys match
    incompatible = model.load_state_dict(lora_state_dict, strict=False)
    print(f"[LoRA Loaded] Successfully loaded adapter from '{load_path}' (Loaded {len(lora_state_dict)} tensors)")
    return model


if __name__ == "__main__":
    import os
    from dit_modified import ContourRendererDiT

    print("--- 1. Testing single LoRALinear layer ---")
    base_linear = nn.Linear(768, 768)
    lora_linear = LoRALinear(base_layer=base_linear, rank=8, alpha=16.0)
    x = torch.randn(2, 32, 768)
    diff = (base_linear(x) - lora_linear(x)).abs().max().item()
    assert diff == 0.0, "LoRA output at init should strictly equal base layer output!"
    print(f"Max difference at init: {diff:.6f} -> PASSED!\n")

    print("--- 2. Testing LoRA injection on ContourRendererDiT ---")
    model = ContourRendererDiT(use_rope=True, use_instrument_embed=True)
    model = inject_lora(model, rank=8, alpha=16.0)

    # Test forward pass with dummy diffusion inputs
    x_latent = torch.randn(2, 32, 256)
    t = torch.randint(0, 1000, (2,))
    contours = torch.randn(2, 32, 6)
    inst_id = torch.tensor([7, 7]) # Violin

    out = model(x_latent, t, contours, instrument_id=inst_id)
    print(f"Forward pass output shape: {out.shape} -> PASSED!\n")

    print("--- 3. Testing Save and Load ---")
    test_file = "test_lora_weights.pt"
    save_lora_weights(model, test_file)
    load_lora_weights(model, test_file)
    if os.path.exists(test_file):
        os.remove(test_file)
    print("Save and Load -> PASSED!")