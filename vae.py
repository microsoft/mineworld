import torch
import torch.nn as nn
import diffusers
from safetensors.torch import load_file as load_safetensors
from utils import print0, get_valid_paths, tensor_to_uint8


class VAE(nn.Module):
    def __init__(self,
                 config_path: str,
                 ckpt_path: str,
    ):
        super().__init__()
        config_path = get_valid_paths(config_path)
        print0(f"[bold magenta]\[VAE][/bold magenta] Loading VQGAN from {config_path}")
        self.model = diffusers.VQModel.from_config(config_path)

        ckpt_path = get_valid_paths(ckpt_path)
        print0(f"[bold magenta]\[VAE][/bold magenta] Use ckpt_path: {ckpt_path}")
        self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(
        self, path: str
    ) -> None:
        if path.endswith("ckpt"):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if "state_dict" in ckpt:
                weights = ckpt["state_dict"]
            else:
                weights = ckpt
        elif path.endswith("safetensors"):
            weights = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(weights, strict=False)
        print0(
            f"[bold magenta]\[tvae.models.amused_vqvae][AutoencodingLegacy][/bold magenta] Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print0(f"[bold magenta]\[tvae.models.amused_vqvae][AutoencodingLegacy][/bold magenta] Missing Keys: {missing}")
        # if len(unexpected) > 0:
        #     print0(f"[bold magenta]\[tvae.models.amused_vqvae][AutoencodingLegacy][/bold magenta] Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def tokenize_images(self, x: torch.Tensor, sane_index_shape: bool = True):
        h = self.model.encoder(x)
        h = self.model.quant_conv(h)
        if sane_index_shape:
            orig_sane_index_shape = self.model.quantize.sane_index_shape
            self.model.quantize.sane_index_shape = True
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.model.quantize(h)
        if sane_index_shape:
            self.model.quantize.sane_index_shape = orig_sane_index_shape
        return min_encoding_indices
    
    # yang ye
    @torch.no_grad()
    def token2image(self, tokens):
        assert tokens.max() < 8192, f"code max value is {tokens.max()}"
        shape = (1, 14, 24, 64) 
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            quant = self.model.quantize.get_codebook_entry(tokens, shape)
            quant2 = self.model.post_quant_conv(quant)
            dec = self.model.decoder(quant2)
        img = tensor_to_uint8(dec[0]).transpose(1, 2, 0)
        return img