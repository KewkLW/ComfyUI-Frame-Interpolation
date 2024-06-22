import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from comfy.model_management import get_torch_device
from vfi_utils import InterpolationStateList, load_file_from_github_release, preprocess_frames, postprocess_frames

DEVICE = get_torch_device()

class FILM_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["film_net_fp32.pt"], ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
                "use_8bit_integers": ("BOOLEAN", {"default": False}),
                "use_gradient_checkpointing": ("BOOLEAN", {"default": False}),
                "optimize_pyramid_handling": ("BOOLEAN", {"default": False}),
                "use_memory_efficient_attention": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def __init__(self):
        self.model = None
        self.use_8bit_integers = False
        self.use_gradient_checkpointing = False
        self.optimize_pyramid_handling = False
        self.use_memory_efficient_attention = False

    def load_model(self, ckpt_name):
        model_path = load_file_from_github_release("film", ckpt_name)
        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()
        self.model = self.model.to(DEVICE)

    def quantize_activations(self, x):
        if self.use_8bit_integers:
            return torch.quantize_per_tensor(x, scale=1/255, zero_point=0, dtype=torch.quint8)
        return x

    def dequantize_activations(self, x):
        if self.use_8bit_integers:
            return x.dequantize()
        return x

    def optimize_pyramid_handling(self, pyramid):
        if self.optimize_pyramid_handling:
            return (level for level in pyramid)  # Generator instead of list
        return pyramid

    def memory_efficient_attention(self, q, k, v):
        if self.use_memory_efficient_attention:
            # Simplified memory-efficient attention
            attention = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
            attention = torch.softmax(attention, dim=-1)
            return torch.matmul(attention, v)
        else:
            # Standard attention
            return nn.functional.multi_head_attention_forward(q, k, v, embed_dim_to_check=q.size(-1))

    def vfi(
        self,
        ckpt_name: str,
        frames: torch.Tensor,
        clear_cache_after_n_frames: int = 10,
        multiplier: int = 2,
        use_8bit_integers: bool = False,
        use_gradient_checkpointing: bool = False,
        optimize_pyramid_handling: bool = False,
        use_memory_efficient_attention: bool = False,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        self.use_8bit_integers = use_8bit_integers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.optimize_pyramid_handling = optimize_pyramid_handling
        self.use_memory_efficient_attention = use_memory_efficient_attention

        if self.model is None:
            self.load_model(ckpt_name)

        interpolation_states = optional_interpolation_states
        frames = preprocess_frames(frames)
        number_of_frames_processed_since_last_cleared_cuda_cache = 0
        output_frames = []

        if isinstance(multiplier, int):
            multipliers = [multiplier] * len(frames)
        else:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * (len(frames) - len(multipliers) - 1)

        for frame_itr in range(len(frames) - 1):
            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
                continue

            frame_0 = frames[frame_itr:frame_itr+1].to(DEVICE).float()
            frame_1 = frames[frame_itr+1:frame_itr+2].to(DEVICE).float()

            # Dequantize before any operation that doesn't support quantized tensors
            frame_0 = self.dequantize_activations(frame_0)
            frame_1 = self.dequantize_activations(frame_1)

            # Create a 2D tensor for the time step
            time_step = torch.tensor([[0.5]], device=DEVICE)

            if self.use_gradient_checkpointing:
                result = checkpoint.checkpoint(self.model, frame_0, frame_1, time_step)
            else:
                result = self.model(frame_0, frame_1, time_step)

            # Dequantize result if necessary
            result = self.dequantize_activations(result)
            output_frames.extend([frame.detach().cpu() for frame in result[:-1]])

            number_of_frames_processed_since_last_cleared_cuda_cache += 1
            if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                print("Comfy-VFI: Clearing cache...", end=' ')
                torch.cuda.empty_cache()
                number_of_frames_processed_since_last_cleared_cuda_cache = 0
                print("Done cache clearing")

        output_frames.append(frames[-1:])
        out = torch.cat(output_frames, dim=0)
        print("Comfy-VFI: Final clearing cache...", end=' ')
        torch.cuda.empty_cache()
        print("Done cache clearing")
        return (postprocess_frames(out), )
