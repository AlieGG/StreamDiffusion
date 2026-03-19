import sys
import os
import gc
import logging

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math
import io

DEFAULT_MODEL = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

AVAILABLE_MODELS = [
    "stabilityai/sd-turbo",
    "bakebrain/bergraffi-berlin-graffiti",
    "nitrosocke/mo-di-diffusion",
    "prompthero/openjourney",
    "dreamlike-art/dreamlike-diffusion-1.0",
]

default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD-Turbo</a
    > with a MJPEG stream server.
</p>
"""


RESOLUTION_OPTIONS = {
    "256x256": (256, 256),
    "384x384": (384, 384),
    "512x512": (512, 512),
    "512x288": (512, 288),
    "768x512": (768, 512),
}


def _parse_resolution(resolution: str) -> tuple:
    return RESOLUTION_OPTIONS.get(resolution, (256, 256))


def _build_t_index_list(noise_level: int, num_steps: int) -> list:
    """Distribute num_steps evenly from noise_level//2 up to noise_level."""
    start = noise_level // 2
    if num_steps == 1:
        return [noise_level]
    step = (noise_level - start) / (num_steps - 1)
    return [max(1, int(start + i * step)) for i in range(num_steps)]


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
            description="Describe the visual style or scene you want the AI to generate.",
        )
        negative_prompt: str = Field(
            default_negative_prompt,
            title="Negative Prompt",
            field="textarea",
            id="negative_prompt",
            description="Describe what you DON'T want in the output. E.g. 'blurry, ugly, distorted, cartoon'. This steers the AI away from these qualities.",
        )
        model_id: str = Field(
            DEFAULT_MODEL,
            title="Model",
            field="select",
            id="model_id",
            values=AVAILABLE_MODELS,
            description="The Stable Diffusion model used for generation. Switching will reload the pipeline. Pre-download models in the Models tab for instant switching.",
        )
        noise_level: int = Field(
            47,
            min=30,
            max=49,
            step=1,
            title="Trippy Level",
            field="range",
            id="noise_level",
            description="Controls which denoising timesteps are used. Higher = more noise injected into each frame = more abstract and detached from camera. Lower = more photo-realistic. Requires pipeline reload.",
        )
        delta: float = Field(
            1.0,
            min=0.0,
            max=2.0,
            step=0.05,
            title="Camera Deviation",
            field="range",
            id="delta",
            description="Multiplies the virtual residual noise in the denoising loop. Higher values push the output further from your camera feed for more surreal results. 0 = very faithful to camera, 2.0 = highly abstract. Takes effect immediately.",
        )
        num_steps: int = Field(
            4,
            min=2,
            max=6,
            step=1,
            title="Steps",
            field="range",
            id="num_steps",
            description="Number of denoising steps per frame. More steps = higher quality but slower. 2 is fastest, 6 is most refined. Requires pipeline reload.",
        )
        cfg_type: str = Field(
            "self",
            title="Guidance Type",
            field="select",
            id="cfg_type",
            values=["none", "self"],
            description="Prompt guidance mode. 'self' reuses internal activations to follow your prompt more strongly with minimal extra cost. 'none' is fastest but less prompt-accurate. Requires pipeline reload.",
        )
        guidance_scale: float = Field(
            1.2,
            min=1.0,
            max=3.0,
            step=0.1,
            title="Guidance Scale",
            field="range",
            id="guidance_scale",
            description="How strongly the prompt steers the output. Higher values follow the prompt more closely but can look over-saturated. Only active when Guidance Type is 'self'.",
        )
        do_add_noise: bool = Field(
            True,
            title="Noise Injection",
            field="checkbox",
            id="do_add_noise",
            description="Whether to inject fresh random noise into each camera frame before denoising. Enabled = more varied, creative output per frame. Disabled = smoother, more temporally stable output closer to the camera feed. Requires pipeline reload.",
        )
        resolution: str = Field(
            "256x256",
            title="Resolution",
            field="select",
            id="resolution",
            values=list(RESOLUTION_OPTIONS.keys()),
            description="Output resolution (width × height). Higher resolution = better detail but significantly slower. 256×256 is best for real-time. 512×512 matches the model's native training size. Requires pipeline reload.",
        )
        width: int = Field(
            256, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            256, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        self.args = args
        self.device = device
        self.torch_dtype = torch_dtype
        self.is_ready = False

        params = self.InputParams()
        self.current_model_id = params.model_id
        self.current_noise_level = params.noise_level
        self.current_num_steps = params.num_steps
        self.current_cfg_type = params.cfg_type
        self.current_guidance_scale = params.guidance_scale
        self.current_delta = params.delta
        self.current_negative_prompt = params.negative_prompt
        self.current_do_add_noise = params.do_add_noise
        self.current_resolution = params.resolution

        w, h = _parse_resolution(params.resolution)
        self.stream = self._build_stream(
            w, h,
            params.noise_level, params.num_steps, params.cfg_type,
            params.do_add_noise, params.model_id,
        )
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt=params.negative_prompt,
            num_inference_steps=50,
            guidance_scale=params.guidance_scale,
            delta=params.delta,
        )
        self.last_frame: bytes | None = None
        self.is_ready = True

    def _build_stream(
        self,
        width: int,
        height: int,
        noise_level: int,
        num_steps: int,
        cfg_type: str,
        do_add_noise: bool,
        model_id: str = DEFAULT_MODEL,
    ):
        t_index_list = _build_t_index_list(noise_level, num_steps)
        return StreamDiffusionWrapper(
            model_id_or_path=model_id,
            use_tiny_vae=self.args.taesd,
            device=self.device,
            dtype=self.torch_dtype,
            t_index_list=t_index_list,
            frame_buffer_size=1,
            width=width,
            height=height,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration=self.args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type=cfg_type,
            do_add_noise=do_add_noise,
            use_safety_checker=self.args.safety_checker,
            engine_dir=self.args.engine_dir,
        )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        reinit_needed = (
            params.model_id != self.current_model_id
            or params.noise_level != self.current_noise_level
            or params.num_steps != self.current_num_steps
            or params.cfg_type != self.current_cfg_type
            or params.do_add_noise != self.current_do_add_noise
            or params.resolution != self.current_resolution
        )
        prepare_needed = (
            reinit_needed
            or params.guidance_scale != self.current_guidance_scale
            or params.delta != self.current_delta
            or params.negative_prompt != self.current_negative_prompt
        )

        if reinit_needed:
            self.is_ready = False
            prev_model_id = self.current_model_id
            prev_noise_level = self.current_noise_level
            prev_num_steps = self.current_num_steps
            prev_cfg_type = self.current_cfg_type
            prev_do_add_noise = self.current_do_add_noise
            prev_resolution = self.current_resolution
            self.current_model_id = params.model_id
            self.current_noise_level = params.noise_level
            self.current_num_steps = params.num_steps
            self.current_cfg_type = params.cfg_type
            self.current_do_add_noise = params.do_add_noise
            self.current_resolution = params.resolution
            # Free old stream memory before loading new model
            old_stream = self.stream
            self.stream = None
            del old_stream
            gc.collect()
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            w, h = _parse_resolution(params.resolution)
            try:
                self.stream = self._build_stream(
                    w, h,
                    params.noise_level, params.num_steps, params.cfg_type,
                    params.do_add_noise, params.model_id,
                )
            except Exception as e:
                logging.error(f"[Pipeline] _build_stream failed for {params.model_id!r}: {e}", exc_info=True)
                self.current_model_id = prev_model_id
                self.current_noise_level = prev_noise_level
                self.current_num_steps = prev_num_steps
                self.current_cfg_type = prev_cfg_type
                self.current_do_add_noise = prev_do_add_noise
                self.current_resolution = prev_resolution
                prev_w, prev_h = _parse_resolution(prev_resolution)
                try:
                    self.stream = self._build_stream(
                        prev_w, prev_h,
                        prev_noise_level, prev_num_steps, prev_cfg_type,
                        prev_do_add_noise, prev_model_id,
                    )
                except Exception:
                    pass
                self.is_ready = True
                return None

        if prepare_needed:
            try:
                self.current_guidance_scale = params.guidance_scale
                self.current_delta = params.delta
                self.current_negative_prompt = params.negative_prompt
                self.stream.prepare(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=params.guidance_scale,
                    delta=params.delta,
                )
                self.is_ready = True
            except Exception as e:
                logging.error(f"[Pipeline] stream.prepare failed: {e}", exc_info=True)
                self.is_ready = True
                return None

        image_tensor = self.stream.preprocess_image(params.image)
        output_image = self.stream(image=image_tensor, prompt=params.prompt)

        frame_data = io.BytesIO()
        output_image.save(frame_data, format="JPEG")
        self.last_frame = frame_data.getvalue()

        return output_image
