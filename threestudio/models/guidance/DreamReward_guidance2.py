import sys
from dataclasses import dataclass, field
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mvdream.camera_utils import convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
import Reward3D as r3d
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def _transform():
    return Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
def calculate_weight(a, b, n=0):
    diff = len(str(a).split(".")[0])-len(str(b).split(".")[0])
    weight = 10**(diff - n)
    return weight
@threestudio.register("DreamReward-guidance2")
class MultiviewDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model_name: str = (
            "sd-v2.1-base-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        )
        reward_model_path: str = (
            ""
        )
        resume_num: int = 0
        alg_type: str = "Reward3D_Scorer"#in[Reward3D_Scorer,Reward3D_CrossViewFusion]
        ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")

        self.model = build_model(self.cfg.model_name, ckpt_path=self.cfg.ckpt_path)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        self.to(self.device)
        self.alg_type = self.cfg.alg_type
        threestudio.info(f"Loaded Multiview Diffusion!")
        med_config_path = "scripts/med_config.json"
        state_dict_path = self.cfg.reward_model_path
        state_dict      = torch.load(state_dict_path)
        if self.alg_type == "Reward3D_Scorer":
            Reward3D        = r3d.Reward3D_(device=self.device, med_config=med_config_path)
        elif self.alg_type == "Reward3D_CrossViewFusion":
            Reward3D        = r3d.Reward3D(device=self.device, med_config=med_config_path)
        msg = Reward3D.load_state_dict(state_dict,strict=False)
        print(msg)
        print(self.cfg.reward_model_path)
        self.Reward3D=Reward3D.to(self.device)
        threestudio.info(f"Loaded Reward3D!")

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32] Latent space image

    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 256 256"]:
        input_dtype = latents.dtype
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        camera = c2w
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = (
                    F.interpolate(
                        rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
                    )
                    * 2
                    - 1
                )
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=latents.device,
            )
            t_ = t.item()
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings,
                    "camera": camera,
                    "num_frames": self.cfg.n_view,
                }
            else:
                context = {"context": text_embeddings}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(
            2
        )  # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(
                latents_noisy, t, noise_pred
            )

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(
                    latents_noisy, t, noise_pred_text
                )
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(
                    -1, self.cfg.n_view, *latents_recon_nocfg.shape[1:]
                )
                latents_recon_reshape = latents_recon.view(
                    -1, self.cfg.n_view, *latents_recon.shape[1:]
                )
                factor = (
                    latents_recon_nocfg_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8
                ) / (latents_recon_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8)

                latents_recon_adjust = latents_recon.clone() * factor.squeeze(
                    1
                ).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = (
                    self.cfg.recon_std_rescale * latents_recon_adjust
                    + (1 - self.cfg.recon_std_rescale) * latents_recon
                )

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = (
                0.5
                * F.mse_loss(latents, latents_recon.detach(), reduction="sum")
                / latents.shape[0]
            )
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # Original SDS
            # w(t), sigma_t^2
            w = 1 - self.alphas_cumprod[t]
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        # Get prompt_tokens
        if not hasattr(self, 'rm_input_ids'):
            self.rm_input_ids = []
            self.rm_attention_mask = []
            prompts_vds = prompt_utils.prompts_vd
            for idx in range(4):
                prompts_vd = prompts_vds[idx]
                g = self.Reward3D.blip.tokenizer(
                    prompts_vd, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=100, 
                    return_tensors="pt"
                )
                self.rm_input_ids.append(g.input_ids)
                self.rm_attention_mask.append(g.attention_mask)
                self.global_step = 0 + self.cfg.resume_num
        else:
            self.global_step += 1

        adding_reward = self.global_step > 1000
        if adding_reward:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in prompt_utils.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = prompt_utils.direction2idx[d.name]
            rm_input_ids      = torch.cat([self.rm_input_ids[idx]      for idx in direction_idx]).to(self.device)
            rm_attention_mask = torch.cat([self.rm_attention_mask[idx] for idx in direction_idx]).to(self.device)
            if t_ <=300 and self.global_step<=9800:
                with torch.no_grad():
                    image = self.decode_latents(latents_recon.detach())
                image = pred_rgb - (pred_rgb - image).detach() 
                image = _transform()(image)
                rewards               = self.Reward3D(image,rm_input_ids, rm_attention_mask)
            else:
                image_render  = _transform()(pred_rgb)
                rewards       = self.Reward3D(image_render,rm_input_ids, rm_attention_mask)         
            loss_reward       = F.relu(-rewards+4).mean()
            weight = calculate_weight(loss.item(),loss_reward.item())
            loss +=  loss_reward*weight*0.3
            if self.global_step>9800:
                loss = loss_reward*1000000
        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
