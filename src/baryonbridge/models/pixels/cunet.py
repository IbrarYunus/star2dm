import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple, Optional, Union

from baryonbridge.models.pixels.blocks import get_conv, get_down_block, get_mid_block, get_up_block
from baryonbridge.models.pixels.embeddings import GaussianFourierProjection, Timesteps, TimestepEmbedding

class CUNet(nn.Module):
    """ Models p(x|y, h)

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        in_channels: int =1,
        out_channels: int =1,
        cond_channels: int = 1,
        num_channels: Tuple[int, ...] = (32, 64, 64,),
        layers_per_block: int = 2,
        act_fn: str = 'SiLU',
        num_groups_norm: int = 32,
        norm_eps: float = 1e-5,
        interpolate_up_down: bool = True,
        transpose_up: bool = False,
        # time 
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_cond_proj_dim: Optional[int] = None,
        flip_sin_to_cos: bool = True,
        freq_shift: float= 0.,
        # h conditioning
        num_h_embeds: Optional[int] = None,
        concat_h_and_time: bool = False,
        spatial_theta_conditioning: bool = False,
        spatial_theta_hidden: int = 64,
        dropout_prob: float = 0.1,
        conv_kernel_size: int = 3,
        conv_padding_mode: str = "circular",
        downsample_padding: int = 1,
        dim: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels 
        self.num_channels = num_channels
        self.layers_per_block = layers_per_block
        self.act_fn = act_fn
        self.num_groups_norm = num_groups_norm
        self.norm_eps = norm_eps
        self.interpolate_up_down = interpolate_up_down
        self.transpose_up = transpose_up
        self.time_embedding_type = time_embedding_type
        self.dropout_prob = dropout_prob
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding_mode = conv_padding_mode
        self.spatial_theta_conditioning = spatial_theta_conditioning
        self.spatial_theta_hidden = spatial_theta_hidden
        if spatial_theta_conditioning and num_h_embeds is None:
            raise ValueError(
                "spatial_theta_conditioning requires num_h_embeds to be set "
                "(theta enters the spatial branch alongside the radius channel)."
            )
        # theta channels + one normalized radius channel
        spatial_theta_dim = num_h_embeds + 1 if spatial_theta_conditioning else None
        activation_fn = getattr(nn, act_fn)()

        # input convolution:
        padding = (conv_kernel_size - 1) // 2
        self.conv_in = get_conv(
            self.in_channels + self.cond_channels,
            self.num_channels[0],
            kernel_size=conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
            dim=dim,
        )

        # time embedding:
        time_embed_dim, timestep_input_dim = self.get_time_proj(
            time_embedding_type,
            num_channels=num_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=time_embedding_dim,
        )
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=None,
            cond_proj_dim=time_cond_proj_dim,
        ) 
        self.concat_h_and_time = concat_h_and_time

        blocks_time_embed_dim = time_embed_dim
        # h conditioning embedding:
        if num_h_embeds is not None:
            self.h_embedding = TimestepEmbedding(
                num_h_embeds,
                time_embed_dim,
                act_fn=act_fn,
                post_act_fn=None,
                cond_proj_dim=time_cond_proj_dim,
            ) 
            if self.concat_h_and_time:
                # The time embeddings are concatenated with the class embeddings. The dimension of the
                # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
                # regular time embeddings
                blocks_time_embed_dim *=  2
        # down blocks
        output_channel = num_channels[0]
        self.down_blocks = nn.ModuleList([])
        for i, _ in enumerate(num_channels):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1
            down_block = get_down_block(
                num_resnet_blocks=layers_per_block,
                input_channels=input_channel,
                output_channels=output_channel,
                conditioning_dim=blocks_time_embed_dim,
                dropout_prob=dropout_prob,
                num_groups_norm = num_groups_norm,
                activation_fn = activation_fn,
                kernel_size= conv_kernel_size,
                padding = downsample_padding,
                padding_mode=conv_padding_mode,
                add_downsample=not is_final_block,
                interpolate_down = interpolate_up_down,
                dim=dim,
            )
            self.down_blocks.append(down_block)

        # mid blocks
        self.mid_block = get_mid_block(
                num_resnet_blocks=layers_per_block,
                input_channels=num_channels[-1],
                output_channels=output_channel,
                conditioning_dim=blocks_time_embed_dim,
                dropout_prob=dropout_prob,
                num_groups_norm = num_groups_norm,
                activation_fn = activation_fn,
                kernel_size= conv_kernel_size,
                padding = 1,
                padding_mode=conv_padding_mode,
                spatial_theta_dim=spatial_theta_dim,
                spatial_theta_hidden=spatial_theta_hidden,
                dim=dim,
        )
        # up blocks
        # count how many layers upsample the images
        self.num_upsamplers = 0

        self.up_blocks = nn.ModuleList([])
        reversed_num_channels = list(reversed(num_channels))
        output_channel = reversed_num_channels[0]
        for i, _ in enumerate(num_channels):
            prev_output_channel = output_channel
            output_channel = reversed_num_channels[i]
            input_channel = reversed_num_channels[min(i + 1, len(num_channels) - 1)]
            is_final_block = i == len(num_channels) - 1
            up_block = get_up_block(
                num_resnet_blocks=layers_per_block + 1,
                input_channels=input_channel,
                output_channels=output_channel,
                prev_output_channels=prev_output_channel,
                conditioning_dim=blocks_time_embed_dim,
                dropout_prob=dropout_prob,
                num_groups_norm = num_groups_norm,
                activation_fn = activation_fn,
                kernel_size= conv_kernel_size,
                padding = 1,
                padding_mode=conv_padding_mode,
                add_upsample=not is_final_block,
                interpolate_up = interpolate_up_down,
                transpose_conv=transpose_up,
                spatial_theta_dim=spatial_theta_dim,
                spatial_theta_hidden=spatial_theta_hidden,
                dim=dim,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if num_groups_norm is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=num_channels[0], num_groups=num_groups_norm, eps=norm_eps
            )

            self.conv_act = activation_fn

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_kernel_size - 1) // 2
        self.conv_out = get_conv(
            num_channels[0], 
            out_channels, 
            kernel_size=conv_kernel_size, 
            padding=conv_out_padding, 
            padding_mode=conv_padding_mode,
            dim=dim,
        )

    @property
    def hparams(self,):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "cond_channels": self.cond_channels,
            "num_channels": self.num_channels,
            "layers_per_block": self.layers_per_block,
            "act_fn": self.act_fn,
            "num_groups_norm": self.num_groups_norm,
            "norm_eps": self.norm_eps,
            "interpolate_up_down": self.interpolate_up_down,
            "transpose_up": self.transpose_up,
            "time_embedding_type": self.time_embedding_type,
            "dropout_prob": self.dropout_prob,
            "conv_kernel_size": self.conv_kernel_size,
            "conv_padding_mode": self.conv_padding_mode,
            "spatial_theta_conditioning": self.spatial_theta_conditioning,
            "spatial_theta_hidden": self.spatial_theta_hidden,
        }


    def get_time_proj(
        self,
        time_embedding_type: str,
        num_channels: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or num_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or num_channels[0] * 4

            self.time_proj = Timesteps(num_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = num_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb

    def get_spatial_theta_map(self, sample: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Builds the spatial conditioning map for SPADE-style modulation.

        Broadcasts h over the pixel grid and appends a normalized radius
        channel (r = sqrt(sum coords^2) with coords in [-1, 1] per axis, so the
        FOV keeps a fixed normalized-to-physical mapping). Returned at the
        input resolution, shape (B, num_h_embeds + 1, *spatial); the modulation
        blocks resample it per feature-map resolution.
        """
        spatial_shape = sample.shape[2:]
        coords = [
            torch.linspace(-1.0, 1.0, s, device=sample.device, dtype=sample.dtype)
            for s in spatial_shape
        ]
        grids = torch.meshgrid(*coords, indexing="ij")
        radius = torch.sqrt(sum(g ** 2 for g in grids))
        radius = radius[(None, None)].expand(sample.shape[0], 1, *spatial_shape)
        h_map = h.view(h.shape + (1,) * len(spatial_shape)).expand(*h.shape, *spatial_shape)
        return torch.cat([h_map, radius], dim=1)

    def forward(
        self,
        timestep: Union[torch.Tensor, float, int],
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # 0. concat image conditioning
        if conditioning is not None:
            x = torch.concat(
                (x, conditioning),
                axis=1,
            )
        # 1. time
        t_emb = self.get_time_embed(sample=x, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        # 2. h embedding
        if h is not None:
            h_emb = self.h_embedding(h,)
            if self.concat_h_and_time:
                emb = torch.cat([emb, h_emb], dim=-1)
            else:
                emb = emb + h_emb

        # 2b. spatial theta conditioning map (theta broadcast + radius channel)
        spatial_cond = None
        if self.spatial_theta_conditioning and h is not None:
            spatial_cond = self.get_spatial_theta_map(sample=x, h=h)

        # 2. pre-process x
        x = self.conv_in(x)

        # 3. down
        down_block_res_samples = []
        down_block_res_samples = (x,)
        for i, downsample_block in enumerate(self.down_blocks):
            x, res_samples = downsample_block(hidden_states=x, conditioning=emb)
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            x = self.mid_block(x, conditioning=emb, spatial_conditioning=spatial_cond)
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            x = upsample_block(
                hidden_states=x,
                conditioning=emb,
                res_hidden_states_tuple=res_samples,
                spatial_conditioning=spatial_cond,
            )
        # 6. post-process
        if self.conv_norm_out:
            x = self.conv_norm_out(x)
            x = self.conv_act(x)
        return self.conv_out(x)

