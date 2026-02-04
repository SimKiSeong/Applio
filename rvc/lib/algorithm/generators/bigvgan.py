"""
BigVGAN Generator for Applio
Adapter for NVIDIA's BigVGAN vocoder to work with Applio's RVC architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from bigvgan import BigVGAN as BigVGANBase
    BIGVGAN_AVAILABLE = True
except ImportError:
    BIGVGAN_AVAILABLE = False
    print("Warning: BigVGAN not installed. Install with: pip install bigvgan")


class BigVGANGenerator(nn.Module):
    """
    BigVGAN Generator adapter for Applio's RVC pipeline.

    This adapter converts Applio's latent space output (z) from the normalizing flow
    to mel-spectrogram-like features that BigVGAN can process.

    Args:
        initial_channel (int): Number of input channels from the flow encoder
        sample_rate (int): Audio sample rate (should be 44000 for BigVGAN 44kHz model)
        upsample_initial_channel (int): Not used in BigVGAN but kept for compatibility
        gin_channels (int): Global conditioning channels (speaker embedding)
        pretrained_model (str): HuggingFace model name for BigVGAN
        use_cuda_kernel (bool): Use CUDA optimized kernels (inference only)
        **kwargs: Additional arguments (ignored for compatibility)
    """

    def __init__(
        self,
        initial_channel: int = 192,
        sample_rate: int = 44000,
        upsample_initial_channel: int = 512,
        gin_channels: int = 256,
        pretrained_model: str = "nvidia/bigvgan_v2_44khz_128band_512x",
        use_cuda_kernel: bool = False,
        **kwargs
    ):
        super().__init__()

        if not BIGVGAN_AVAILABLE:
            raise ImportError(
                "BigVGAN is not installed. Please install it with:\n"
                "pip install bigvgan"
            )

        self.sample_rate = sample_rate
        self.gin_channels = gin_channels

        # Load pretrained BigVGAN model from HuggingFace
        print(f"Loading BigVGAN model: {pretrained_model}")
        self.bigvgan = BigVGANBase.from_pretrained(
            pretrained_model,
            use_cuda_kernel=use_cuda_kernel
        )

        # Get mel config from BigVGAN
        self.h = self.bigvgan.h
        self.num_mels = self.h.num_mels

        # Adapter layer: Convert flow output (z) to mel-like features
        # The flow outputs 'initial_channel' features, we need to convert to num_mels
        self.z_to_mel = nn.Conv1d(
            initial_channel,
            self.num_mels,
            kernel_size=1,
            bias=True
        )

        # Global conditioning (speaker embedding) adapter
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, self.num_mels, kernel_size=1)

        # Initialize adapter layers
        nn.init.xavier_uniform_(self.z_to_mel.weight)
        if self.z_to_mel.bias is not None:
            nn.init.zeros_(self.z_to_mel.bias)

        if gin_channels > 0:
            nn.init.xavier_uniform_(self.cond.weight)
            if self.cond.bias is not None:
                nn.init.zeros_(self.cond.bias)

    def forward(
        self,
        z: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through BigVGAN generator.

        Args:
            z (Tensor): Latent features from normalizing flow [B, C, T]
            f0 (Tensor, optional): F0/pitch information (not used by BigVGAN)
            g (Tensor, optional): Global conditioning (speaker embedding) [B, gin_channels, 1]

        Returns:
            Tensor: Generated audio waveform [B, 1, T*hop_length]
        """
        # Convert latent z to mel-like features
        mel_like = self.z_to_mel(z)

        # Add global conditioning if provided
        if g is not None and self.gin_channels > 0:
            g_processed = self.cond(g)
            mel_like = mel_like + g_processed

        # Generate audio with BigVGAN
        audio = self.bigvgan(mel_like)

        return audio

    def remove_weight_norm(self):
        """
        Remove weight normalization from BigVGAN layers for inference.
        This should be called before inference to speed up generation.
        """
        if hasattr(self.bigvgan, 'remove_weight_norm'):
            self.bigvgan.remove_weight_norm()

    def eval(self):
        """
        Set model to evaluation mode and optionally remove weight norm.
        """
        super().eval()
        # Note: remove_weight_norm should be called explicitly by user
        return self

    def inference(self, z: torch.Tensor, g: Optional[torch.Tensor] = None):
        """
        Inference-optimized forward pass.

        Args:
            z (Tensor): Latent features [B, C, T]
            g (Tensor, optional): Global conditioning [B, gin_channels, 1]

        Returns:
            Tensor: Generated audio [B, 1, T*hop_length]
        """
        with torch.no_grad():
            return self.forward(z, f0=None, g=g)


class BigVGANGeneratorSimple(nn.Module):
    """
    Simplified BigVGAN adapter without latent-to-mel conversion.
    Use this if you want to pass mel-spectrograms directly.
    """

    def __init__(
        self,
        sample_rate: int = 44000,
        pretrained_model: str = "nvidia/bigvgan_v2_44khz_128band_512x",
        use_cuda_kernel: bool = False,
        gin_channels: int = 0,
        **kwargs
    ):
        super().__init__()

        if not BIGVGAN_AVAILABLE:
            raise ImportError("BigVGAN is not installed.")

        self.bigvgan = BigVGANBase.from_pretrained(
            pretrained_model,
            use_cuda_kernel=use_cuda_kernel
        )

        self.gin_channels = gin_channels
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, self.bigvgan.h.num_mels, 1)

    def forward(self, mel: torch.Tensor, f0=None, g=None):
        """Direct mel-spectrogram to audio generation."""
        if g is not None and self.gin_channels > 0:
            mel = mel + self.cond(g)
        return self.bigvgan(mel)

    def remove_weight_norm(self):
        if hasattr(self.bigvgan, 'remove_weight_norm'):
            self.bigvgan.remove_weight_norm()
