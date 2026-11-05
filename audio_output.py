import soundfile as sf
import torch

OUTPUT_SAMPLE_RATE = 32000
LIMITER_THRESHOLD = 0.9
# Largest positive 16-bit PCM value. tanh saturates to exactly 1.0 in float32 for large
# inputs, so the limiter approaches this ceiling instead of 1.0.
LIMITER_CEILING = 32767 / 32768


def soft_clip(wav, threshold=LIMITER_THRESHOLD):
    # |x| <= threshold: unchanged; above: compressed smoothly toward LIMITER_CEILING (< 1.0).
    # No whole-song gain change, so regenerated sections (#13) keep the original's level.
    over = wav.abs() > threshold
    headroom = LIMITER_CEILING - threshold
    limited = torch.sign(wav) * (threshold + headroom * torch.tanh((wav.abs() - threshold) / headroom))
    return torch.where(over, limited, wav)


def save_output_wav(path, wav, sample_rate=OUTPUT_SAMPLE_RATE):
    """Limit the MusicGen output and save it as 16-bit PCM wav."""
    wav = soft_clip(wav.detach().cpu().float())
    # soundfile instead of torchaudio.save: newer torchaudio (TorchCodec) ignores encoding/bits_per_sample
    sf.write(path, wav.numpy().T, sample_rate, subtype="PCM_16")
