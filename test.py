#!/usr/bin/env python3
"""
This file contains encoder module implementation for Whisper Large V3.
This file with the gpu implmentation was made with help of Kimi2-5, using openrouter. 
Author: Martin Vaculik (xvaculm00@stud.fit.vutbr.cz)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)
from transformers.models.whisper.modeling_whisper import WhisperModel
import sys 
from encoder import WhisperEncoderLarge
from transformers.models.whisper.feature_extraction_whisper import (
        WhisperFeatureExtractor,
    )


def test_mel_dimensions():
    """Test that our GPU implementation produces correct mel dimensions."""
    print("=" * 60)
    print("Testing Mel-Spectrogram Dimensions")
    print("=" * 60)

    ### Expected dimensions
    EXPECTED_TIME_FRAMES = 3000
    EXPECTED_MEL_BINS = 80
    SAMPLE_RATE = 16000
    MAX_SECONDS = 30
    MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS

    ### Test with different audio lengths
    test_lengths = [
        16000,  # 1 second
        80000,  # 5 seconds
        160000,  # 10 seconds
        480000,  # 30 seconds (full)
    ]


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder = WhisperEncoderLarge(device=device, dtype=torch.bfloat16)
    encoder.eval()

    print(f"\nExpected: ({EXPECTED_MEL_BINS}, {EXPECTED_TIME_FRAMES})")

    for length in test_lengths:
        ### Create test audio
        audio = torch.randn(1, length, device=device)

        ### Compute mel spectrogram
        with torch.no_grad():
            mel = encoder._compute_log_mel_spectrogram(audio)

        print(
            f"Audio length: {length:>7} samples ({length / SAMPLE_RATE:>5.1f}s) -> "
            f"Mel shape: {mel.shape}"
        )

        if length == MAX_SAMPLES:
            if mel.shape[-1] != EXPECTED_TIME_FRAMES:
                print(
                    f"  ERROR: Expected {EXPECTED_TIME_FRAMES} time frames, got {mel.shape[-1]}"
                )
            else:
                print(f"  ✓ Correct dimensions!")

    print()


def compare_with_original():
    """Compare GPU implementation with original Whisper implementation."""
    print("=" * 60)
    print("Comparing GPU vs Original Implementation")
    print("=" * 60)

    MODEL_NAME = "openai/whisper-large-v3"
    SAMPLE_RATE = 16000
    MAX_SAMPLES = 30 * SAMPLE_RATE

    ### Load original extractor
    print("Loading original Whisper feature extractor...")
    extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)

    ### Load GPU encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder = WhisperEncoderLarge(device=device, dtype=torch.bfloat16)
    encoder.eval()

    ### Create test audio
    print("\nGenerating test audio...")
    np.random.seed(42)
    test_audio_np = np.random.randn(MAX_SAMPLES).astype(np.float32) * 0.1
    test_audio_torch = torch.from_numpy(test_audio_np).to(device)

    ### Original implementation
    print("\nProcessing with original implementation (CPU)...")
    start = time.time()
    processed = extractor(
        test_audio_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    original_mel = processed.input_features
    original_time = time.time() - start
    print(f"Original time: {original_time:.4f}s")
    print(f"Original mel shape: {original_mel.shape}")

    ### GPU implementation
    print("\nProcessing with GPU implementation...")
    start = time.time()
    with torch.no_grad():
        gpu_mel = encoder._compute_log_mel_spectrogram(test_audio_torch.unsqueeze(0))
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"GPU mel shape: {gpu_mel.shape}")

    print(f"\nSpeedup: {original_time / gpu_time:.2f}x")
    original_mel = original_mel.to(device)

    print("\nNumerical Comparison:")
    print(f"Original range: [{original_mel.min():.4f}, {original_mel.max():.4f}]")
    print(f"GPU range:      [{gpu_mel.min():.4f}, {gpu_mel.max():.4f}]")

    diff = torch.abs(original_mel - gpu_mel).max()
    mean_diff = torch.abs(original_mel - gpu_mel).mean()
    print(f"Max absolute difference: {diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    if diff < 0.001:
        print("✓ Results match closely!")
    elif diff < 0.001:
        print("Results are close")
    else:
        print("✗ Results differ")

    ### Check dimensions

    print()


def test_encoder_forward():
    """Test full encoder forward pass."""
    print("=" * 60)
    print("Testing Full Encoder Forward Pass")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder = WhisperEncoderLarge(device=device, dtype=torch.bfloat16)
    encoder.eval()

    # Test with different batch sizes and lengths
    test_cases = [
        (1, 16000),  # 1 second
        (2, 80000),  # 5 seconds, batch of 2
        (4, 480000),  # 30 seconds, batch of 4
    ]

    for batch_size, length in test_cases:
        print(f"\nTest case: batch={batch_size}, length={length}")

        ### Create test audio
        audio = torch.randn(batch_size, length, device=device)

        ### Forward pass
        try:
            with torch.no_grad():
                hidden_states, attention_mask = encoder(audio)

            print(f"  ✓ Success!")
            print(f"    Hidden states shape: {hidden_states.shape}")
            print(f"    Attention mask shape: {attention_mask.shape}")

            expected_seq_len = 1500  
            if hidden_states.shape[1] == expected_seq_len:
                print(f"    ✓ Sequence length correct: {hidden_states.shape[1]}")
            else:
                print(
                    f"    ⚠ Unexpected sequence length: {hidden_states.shape[1]} (expected {expected_seq_len})"
                )

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print()


def test_whisper_native_extractor():
    # test func 
    print("=" * 60)
    print("Analyzing Native Whisper Extractor")
    print("=" * 60)

    MODEL_NAME = "openai/whisper-large-v3"
    SAMPLE_RATE = 16000

    extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)

    print(f"Feature extractor config:")
    print(f"  sample_rate: {extractor.sampling_rate}")
    print(f"  feature_size: {extractor.feature_size}")
    print(f"  hop_length: {extractor.hop_length}")
    print(f"  n_fft: {extractor.n_fft}")
    print(f"  n_samples: {extractor.n_samples}")
    print(f"  nb_max_frames: {extractor.nb_max_frames}")

    audio_30s = np.random.randn(480000).astype(np.float32) * 0.1
    processed = extractor(
        audio_30s,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    print(f"\n30-second audio:")
    print(f"  Input samples: {len(audio_30s)}")
    print(f"  Output mel shape: {processed.input_features.shape}")
    print(f"  Expected: (1, 80, 3000)")

    audio_5s = np.random.randn(80000).astype(np.float32) * 0.1
    processed_5s = extractor(
        audio_5s,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    print(f"\n5-second audio:")
    print(f"  Input samples: {len(audio_5s)}")
    print(f"  Output mel shape: {processed_5s.input_features.shape}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Whisper Encoder GPU Implementation Tests")
    print("=" * 60 + "\n")

    ### Run tests
    test_whisper_native_extractor()
    test_mel_dimensions()
    compare_with_original()
    test_encoder_forward()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)