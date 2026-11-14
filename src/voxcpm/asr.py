"""
Lightweight ASR helper used by the demo / WebUI to auto-fill reference transcripts.

Uses a Whisper model loaded from the Hugging Face Hub via ``transformers``.
The model id can be overridden with the ``VOXCPM_ASR_MODEL`` environment variable
(either a Hub repo id or a local directory).
"""

import os
from typing import Optional

import numpy as np

DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"
_SAMPLE_RATE = 16000


class ASRModel:
    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        import torch
        from transformers import pipeline

        self.model_id = model_id or os.environ.get("VOXCPM_ASR_MODEL", DEFAULT_ASR_MODEL)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            torch_dtype=dtype,
            device=device,
        )

    def transcribe(self, audio_path: str) -> str:
        import librosa

        wav, _ = librosa.load(audio_path, sr=_SAMPLE_RATE, mono=True)
        result = self._pipe(
            {"raw": wav.astype(np.float32), "sampling_rate": _SAMPLE_RATE},
            # Whisper needs timestamp mode for native long-form (>30 s) decoding.
            return_timestamps=len(wav) > 30 * _SAMPLE_RATE,
            generate_kwargs={"task": "transcribe"},
        )
        return result["text"].strip()
