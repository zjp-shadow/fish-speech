import base64
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field, conint, model_validator
from pydantic.functional_validators import SkipValidation
from typing_extensions import Annotated

from fish_speech.content_sequence import TextPart, VQPart


class ServeVQPart(BaseModel):
    type: Literal["vq"] = "vq"
    codes: SkipValidation[list[list[int]]]


class ServeTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ServeAudioPart(BaseModel):
    type: Literal["audio"] = "audio"
    audio: bytes


class ServeRequest(BaseModel):
    # Raw content sequence dict that we can use with ContentSequence(**content)
    content: dict
    max_new_tokens: int = 600
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    streaming: bool = False
    num_samples: int = 1
    early_stop_threshold: float = 1.0


class ServeVQGANEncodeRequest(BaseModel):
    # The audio here should be in wav, mp3, etc
    audios: list[bytes]


class ServeVQGANEncodeResponse(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeRequest(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeResponse(BaseModel):
    # The audio here should be in PCM float16 format
    audios: list[bytes]


class AudioExample:
    """
    A class to represent an audio example with its text transcription.
    This class handles loading audio from a file path and converting it to bytes.
    
    Default values match the client's defaults in api_client.py:
    - audio_path: "/home/zjp/fish-speech/reference/cj_long.MP3"
    - text: "我在观察那个弹幕姬。好了好了，看完了。谢谢潇潇我大号的醒目留言，Thank You。然后今天晚上有个新的皮肤，就新的纸片人给大家看。我昨天还有今天早上测试了一下，还挺可爱的，不是挺可爱哦。超级，就超级可爱，谢谢kita得私的舰长，谢谢无尘的流石流石，大家中午好。"
    """
    
    DEFAULT_AUDIO_PATH = "/home/zjp/fish-speech/reference/cj_long.MP3"
    DEFAULT_TEXT = "我在观察那个弹幕姬。好了好了，看完了。谢谢潇潇我大号的醒目留言，Thank You。然后今天晚上有个新的皮肤，就新的纸片人给大家看。我昨天还有今天早上测试了一下，还挺可爱的，不是挺可爱哦。超级，就超级可爱，谢谢kita得私的舰长，谢谢无尘的流石流石，大家中午好。"
    
    @classmethod
    def load_from_path(cls, audio_path=DEFAULT_AUDIO_PATH, text=DEFAULT_TEXT):
        """
        Load audio from a file path and create a ServeReferenceAudio instance.
        
        Args:
            audio_path: Path to the audio file
            text: Text transcription of the audio
            
        Returns:
            ServeReferenceAudio instance
        """
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            return ServeReferenceAudio(audio=audio_bytes, text=text)
        except Exception as e:
            raise ValueError(f"Failed to load audio from {audio_path}: {e}")


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    @model_validator(mode="before")
    def decode_audio(cls, values):
        audio = values.get("audio")
        if (
            isinstance(audio, str) and len(audio) > 255
        ):  # Check if audio is a string (Base64)
            try:
                values["audio"] = base64.b64decode(audio)
            except Exception as e:
                # If the audio is not a valid base64 string, we will just ignore it and let the server handle it
                pass
        return values

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "wav"
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # not usually used below
    streaming: bool = False
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.1
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8

    class Config:
        # Allow arbitrary types for pytorch related types
        arbitrary_types_allowed = True
