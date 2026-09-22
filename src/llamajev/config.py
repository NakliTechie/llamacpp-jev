from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_QUESTIONS = 64
MAX_ANSWERS = 64


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLAMAJEV_", extra="ignore")

    # Public API
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    model_alias: str = "jev-latest"
    served_model_name: str | None = Field(
        default=None, description="Public model ID; defaults to the GGUF file stem from /props."
    )
    max_body_bytes: int = Field(default=2 * 1024 * 1024, gt=0)
    body_read_timeout: float = Field(
        default=30, gt=0, description="Deadline to receive the whole request body (guards slow uploads)."
    )
    max_concurrent_requests: int = Field(default=16, gt=0)
    request_timeout: float = Field(default=120, gt=0, description="Deadline for one whole evaluation (seconds).")
    startup_timeout: float = Field(default=600, gt=0)
    # Vision resource bounds — a small compressed image can decode to a large allocation.
    max_images: int = Field(default=8, gt=0, description="Maximum image_url parts per request.")
    max_image_bytes: int = Field(
        default=8 * 1024 * 1024, gt=0, description="Maximum decoded bytes of a single image payload."
    )
    max_image_pixels: int = Field(
        default=24_000_000, gt=0, description="Maximum width×height of a single image (decompression-bomb guard)."
    )
    max_total_image_pixels: int = Field(
        default=64_000_000, gt=0, description="Maximum summed width×height across all images in a request."
    )
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    top_n: int = Field(default=256, ge=2, description="n_probs requested per branch (readout depth).")
    pin_slot: bool = Field(
        default=True,
        description=(
            "Send every branch to the slot that ran the warm-up, so the prefix state (including "
            "image tokens) is restored from that slot's checkpoint instead of re-processed by "
            "other slots. Measured 2026-09-21 on a fresh 448x448 image: 0.84 s pinned vs 1.9 s spread."
        ),
    )

    # Backend (llama-server)
    backend_url: str = "http://127.0.0.1:8090"
    backend_port: int = Field(default=8090, ge=1, le=65535)
    llama_server: str = Field(
        default="llama-server", description="Path to the llama-server binary when launching it."
    )
    model: str | None = Field(default=None, description="GGUF path, used only when launching.")
    mmproj: str | None = Field(default=None, description="Multimodal projector GGUF, optional.")
    n_slots: int = Field(default=4, ge=1)
    n_ctx: int = Field(default=8192, ge=512)
    n_gpu_layers: int = Field(default=99, ge=0)
    cache_ram_mib: int = Field(
        default=0,
        ge=-1,
        description=(
            "llama-server --cache-ram. Off by default: with slot pinning the RAM prompt cache adds "
            "nothing, and with it on (4096) two of two repeated-image requests hung inside "
            "llama-server on 2026-09-21 (task launched, never finished, freed only by cancel)."
        ),
    )
