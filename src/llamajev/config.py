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
    max_concurrent_requests: int = Field(default=16, gt=0)
    max_concurrent_branches: int | None = Field(
        default=None, description="Concurrent backend calls; defaults to the backend's slot count."
    )
    request_timeout: float = Field(default=120, gt=0)
    startup_timeout: float = Field(default=600, gt=0)
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    top_n: int = Field(default=256, ge=2, description="n_probs requested per branch (readout depth).")

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
    cache_ram_mib: int = Field(default=4096, ge=-1)
