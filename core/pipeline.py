"""ProcurementQA Agent pipeline: configuration and wiring of the model, embeddings and agent workflow.

Retrieval runs through core/retrieval/retriever.py against the tri-store (USE_TRI_STORE=true)
or the in-memory store built from outputs/corpus (local PoC); the pipeline itself holds no corpus.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.models import BaseModel
from core.utils.settings import env


def _tri_store_enabled() -> bool:
    return os.getenv("USE_TRI_STORE", "false").lower() in ("true", "1", "yes")


@dataclass
class ModelConfig:
    """LLM configuration (OpenAI-compatible endpoint, OpenRouter by default)."""
    model_name: str = "openrouter"
    device: str = "cuda:0"
    prompt_language: str = "th"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_length: int = 4096
    temperature: float = 0.1

    def __post_init__(self):
        # "openrouter" (model taken from LLM_MODEL), "provider/model" or "openrouter:provider/model"
        if (
            self.model_name != "openrouter"
            and "/" not in self.model_name
            and not self.model_name.startswith("openrouter:")
        ):
            raise ValueError(
                f"Invalid model_name: {self.model_name}. "
                "Use 'openrouter' or an OpenRouter model string (e.g. 'qwen/qwen3.5-9b')"
            )
        if self.prompt_language.lower() not in ("th", "thai", "default"):
            raise ValueError(f"Invalid prompt_language: {self.prompt_language}. Must be one of th, thai")


@dataclass
class DataConfig:
    """Dataset and output locations (the corpus itself lives in the stores / outputs/corpus)."""
    datasets_path: Optional[str] = None
    output_dir: str = "./outputs"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class EmbeddingConfig:
    provider: str = "local"  # "local" or "tokenmind"
    api_url: str = "http://localhost:11434/api/embed"
    model: str = "BAAI/bge-m3"
    tokenmind_api_key: Optional[str] = None
    tokenmind_base_url: Optional[str] = None
    tokenmind_model: str = "BAAI/bge-m3"


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    agentic_max_retries: int = 2

    @classmethod
    def from_env_file(cls, dotenv_path: Optional[str] = None) -> "PipelineConfig":
        from dotenv import load_dotenv
        # Keep variables already set (e.g. Docker compose networking overrides)
        load_dotenv(dotenv_path=dotenv_path or ".env", override=False)

        model_name = env("MODEL_NAME") or os.getenv("LLM_MODEL") or "openrouter"
        base_url = env("LLM_BASE_URL") or (
            "https://openrouter.ai/api/v1"
            if (env("OPENROUTER_API_KEY") or model_name == "openrouter" or "/" in model_name) else None
        )
        return cls(
            model=ModelConfig(
                model_name=model_name,
                device=env("DEVICE", "cpu"),
                prompt_language=env("PROMPT_LANGUAGE", "th"),
                api_key=env("OPENROUTER_API_KEY"),
                base_url=base_url,
                max_length=int(env("LLM_MAX_TOKENS", 4096)),
                temperature=float(env("LLM_TEMPERATURE", 0.1)),
            ),
            data=DataConfig(
                datasets_path=env("DATASETS_PATH", "./datasets"),
                output_dir=env("OUTPUT_DIR", "./outputs"),
            ),
            embedding=EmbeddingConfig(
                provider=env("EMBEDDING_PROVIDER", "local").lower(),
                api_url=env("EMBEDDING_API_URL", "http://localhost:11434/api/embed"),
                model=env("EMBEDDING_MODEL", "BAAI/bge-m3"),
                tokenmind_api_key=env("TOKENMIND_API_KEY"),
                tokenmind_base_url=env("TOKENMIND_BASE_URL", "https://tokenmind.abdul.in.th/v1"),
                tokenmind_model=env("TOKENMIND_EMBEDDING_MODEL", "BAAI/bge-m3"),
            ),
            agentic_max_retries=int(os.getenv("AGENTIC_MAX_RETRIES", 2)),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            model=ModelConfig(**d.get("model", {})),
            data=DataConfig(**d.get("data", {})),
            embedding=EmbeddingConfig(**d.get("embedding", {})),
            agentic_max_retries=int(d.get("agentic_max_retries", 2)),
        )

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "embedding": asdict(self.embedding),
            "agentic_max_retries": self.agentic_max_retries,
        }

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PipelineConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class ProcurementQAPipeline:
    """Model + embeddings + agent workflow. Retrieval state lives in core.retrieval.retriever."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        from core.prompts import set_prompt_language
        from core.retrieval.embedding import configure_embedding

        set_prompt_language(self.config.model.prompt_language)
        e = self.config.embedding
        configure_embedding(api_url=e.api_url, model=e.model, provider=e.provider,
                            tokenmind_api_key=e.tokenmind_api_key, tokenmind_base_url=e.tokenmind_base_url,
                            tokenmind_model=e.tokenmind_model)
        self.model = self._init_model()

        if _tri_store_enabled():
            from core.database import StorageManager
            self.storage = StorageManager.get_instance()
            self.storage.init_all_stores()
            print("Tri-Store database active (PostgreSQL + Qdrant + Neo4j).")
        else:
            print("Local mode: retrieval uses the in-memory store built from outputs/corpus (data_ocr/).")

    def _init_model(self) -> BaseModel:
        """Initialize the OpenAI-compatible chat model (OpenRouter by default)."""
        from core.models import OpenRouterChatbot

        actual_model = self.config.model.model_name
        if actual_model == "openrouter":
            actual_model = os.getenv("LLM_MODEL", "google/gemma-3-4b-it")
        elif actual_model.startswith("openrouter:"):
            actual_model = actual_model.split(":", 1)[1]
        return OpenRouterChatbot(
            model_name=actual_model,
            device=self.config.model.device,
            api_key=self.config.model.api_key,
            base_url=self.config.model.base_url,
        )

    def analyze_case(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run the agent workflow on one question ({'question'|'fact', 'id', ...})."""
        from core.agent import ProcurementAgenticWorkflow
        workflow = ProcurementAgenticWorkflow(self.model, max_retries=self.config.agentic_max_retries)
        return [workflow.invoke(case)]

    def analyze_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"case_id": c.get("id"), "question": c.get("question") or c.get("fact"), "analysis": self.analyze_case(c)}
                for c in cases]
