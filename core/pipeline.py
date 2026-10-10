"""ProcurementQA Agent main class"""
import os
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from tqdm import tqdm

from core.models import BaseModel
from core.graph.local_graph import GraphDBManager
from core.utils.settings import env


def _tri_store_enabled() -> bool:
    return os.getenv("USE_TRI_STORE", "false").lower() in ("true", "1", "yes")


def _ensure_local_corpus(path: str) -> None:
    """The corpus JSON is not committed: build it from the OCR directory (OCR_DIR, default
    data_ocr/) when it is missing or older than the OCR files, so a fresh clone needs only data_ocr/."""
    from core.chunking.corpus import ensure_corpus
    ocr_dir = os.getenv("OCR_DIR", "data_ocr")
    if not os.path.isdir(ocr_dir) and os.path.exists(path):
        return
    try:
        summary = ensure_corpus(ocr_dir, os.path.dirname(path) or ".")
        if summary:
            print(f"Built corpus from {ocr_dir}: {summary['chunks']} chunks -> {os.path.dirname(path)}")
    except FileNotFoundError:
        pass  # reported by the caller with the expected path


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "openrouter"
    device: str = "cuda:0"
    prompt_language: str = "th"
    # OpenAI-type model configuration
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Generation parameters
    max_length: int = 4096
    temperature: float = 0.1
    
    def __post_init__(self):
        """Validate model name"""
        # Every model is served through an OpenAI-compatible endpoint (OpenRouter by default):
        # "openrouter" (model taken from LLM_MODEL), "provider/model" or "openrouter:provider/model".
        if (
            self.model_name != "openrouter"
            and "/" not in self.model_name
            and not self.model_name.startswith("openrouter:")
        ):
            raise ValueError(
                f"Invalid model_name: {self.model_name}. "
                "Use 'openrouter' or an OpenRouter model string (e.g. 'qwen/qwen3.5-9b')"
            )
        valid_prompt_languages = ["en", "zh", "cn", "chinese", "english", "th", "thai", "default"]
        if self.prompt_language.lower() not in valid_prompt_languages:
            raise ValueError(
                f"Invalid prompt_language: {self.prompt_language}. "
                f"Must be one of {valid_prompt_languages}"
            )


@dataclass
class DataConfig:
    """Data path configuration"""
    case_db_path: str = "./outputs/corpus/cases_with_feature.json"
    law_to_crime_path: str = "./outputs/corpus/law_to_crime.json"
    datasets_path: Optional[str] = None  # Dataset root directory
    output_dir: str = "./outputs"
    
    def __post_init__(self):
        """Create output directory"""
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class RetrieveConfig:
    """Retrieval configuration"""
    top_retrieve: bool = True
    direct_retrieve: bool = True
    augment_retrieve: bool = True
    top_retrieve_top_k: int = 3
    direct_retrieve_top_k: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "top_retrieve": self.top_retrieve,
            "direct_retrieve": self.direct_retrieve,
            "augment_retrieve": self.augment_retrieve,
            "top_retrieve_top_k": self.top_retrieve_top_k,
            "direct_retrieve_top_k": self.direct_retrieve_top_k
        }


@dataclass
class GraphConfig:
    """Graph database configuration"""
    graph_db_path: Optional[str] = None  # Graph database save/load path
    embedding_provider: str = "local"  # "local" or "tokenmind"
    embedding_api_url: str = "http://localhost:11434/api/embed"
    embedding_model: str = "unsloth/embeddinggemma-300m"
    tokenmind_api_key: Optional[str] = None
    tokenmind_base_url: Optional[str] = None
    tokenmind_embedding_model: str = "BAAI/bge-m3"
    auto_save: bool = True  # Whether to auto-save graph database
    auto_build: bool = True  # Whether to auto-build if graph doesn't exist


@dataclass
class CRAGConfig:
    """CRAG multi-agent loop configuration"""
    enabled: bool = True
    max_retry: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_retry": self.max_retry
        }


@dataclass
class PipelineConfig:
    """ProcurementQA Agent complete configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    crag: Optional[CRAGConfig] = field(default_factory=CRAGConfig)
    rag_mode: str = "agentic"  # Pure Agentic RAG
    agentic_max_retries: int = 2
    
    @classmethod
    def from_env_file(cls, dotenv_path: str = None) -> "PipelineConfig":
        """
        Load configuration from .env file
        
        Args:
            dotenv_path: Path to .env file
            
        Returns:
            PipelineConfig instance
        """
        if dotenv_path is None:
            dotenv_path = "configs/thai_procurement.env" if os.path.exists("configs/thai_procurement.env") else ".env"
        from dotenv import load_dotenv
        # Retain existing environment variables (such as Docker compose networking overrides)
        load_dotenv(dotenv_path=dotenv_path, override=False)
        
        # If API key is not present, check workspace root .env
        if not env("OPENROUTER_API_KEY"):
            for candidate in ["../.env", "../../.env", ".env"]:
                cand_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(dotenv_path)), candidate))
                if os.path.exists(cand_path):
                    load_dotenv(dotenv_path=cand_path, override=False)
                    if env("OPENROUTER_API_KEY"):
                        break

        # Model configuration
        env_model_name = env("MODEL_NAME") or os.getenv("LLM_MODEL") or "openrouter"
        env_api_key = (
            env("OPENROUTER_API_KEY")
        )
        env_base_url = (
            env("LLM_BASE_URL")
            or ("https://openrouter.ai/api/v1" if (env("OPENROUTER_API_KEY") or env_model_name == "openrouter" or "/" in env_model_name) else None)
        )
        model_config = ModelConfig(
            model_name=env_model_name,
            device=env("DEVICE", "cpu"),
            prompt_language=env("PROMPT_LANGUAGE", "th"),
            api_key=env_api_key,
            base_url=env_base_url,
            max_length=int(env("LLM_MAX_TOKENS", 4096)),
            temperature=float(env("LLM_TEMPERATURE", 0.1))
        )
        
        # Data configuration
        # Written by scripts/build_corpus.py (or the migration) from data_ocr/
        default_case_db = "./outputs/corpus/cases_with_feature.json"
        default_law_to_crime = "./outputs/corpus/law_to_crime.json"
        data_config = DataConfig(
            case_db_path=env("CASE_DB_PATH", default_case_db),
            law_to_crime_path=env("LAW_TO_CRIME_PATH", default_law_to_crime),
            datasets_path=env("DATASETS_PATH", "./datasets"),
            output_dir=env("OUTPUT_DIR", "./outputs")
        )
        
        # Retrieval configuration
        def _parse_bool(val: Optional[str], default: bool) -> bool:
            if val is None:
                return default
            return str(val).strip().lower() in ("true", "1", "yes")

        retrieve_config = RetrieveConfig(
            top_retrieve=_parse_bool(env("TOP_RETRIEVE"), False),
            direct_retrieve=_parse_bool(env("DIRECT_RETRIEVE"), True),
            augment_retrieve=_parse_bool(env("AUGMENT_RETRIEVE"), True),
            top_retrieve_top_k=int(env("TOP_RETRIEVE_TOP_K", 3)),
            direct_retrieve_top_k=int(env("DIRECT_RETRIEVE_TOP_K", 10))
        )
        
        # Graph configuration
        default_graph_db = "./outputs/graph_db.pkl" if os.path.exists("./outputs/graph_db.pkl") else None
        graph_config = GraphConfig(
            graph_db_path=env("GRAPH_DB_PATH", default_graph_db),
            embedding_provider=env("EMBEDDING_PROVIDER", "local").lower(),
            embedding_api_url=env("EMBEDDING_API_URL", "http://localhost:11434/api/embed"),
            embedding_model=env("EMBEDDING_MODEL", "unsloth/embeddinggemma-300m"),
            tokenmind_api_key=env("TOKENMIND_API_KEY"),
            tokenmind_base_url=env("TOKENMIND_BASE_URL", "https://tokenmind.abdul.in.th/v1"),
            tokenmind_embedding_model=env("TOKENMIND_EMBEDDING_MODEL", "BAAI/bge-m3"),
            auto_save=_parse_bool(env("AUTO_SAVE"), True),
            auto_build=_parse_bool(env("AUTO_BUILD"), True)
        )
        
        # CRAG configuration
        crag_config = CRAGConfig(
            enabled=env("CRAG_ENABLED", "True").lower() in ("true", "1", "yes"),
            max_retry=int(env("CRAG_MAX_RETRY", 1))
        )

        # RAG mode configuration
        rag_mode = os.getenv("RAG_MODE", "agentic").lower()
        agentic_max_retries = int(os.getenv("AGENTIC_MAX_RETRIES", 2))
        
        return cls(
            model=model_config,
            data=data_config,
            retrieve=retrieve_config,
            graph=graph_config,
            crag=crag_config,
            rag_mode=rag_mode,
            agentic_max_retries=agentic_max_retries
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """
        Create configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            PipelineConfig instance
        """
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        retrieve_config = RetrieveConfig(**config_dict.get("retrieve", {}))
        graph_config = GraphConfig(**config_dict.get("graph", {}))
        crag_config = CRAGConfig(**config_dict.get("crag", {}))
        rag_mode = config_dict.get("rag_mode", "agentic")
        agentic_max_retries = int(config_dict.get("agentic_max_retries", 2))
        
        return cls(
            model=model_config,
            data=data_config,
            retrieve=retrieve_config,
            graph=graph_config,
            crag=crag_config,
            rag_mode=rag_mode,
            agentic_max_retries=agentic_max_retries
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model": {
                "model_name": self.model.model_name,
                "device": self.model.device,
                "prompt_language": self.model.prompt_language,
                "api_key": self.model.api_key,
                "base_url": self.model.base_url,
                "max_length": self.model.max_length,
                "temperature": self.model.temperature
            },
            "data": {
                "case_db_path": self.data.case_db_path,
                "law_to_crime_path": self.data.law_to_crime_path,
                "datasets_path": self.data.datasets_path,
                "output_dir": self.data.output_dir
            },
            "retrieve": self.retrieve.to_dict(),
            "graph": {
                "graph_db_path": self.graph.graph_db_path,
                "embedding_provider": self.graph.embedding_provider,
                "embedding_api_url": self.graph.embedding_api_url,
                "embedding_model": self.graph.embedding_model,
                "tokenmind_api_key": self.graph.tokenmind_api_key,
                "tokenmind_base_url": self.graph.tokenmind_base_url,
                "tokenmind_embedding_model": self.graph.tokenmind_embedding_model,
                "auto_save": self.graph.auto_save,
                "auto_build": self.graph.auto_build
            },
            "crag": self.crag.to_dict(),
            "rag_mode": self.rag_mode,
            "agentic_max_retries": self.agentic_max_retries
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "PipelineConfig":
        """Load configuration from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class ProcurementQAPipeline:
    """ProcurementQA Agent main class"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize ProcurementQA Agent
        
        Args:
            config: Configuration object, if None use default configuration
        """
        self.config = config or PipelineConfig()
        from core.prompts import set_prompt_language
        from core.retrieval.search import configure_embedding
        set_prompt_language(self.config.model.prompt_language)
        configure_embedding(
            api_url=self.config.graph.embedding_api_url,
            model=self.config.graph.embedding_model,
            provider=self.config.graph.embedding_provider,
            tokenmind_api_key=self.config.graph.tokenmind_api_key,
            tokenmind_base_url=self.config.graph.tokenmind_base_url,
            tokenmind_model=self.config.graph.tokenmind_embedding_model
        )

        # Initialize model
        self.model = self._init_model()
        
        # Load data
        self.cases_db = self._load_cases_db()
        self.law_to_crime = self._load_law_to_crime()
        
        # Initialize graph database
        if os.getenv("USE_TRI_STORE", "false").lower() in ("true", "1", "yes"):
            from core.database import StorageManager
            self.storage = StorageManager.get_instance()
            self.storage.init_all_stores()
            print("Tri-Store database active (PostgreSQL + Qdrant + Neo4j).")
        elif self.config.graph.graph_db_path and os.path.exists(self.config.graph.graph_db_path):
            GraphDBManager.load(self.config.graph.graph_db_path)
            print(f"Graph database loaded from {self.config.graph.graph_db_path}")
        else:
            GraphDBManager.initialize()
            # If auto-build is enabled and graph_db_path is configured, auto-build the graph
            if self.config.graph.auto_build and self.config.graph.graph_db_path:
                print("Graph database not found. Auto-building graph...")
                print("="*60)
                self.build_graph(force_rebuild=False)
                # After graph construction, reload if graph database file was created
                if os.path.exists(self.config.graph.graph_db_path):
                    GraphDBManager.load(self.config.graph.graph_db_path)
                    print(f"Graph database loaded after construction from {self.config.graph.graph_db_path}")
                print("="*60)
            elif self.config.graph.graph_db_path:
                print(f"Warning: Graph database not found at {self.config.graph.graph_db_path}")
                print("Set auto_build=True in config to automatically build the graph")
            else:
                print("Warning: graph_db_path not configured. Graph will not be persisted.")
    
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
            base_url=self.config.model.base_url
        )

    def _load_cases_db(self) -> List[Dict[str, Any]]:
        """Load case database"""
        _ensure_local_corpus(self.config.data.case_db_path)
        if not os.path.exists(self.config.data.case_db_path):
            if _tri_store_enabled():
                # Retrieval reads the stores; the local corpus file only feeds the legacy path
                return []
            raise FileNotFoundError(
                f"Case database not found: {self.config.data.case_db_path}"
            )
        with open(self.config.data.case_db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_law_to_crime(self) -> List[Dict[str, Any]]:
        """Load law to crime mapping"""
        _ensure_local_corpus(self.config.data.law_to_crime_path)
        if not os.path.exists(self.config.data.law_to_crime_path):
            if _tri_store_enabled():
                # Retrieval reads the stores; the local corpus file only feeds the legacy path
                return []
            raise FileNotFoundError(
                f"Law to crime mapping not found: {self.config.data.law_to_crime_path}"
            )
        with open(self.config.data.law_to_crime_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def analyze_case(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze a single case using compiled LangGraph Agentic RAG workflow.
        
        Args:
            case: Case dictionary containing "fact" and "name" fields
            
        Returns:
            List of analysis results, each element corresponds to a defendant's analysis result
        """
        retrieve_config = self.config.retrieve.to_dict()
        from core.agent import ProcurementAgenticWorkflow
        workflow = ProcurementAgenticWorkflow(
            self.model,
            retrieve_config=retrieve_config,
            max_retries=getattr(self.config, "agentic_max_retries", 2)
        )
        agent_res = workflow.invoke(case)
        return [agent_res]
    
    def analyze_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch analyze cases
        
        Args:
            cases: List of cases
            
        Returns:
            List of analysis results
        """
        results = []
        for case in cases:
            case_result = self.analyze_case(case)
            results.append({
                "case_id": case.get("id"),
                "fact": case.get("fact"),
                "analysis": case_result
            })
        return results
    
    def save_graph_db(self, filepath: Optional[str] = None):
        """Save graph database"""
        save_path = filepath or self.config.graph.graph_db_path
        if not save_path:
            raise ValueError("Graph database path not specified")
        db = GraphDBManager.get_db()
        if len(db.nodes_data) == 0 and os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
            print(f"Warning: Attempted to save empty graph database over existing file ({save_path}). Aborting save.")
            return
        GraphDBManager.save(save_path)
        print(f"Graph database saved to {save_path}")
    
    def load_graph_db(self, filepath: Optional[str] = None):
        """Load graph database"""
        load_path = filepath or self.config.graph.graph_db_path
        if not load_path:
            raise ValueError("Graph database path not specified")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Graph database not found: {load_path}")
        GraphDBManager.load(load_path)
        print(f"Graph database loaded from {load_path}")
    
    def _concat_feature_descriptions(self, description: Dict[str, Any]) -> str:
        """
        Concatenate feature descriptions
        
        Args:
            description: Feature dictionary containing defendant_info, criminal_acts, 
                        victim_property_details, intent_remorse fields, or generic fields
            
        Returns:
            Concatenated description string
        """
        res = ""
        if description.get("defendant_info"):
            res += "Defendant Info: " + ", ".join(description.get("defendant_info", [])) + ". "
        if description.get("criminal_acts"):
            res += "Criminal Acts / Topics: " + ", ".join(description.get("criminal_acts", [])) + ". "
        if description.get("victim_property_details"):
            res += "Target / Property Details: " + ", ".join(description.get("victim_property_details", [])) + ". "
        if description.get("intent_remorse"):
            res += "Intent / Remarks: " + ", ".join(description.get("intent_remorse", [])) + ". "
        
        # Support generic inquiry fields
        if not res.strip():
            parts = []
            for k, v in description.items():
                if isinstance(v, list) and v:
                    parts.append(f"{k}: {', '.join(str(x) for x in v)}")
                elif isinstance(v, str) and v.strip():
                    parts.append(f"{k}: {v.strip()}")
            res = " ".join(parts)
            
        return res
    
    def _prepare_nodes_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Prepare node data
        
        Returns:
            Dictionary containing 'case', 'law', 'crime' keys
        """
        case_nodes_data = []
        law_nodes_data = []
        crime_nodes_data = []
        
        # Process case nodes
        for case in tqdm(self.cases_db, desc="Preparing case nodes"):
            description = ""
            if "features" in case and case["features"]:
                description = self._concat_feature_descriptions(case["features"])
            if not description.strip():
                description = case.get("fact") or case.get("description") or case.get("question", "")

            case_nodes_data.append({
                'id': str(uuid.uuid4()),
                'description': description,
                'caseId': str(case.get("id", "")),
                'crime': case.get("crime", []),
                'law': [str(l) for l in case.get("laws", case.get("law", []))],
                'type': 'case'
            })
        
        # Process law nodes and crime nodes
        crimes = set()
        for law in tqdm(self.law_to_crime, desc="Preparing law and crime nodes"):
            text_id = law.get("id")
            
            # Process items field
            if "items" in law and law["items"]:
                for item in law["items"]:
                    # Collect crimes
                    if "crime" in item:
                        if isinstance(item["crime"], list):
                            crimes.update(item["crime"])
                        else:
                            crimes.add(item["crime"])
                    
                    # Create law node
                    law_nodes_data.append({
                        'id': str(uuid.uuid4()),
                        'entry': str(text_id),
                        'description': item.get("text", item.get("description", "")),
                        'crimes': item.get("crime", []),
                        "judge_dep": str(item.get("judge_dep", [])),
                        "related_laws": str(item.get("related_laws", [])),
                        'type': 'law'
                    })
            else:
                # If no items field, process law object directly
                if "crime" in law:
                    if isinstance(law["crime"], list):
                        crimes.update(law["crime"])
                    else:
                        crimes.add(law["crime"])
                
                law_nodes_data.append({
                    'id': str(uuid.uuid4()),
                    'entry': str(text_id),
                    'description': law.get("text", law.get("description", "")),
                    'crimes': law.get("crime", []),
                    "judge_dep": str(law.get("judge_dep", [])),
                    "related_laws": str(law.get("related_laws", [])),
                    'type': 'law'
                })
        
        # Create crime / topic nodes
        crimes = list(crimes)
        for crime in crimes:
            if crime and str(crime).strip():
                crime_nodes_data.append({
                    'id': str(uuid.uuid4()),
                    'description': str(crime).strip(),
                    'type': 'crime'
                })
        
        return {
            'case': case_nodes_data,
            'law': law_nodes_data,
            'crime': crime_nodes_data
        }
    
    def build_graph(self, force_rebuild: bool = False):
        """
        Build graph structure
        
        Args:
            force_rebuild: If True, rebuild even if graph database already exists
        """
        from core.retrieval.search import construct_feature_graph
        
        # Check if graph database already exists
        if not force_rebuild and self.config.graph.graph_db_path and os.path.exists(self.config.graph.graph_db_path):
            print(f"Graph database already exists at {self.config.graph.graph_db_path}")
            print("Use force_rebuild=True to rebuild the graph")
            return
        
        print("Starting graph construction...")
        
        # Prepare node data
        nodes_data = self._prepare_nodes_data()
        
        print(f"Prepared {len(nodes_data['case'])} case nodes, "
              f"{len(nodes_data['law'])} law nodes, "
              f"{len(nodes_data['crime'])} crime nodes")
        
        # Build graph structure
        construct_feature_graph(self.model, nodes_data)
        
        # Save graph database
        if self.config.graph.graph_db_path:
            self.save_graph_db()
            print(f"Graph construction completed and saved to {self.config.graph.graph_db_path}")
        else:
            print("Graph construction completed (not saved, graph_db_path not specified)")
    
    def __del__(self):
        """Destructor, auto-save graph database safely if interpreter is not tearing down."""
        try:
            import os
            import builtins
            if not hasattr(builtins, "open") or builtins.open is None:
                return
            if os.getenv("USE_TRI_STORE", "false").lower() in ("true", "1", "yes"):
                return
            if hasattr(self, 'config') and getattr(self.config.graph, 'auto_save', False) and getattr(self.config.graph, 'graph_db_path', None):
                db = GraphDBManager.get_db()
                if db is not None and getattr(db, 'nodes_data', None) and len(db.nodes_data) > 0:
                    self.save_graph_db()
        except Exception:
            pass
