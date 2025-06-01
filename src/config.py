import yaml
from dataclasses import dataclass

@dataclass
class ModelConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B"
    time_embed_dim: int = 64
    causal_embed_dim: int = 128
    adapter_rank: int = 8
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 256
    causal_alpha: float = 0.5  # Causal regularization weight

@dataclass
class DataConfig:
    train_path: str = "data/train.jsonl"
    val_path: str = "data/val.jsonl"
    temporal_kb_path: str = "data/wikidata_temporal.parquet"
    causal_graph_path: str = "data/causal_graphs.parquet"
    max_samples: int = 10000  # For debugging

@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = True
    debug_mode: bool = False

def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ExperimentConfig(
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training']),
        data=DataConfig(**config_dict['data'])
    )
