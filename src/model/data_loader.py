import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from .config import DataConfig

class TemporalCausalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256, 
                 temporal_kb=None, causal_graphs=None, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        
        # Load data
        self.data = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.data.append(json.loads(line))
        
        # Load knowledge bases
        self.temporal_kb = pd.read_parquet(temporal_kb) if temporal_kb else None
        self.causal_graphs = pd.read_parquet(causal_graphs) if causal_graphs else None

    def __len__(self):
        return len(self.data)
    
    def _get_temporal_embedding(self, timestamp):
        # Placeholder for time2vec implementation
        year = timestamp.year
        month = timestamp.month
        return [year, month]
    
    def _get_causal_embedding(self, entity):
        # Placeholder for causal graph lookup
        if self.causal_graphs is not None and entity in self.causal_graphs.index:
            return self.causal_graphs.loc[entity].values
        return [0] * 128  # Default zero vector

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize question and answer
        question = item['question']
        answer = item['answer']
        
        # Get temporal context
        timestamp = pd.to_datetime(item['timestamp'])
        time_embed = self._get_temporal_embedding(timestamp)
        
        # Get causal context
        causal_embed = self._get_causal_embedding(item['main_entity'])
        
        # Tokenize inputs
        inputs = self.tokenizer(
            question,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize labels
        labels = self.tokenizer(
            answer,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )['input_ids']
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'time': torch.tensor(time_embed, dtype=torch.float),
            'causal_embed': torch.tensor(causal_embed, dtype=torch.float)
        }

def create_dataloaders(config: DataConfig, tokenizer, debug=False):
    train_dataset = TemporalCausalDataset(
        config.train_path,
        tokenizer,
        max_length=256,
        temporal_kb=config.temporal_kb_path,
        causal_graphs=config.causal_graph_path,
        max_samples=config.max_samples if debug else None
    )
    
    val_dataset = TemporalCausalDataset(
        config.val_path,
        tokenizer,
        max_length=256,
        temporal_kb=config.temporal_kb_path,
        causal_graphs=config.causal_graph_path,
        max_samples=config.max_samples if debug else None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4 if debug else config.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4 if debug else config.batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader
