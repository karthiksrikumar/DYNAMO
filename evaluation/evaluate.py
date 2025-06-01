import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizer
from models.modeling_llama import LlamaWithTemporalAdapters
from utils import normalize_timestamp, get_causal_embedding

class DynamoEvaluator:
    def __init__(self, model_path, qa_path, kb_path, causal_path):
        self.model = LlamaWithTemporalAdapters.from_pretrained(model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qa_path = qa_path
        self.kb_path = kb_path
        self.causal_path = causal_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def load_qa_data(self):
        with open(self.qa_path, 'r') as f:
            return [json.loads(line) for line in f]
    
    def evaluate_temporal_accuracy(self):
        qa_data = self.load_qa_data()
        correct = 0
        total = 0
        
        for item in tqdm(qa_data, desc="Evaluating Temporal Accuracy"):
            timestamp = normalize_timestamp(item['timestamp'])
            causal_emb = get_causal_embedding(item['main_entity'], self.causal_path)
            
            # Prepare input
            input_text = f"Question: {item['question']} Answer:"
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=256, 
                truncation=True
            ).to(self.device)
            
            # Prepare timestamp and causal embedding
            timestamps = torch.tensor([timestamp], device=self.device)
            causal_embs = torch.tensor([causal_emb], device=self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    timestamps=timestamps,
                    causal_embs=causal_embs,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False
                )
            
            # Decode and check answer
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            if item['answer'].lower() in response.lower():
                correct += 1
            total += 1
        
        return correct / total
    
    def evaluate_causal_f1(self):
        # Implementation for causal F1 evaluation
        # This would test the model's ability to identify correct causal relationships
        # across different time periods
        pass
    
    def evaluate_temporal_consistency(self):
        # Implementation for temporal consistency evaluation
        # This would test if the model maintains consistent knowledge
        # about entities across time
        pass


if __name__ == "__main__":
    evaluator = DynamoEvaluator(
        model_path="outputs/final_model",
        qa_path="data/test_temporal_qa.jsonl",
        kb_path="data/temporal_kb.parquet",
        causal_path="data/causal_graphs.parquet"
    )
    
    temporal_accuracy = evaluator.evaluate_temporal_accuracy()
    print(f"Temporal Accuracy: {temporal_accuracy:.4f}")
    
    # Additional evaluations would be called here
    # causal_f1 = evaluator.evaluate_causal_f1()
    # temporal_consistency = evaluator.evaluate_temporal_consistency()
