# File: dynamo_llm/inference/infer.py
import torch
from transformers import LlamaTokenizer, pipeline
from models.modeling_llama import LlamaWithTemporalAdapters
from utils import normalize_timestamp, get_causal_embedding

class DynamoInference:
    def __init__(self, model_path, causal_path):
        self.model = LlamaWithTemporalAdapters.from_pretrained(model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.causal_path = causal_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def generate_response(self, question, entity, timestamp):
        # Normalize timestamp and get causal embedding
        norm_timestamp = normalize_timestamp(timestamp)
        causal_emb = get_causal_embedding(entity, self.causal_path)
        
        # Prepare inputs
        inputs = self.tokenizer(
            f"Question: {question} Answer:",
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)
        
        timestamps = torch.tensor([norm_timestamp], device=self.device)
        causal_embs = torch.tensor([causal_emb], device=self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                timestamps=timestamps,
                causal_embs=causal_embs,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        return response


if __name__ == "__main__":
    inference = DynamoInference(
        model_path="outputs/final_model",
        causal_path="data/causal_graphs.parquet"
    )
    
    # Example usage
    response = inference.generate_response(
        question="What was COVID policy in 2025?",
        entity="COVID-19",
        timestamp="2025-06-01"
    )
    print(f"Response: {response}")
