import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from .adapter import CausalTemporalAdapter

class DynamoModel(nn.Module):
    def __init__(self, base_model_name, time_embed_dim, causal_embed_dim, adapter_rank=8):
        super().__init__()
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.hidden_size = self.base_model.config.hidden_size
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Initialize adapters
        self.adapters = nn.ModuleList([
            CausalTemporalAdapter(
                hidden_size=self.hidden_size,
                time_embed_dim=time_embed_dim,
                causal_embed_dim=causal_embed_dim,
                rank=adapter_rank
            )
            for _ in range(self.base_model.config.num_hidden_layers)
        ])
        
        # Learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, time_input, causal_input, labels=None):
        # Base model forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Apply adapters to each layer's output
        hidden_states = outputs.hidden_states
        new_hidden_states = (hidden_states[0],)  # Embedding layer
        
        for i, (layer, adapter) in enumerate(zip(
            self.base_model.model.layers, 
            self.adapters
        )):
            # Original layer output
            layer_output = hidden_states[i+1]
            
            # Apply adapter
            adapted_output = adapter(
                layer_output,
                time_input,
                causal_input
            )
            
            # Residual connection with scaling
            new_output = layer_output + self.scale * adapted_output
            new_hidden_states += (new_output,)
        
        # Get final hidden states
        last_hidden_state = new_hidden_states[-1]
        
        # Compute logits
        logits = self.base_model.lm_head(last_hidden_state)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': new_hidden_states
        }
