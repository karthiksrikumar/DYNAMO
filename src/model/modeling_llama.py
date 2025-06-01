# File: dynamo_llm/models/modeling_llama.py
import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaPreTrainedModel, LlamaConfig
from .adapters import TemporalAdapter, CausalTemporalAdapter

class LlamaWithTemporalAdapters(LlamaPreTrainedModel):
    config_class = LlamaConfig
    
    def __init__(self, config, adapter_config=None):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.adapter_config = adapter_config or {}
        self.adapters = nn.ModuleDict()
        
        # Initialize adapters if enabled
        if self.adapter_config.get('use_adapters', False):
            adapter_type = self.adapter_config.get('adapter_type', 'temporal')
            hidden_size = config.hidden_size
            temporal_embed_dim = self.adapter_config.get('temporal_embed_dim', 16)
            causal_embed_dim = self.adapter_config.get('causal_embed_dim', 64)
            
            for i in range(config.num_hidden_layers):
                if adapter_type == 'temporal':
                    self.adapters[f'layer_{i}'] = TemporalAdapter(
                        hidden_size, temporal_embed_dim
                    )
                elif adapter_type == 'temporal_causal':
                    self.adapters[f'layer_{i}'] = CausalTemporalAdapter(
                        hidden_size, temporal_embed_dim, causal_embed_dim
                    )
        
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        timestamps=None,        # (batch_size, 1) normalized timestamps
        causal_embs=None        # (batch_size, causal_embed_dim) for each example
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward pass through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]  # (batch_size, seq_len, hidden_size)
        
        # Apply adapters if enabled
        if self.adapter_config.get('use_adapters', False) and output_hidden_states:
            # Apply adapters to each layer's output
            adapted_hidden_states = []
            for i, layer_hidden in enumerate(outputs.hidden_states):
                if f'layer_{i}' in self.adapters:
                    adapter = self.adapters[f'layer_{i}']
                    if isinstance(adapter, TemporalAdapter):
                        layer_hidden = adapter(layer_hidden, timestamps)
                    elif isinstance(adapter, CausalTemporalAdapter):
                        layer_hidden = adapter(layer_hidden, timestamps, causal_embs)
                adapted_hidden_states.append(layer_hidden)
            
            # Use the last layer's adapted hidden state
            hidden_states = adapted_hidden_states[-1]
        
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': lm_logits,
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }
