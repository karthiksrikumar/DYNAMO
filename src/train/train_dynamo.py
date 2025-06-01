import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from src.config import load_config
from src.data_loader import create_dataloaders
from src.model import DynamoModel
from src.trainer import DynamoTrainer
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", 
                        help="Run in debug mode with small dataset")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config.debug_mode = args.debug
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config.data, 
        tokenizer, 
        debug=config.debug_mode
    )
    
    # Initialize model
    model = DynamoModel(
        base_model_name=config.model.base_model,
        time_embed_dim=config.model.time_embed_dim,
        causal_embed_dim=config.model.causal_embed_dim,
        adapter_rank=config.model.adapter_rank
    ).to(device)
    
    # Set up optimizer (only train adapter parameters)
    trainable_params = []
    for name, param in model.named_parameters():
        if "adapter" in name or "scale" in name:
            trainable_params.append(param)
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Set up learning rate scheduler
    total_steps = len(train_loader) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize trainer
    trainer = DynamoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
