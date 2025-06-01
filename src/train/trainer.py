import torch
from tqdm import tqdm
import wandb
import os
from datetime import datetime

class DynamoTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = next(model.parameters()).device
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = os.path.join(config.checkpoint_dir, f"run_{timestamp}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(project="dynamo-llm")
            wandb.config.update({
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "epochs": config.training.epochs,
                "time_embed_dim": config.model.time_embed_dim,
                "causal_embed_dim": config.model.causal_embed_dim
            })
            wandb.watch(self.model)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_steps = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            time_input = batch['time'].to(self.device)
            causal_input = batch['causal_embed'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                time_input=time_input,
                causal_input=causal_input,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Update tracking
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "step": epoch * total_steps + step
                })
        
        return total_loss / total_steps

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_steps = len(self.val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Validation {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                time_input = batch['time'].to(self.device)
                causal_input = batch['causal_embed'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    time_input=time_input,
                    causal_input=causal_input,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)
                
                progress_bar.set_postfix({"val_loss": f"{avg_loss:.4f}"})
                
                # Log to wandb
                if self.config.use_wandb:
                    wandb.log({
                        "val/loss": loss.item(),
                        "val/avg_loss": avg_loss,
                        "step": epoch * len(self.train_loader) + step
                    })
        
        avg_val_loss = total_loss / total_steps
        return avg_val_loss

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        filename = f"checkpoint_epoch_{epoch+1}.pt"
        if is_best:
            filename = "best_checkpoint.pt"
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))

    def train(self):
        for epoch in range(self.config.training.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            print(f"Epoch {epoch+1}/{self.config.training.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping if needed (optional)
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "final_model.pt"))
        
        if self.config.use_wandb:
            wandb.finish()
