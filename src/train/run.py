python run_trainer.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --train_qa_path data/temporal_qa.jsonl \
  --kb_path data/temporal_kb.parquet \
  --causal_path data/causal_graphs.parquet \
  --output_dir outputs \
  --epochs 10 \
  --batch_size 4 \
  --learning_rate 5e-5
