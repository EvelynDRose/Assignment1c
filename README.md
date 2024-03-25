# Assignment 1 c

This code provides the implementation of *RoBERTa-PFGCN* as described in out paper, a method to generate Graph of 
Program dubbed SVG with our novel Poacher Flow Edges. We use RoBERTa to generate embeddings and GCN for vulnerability detection and classification.

#### Requirements
- Python 	3.11
- Pytorch 	 
- Transformer 	
- datasets
- evaluate
- trl
- peft
- tabulate
- statistics

### Datasets
- Download the Stanford Alpaca dataset at https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release

### Reproducibility
Just uncomment the model youd like to use and run the python file

## References  
- https://huggingface.co/docs/transformers/en/training
- https://www.datacamp.com/tutorial/fine-tuning-llama-2
- https://huggingface.co/docs/peft/main/en/tutorial/peft_model_config
- https://github.com/brevdev/notebooks/blob/main/llama2-finetune-own-data.ipynb

