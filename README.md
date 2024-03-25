# Assignment 1 c

This code fine tunes 3 models (Llama2, Phi2, and Mistral) on the Standord Alpaca dataset. It then evaluates the models' BLEU, Rogue-L, and BERTScores.

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

