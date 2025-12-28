import torch
input_directory = "./input/"
corpus_file = "./corpus/arxiv_ai_corpus.txt"
output_dir ="./output/"
bin_dir = "./bin/"
block_size = 512
batch_size = 4
device =  "cuda" if torch.cuda.is_available() else "cpu"
n_layers = 8
vocab_size = 50257
n_head=8
n_embd=256
dropout=0.1
bias=True
gradient_accumulation_steps=8
learning_rate =  3e-4
min_lr = 2e-5
max_iters = 56001 