from modules.tokenizer_wrapper import TokenizerWrapper
import csv
import os
from tqdm import tqdm
import numpy as np
import config
import torch


class DatasetBuilder:
    def __init__(self,corpus,bin_dir,output_path,tokenizer:TokenizerWrapper,val_split=0.1
                 ,read_batch_size=10000,write_batch_size = 1024):
        
        self.corpus = corpus
        self.bin_dir = bin_dir
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.val_split = val_split
        self.read_batch_size = read_batch_size
        self.write_batch_size = write_batch_size

        os.makedirs(self.output_path,exist_ok=True)
        self.unique_tokens = set()
        self.total_tokens = 0

    def read_corpus_batches(self):

        with open(self.corpus,'r',encoding='utf-8') as f:
           # csvreader = csv.reader(f,delimiter="|")
            csvreader = csv.reader(f)
            lines = []
            no_row = 0
            for row in csvreader:
                no_row +=1
              #  print("Row:",row)
                #line = ", ".join(row[1])
                line = ", ".join(row).strip()
              #  print("Line",line)
                if not line:
                    continue
             #   line += '\n'
                lines.append(line + '\n')

                if len(lines) >= self.read_batch_size:
                    yield  lines
                   # yield [self.tokenizer.encode(l) for l in lines]
                    lines = []

             #   if no_row > 100:
              #      break

            if lines:
                #yield([self.tokenizer.encode(l) for l in lines])
                yield lines

        print("Total Number of Rows Read:",no_row)

    def process_corpus_batches(self):
        tokenized_train=[]
        tokenized_val=[]

        for batch_idx,batch in enumerate(tqdm(self.read_corpus_batches())):
            batch_tokens = [self.tokenizer.encode(line) for line in batch]

            for toks in batch_tokens:
                self.unique_tokens.update(toks)
                self.total_tokens += len(toks)

            n = len(batch_tokens)
            n_val = int(n * self.val_split)
            tokenized_train.extend(batch_tokens[:n - n_val])
            tokenized_val.extend(batch_tokens[n - n_val:])

        return tokenized_train, tokenized_val

           # print(f"Processed batch {batch_idx + 1}: "
           #       f"{len(batch_tokens)} lines, total tokens so far = {self.total_tokens:,}")
          #  print("\n📊 Tokenization Summary")
          #  print(f"Total tokens processed: {self.total_tokens:,}")
          #  print(f"Unique token IDs used: {len(self.unique_tokens):,}")

    def write_corpus(self,token_ids,split):
        bin_filename = os.path.join(self.bin_dir,f'{split}.bin')
        arr_len = sum(len(x) for x in token_ids)
        dtype =np.uint16

        arr = np.memmap(bin_filename,dtype=dtype,mode='w+',shape=(arr_len,))
        idx = 0
        for toks in tqdm(token_ids,desc=f'Writing {split}.bin'):
            arr[idx:idx+len(toks)] = toks
            idx += len(toks)
        
        arr.flush()
        print(f'{split}.bin.written:{arr_len:,} tokens')

    def get_batch(self,split):

        if split == 'train':
            data = np.memmap(config.bin_dir+'/train.bin',dtype=np.uint16,mode='r')
        else:
            data=np.memmap(config.bin_dir+'/val.bin',dtype=np.uint16,mode='r')
            
        ix = torch.randint(len(data) - config.block_size,(config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) 
                            for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) 
                         for i in ix])
        
        if config.device=='cuda':
            x,y = x.pin_memory().to(config.device,non_blocking=True),y.pin_memory().to(config.device,non_blocking=True)
        else:
            x,y = x.to(config.device),y.to(config.device)

        return x,y



    def build(self):

     train_ids, val_ids = self.process_corpus_batches()

     self.write_corpus(train_ids,'train')
     self.write_corpus(val_ids,'val')

     # Token usage summary
     print("\n📊 Tokenization Summary")
     print(f"Total tokens processed: {self.total_tokens:,}")
     print(f"Unique token IDs used: {len(self.unique_tokens):,}")
    # print(f"GPT-2 vocab coverage: {len(self.unique_tokens) / self.tokenizer.vocab_size * 100:.2f}%")
     #print(f"GPT-2 vocab size: {self.tokenizer.vocab_size:,}")
     #print(f"Output written to: {os.path.abspath(self.output_dir)}")
       
      




