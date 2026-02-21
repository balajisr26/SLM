import re
from modules.tokenizer_wrapper import TokenizerWrapper
import csv
import os
from tqdm import tqdm
import numpy as np
import config
import torch
import re
import sys
import config as ds_config 


class TestBuilder:
    def __init__(self,test_corpus,bin_dir,output_path,tokenizer:TokenizerWrapper
                 ,read_batch_size=10000,write_batch_size = 1024):
        
        self.test_corpus = test_corpus
        self.bin_dir = bin_dir
        self.output_path = output_path
        self.tokenizer = tokenizer
    
        self.read_batch_size = read_batch_size
        self.write_batch_size = write_batch_size

        os.makedirs(self.output_path,exist_ok=True)
        self.unique_tokens = set()
        self.total_tokens = 0

    def text_formatter(self,text):
      
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s.,;:!?%$()\-/]","",text)
        cleaned_text = re.sub(r"\s+"," ",cleaned_text)
        cleaned_text =re.sub(r"[\t]+"," ",cleaned_text)
       
        return cleaned_text

   

    def read_corpus_batches(self):
        
        new_limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(new_limit)
                break
            except OverflowError:
                # Decrease the limit by a factor of 10 if it's too large
                new_limit = int(new_limit / 10)
        with open(self.test,'r',encoding='utf-8') as f:
           # csvreader = csv.reader(f,delimiter="|")
            csvreader = csv.reader(f)
            lines = []
            no_row = 0
            for row in csvreader:
                no_row +=1
               # print("No Row",no_row)
                line = ", ".join(row).strip()
                line = self.text_formatter(line)
                
              
                if not line:
                    continue
           
                lines.append(line + '\n')

                if len(lines) >= self.read_batch_size:
                    yield  lines
                   # yield [self.tokenizer.encode(l) for l in lines]
                    lines = []

             #   if no_row > 100:
              #      break

            if lines:
                
                yield lines

        print("Total Number of Rows Read:",no_row)

    def process_corpus_batches(self):
        tokenized_batch=[]
      #  tokenized_val=[]

        for batch_idx,batch in enumerate(tqdm(self.read_corpus_batches())):
            batch_tokens = [self.tokenizer.encode(line) for line in batch]

            for toks in batch_tokens:
                self.unique_tokens.update(toks)
                self.total_tokens += len(toks)

           # n = len(batch_tokens)
           # n_val = int(n * self.val_split)
            tokenized_batch.extend(batch_tokens)
          #  tokenized_val.extend(batch_tokens[n - n_val:])

        return tokenized_batch

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
            
        #data = self.make_strided_data(data, config.block_size, config.stride)
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

        print("Inside Build")

        test_ids = self.process_corpus_batches()

        print("Write Corpus")
        self.write_corpus(test_ids,'test')


        # Token usage summary
        print("\n📊 Tokenization Summary")
        print(f"Total tokens processed: {self.total_tokens:,}")
        print(f"Unique token IDs used: {len(self.unique_tokens):,}")
    
if __name__ == "__main__":

    print("Inside")