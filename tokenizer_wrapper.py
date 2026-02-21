import tiktoken
import sentencepiece as spm

class TokenizerWrapper:
    def __init__(self,tokenizer_type:str='gpt2',model_path:str=None):
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == 'gpt2':
            self.enc = tiktoken.get_encoding(tokenizer_type)
            self.vocab_size = self.enc.n_vocab
        elif tokenizer_type == 'sentencepiece':
            if model_path is None:
                raise ValueError("model_path must be provided for sentencepiece tokenizer")
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            self.vocab_size = self.sp.get_piece_size()
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    def encode(self,text:str) -> list[int]:
        if self.tokenizer_type == 'gpt2':
            return self.enc.encode_ordinary(text)
        elif self.tokenizer_type == 'sentencepiece':
            
            return self.sp.encode(text,
                out_type=int
                #enable_sampling=train,   
                #alpha=0.1,
                #nbest_size=-1
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

    def decode(self,ids:list[int]) ->str:
        if self.tokenizer_type == 'gpt':
            return self.enc.decode(ids)
        elif self.tokenizer_type == 'sentencepiece':
            return self.sp.decode(ids)
        