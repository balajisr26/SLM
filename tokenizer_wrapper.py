import tiktoken

class TokenizerWrapper:
    def __init__(self,model_name:str='gpt2'):
        self.enc = tiktoken.get_encoding(model_name)

    def encode(self,text:str) -> list[int]:
        return self.enc.encode_ordinary(text)
    
    def decode(self,ids:list[int]) ->str:
        return self.enc.decode(ids)
