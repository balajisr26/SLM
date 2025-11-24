import fitz
import config
import os
import pandas as pd
import re

class LoadCorpus:
    def __init__(self):
        print(config.input_directory)

    def text_formatter(self,text):
      #  cleaned_text = text.replace(r'\n','').strip()
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s.,;:!?%$()\-/]","",text)
        cleaned_text = re.sub(r"\s+"," ",cleaned_text)
        cleaned_text =re.sub(r"[\t]+"," ",cleaned_text)
        return cleaned_text
    
    def split_sentences(self,text):
        
        text = re.split(r'(?<=[.!?])\s+',text)
        return text

    def CreateCorpus(self):
        files = [file for file in os.listdir(config.input_directory) if file.lower().endswith('.pdf')]
        corpus = []
        for file in files:
            print('Reading File:',file)
            doc = fitz.open(os.path.join(config.input_directory,file))
            for page in doc:
                text = page.get_text()
                text = self.text_formatter(text)
                corpus.extend(self.split_sentences(text))
        corpus = '\n'.join(corpus)

        with open(config.corpus_file,'w',encoding='utf-8') as f:
            f.write(corpus)
      #  return corpus                               

if __name__=="__main__":
    reader = LoadCorpus()
    text = reader.CreateCorpus()
    with open(config.corpus_file,'w',encoding='utf-8') as f:
        f.write(text)