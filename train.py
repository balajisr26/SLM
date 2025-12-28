from modules.tokenizer_wrapper import TokenizerWrapper
#from modules.PDFLoader import LoadCorpus
from modules.Corpus_Builder import DatasetBuilder
from modules.Transformers import TransformerBlock
from modules.HelperClasses import LayerNorm
from modules.GPT import GPT
import config as ds_config 
import pandas as pd
import numpy as np
import csv
import sys
import torch
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR
from tqdm.auto import tqdm
import config

resume = int(sys.argv[1])
num_epochs = int(sys.argv[2])

print("resume",resume)
print("num_epochs",num_epochs)

#reader = LoadCorpus()
#reader.CreateCorpus()
tokenizer = TokenizerWrapper('gpt2')
db = DatasetBuilder(ds_config.corpus_file,ds_config.bin_dir,ds_config.output_dir
                        ,read_batch_size=1024,tokenizer=tokenizer)
db.build()

GPT = GPT
model = GPT(ds_config)


learning_rate =  config.learning_rate
max_iters = config.max_iters 
min_lr = config.min_lr
eval_iters = 500 
gradient_accumulation_steps = config.gradient_accumulation_steps 

max_optimizer_steps = max_iters // gradient_accumulation_steps
warmup_steps = int(0.05 * max_optimizer_steps)
decay_steps = max_optimizer_steps - warmup_steps

device =  "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler

# How to use autocast https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_device(device)

torch.manual_seed(42)

##PUT IN WEIGHT DECAY, CHANGED BETA2 to 0.95
optimizer =  torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9) #weight decay for regularization

scheduler_warmup = LinearLR(optimizer, start_factor=0.01,total_iters = warmup_steps) #Implement linear warmup
scheduler_decay = CosineAnnealingLR(optimizer,T_max = decay_steps, eta_min = min_lr) #Implement lr decay
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps]) #Switching from warmup to decay

# https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

def estimate_loss(model):
   # print("Inside Estimate Loss Function")
    out = {}
    model.eval()
    eval_iters = 200
    with torch.inference_mode():
       # print("In Inference Mode")
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
           # print("Get Batch and Loss")
            # Change to 100 from eval_iters for quicker eval
            for k in range(eval_iters):
                X, Y = db.get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
          #  print("Out of Loop")
    model.train()
    return out

optimizer_step=0
best_val_loss = float('inf')
best_model_params_path = "arxiv_8_8_512.pt"
eval_iters = 500

# Checkpointing - Uncomment if required
if resume > 0:
    checkpoint = torch.load(best_model_params_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs_list = checkpoint['epochs_list']
    train_loss_list = checkpoint['train_loss']
    validation_loss_list = checkpoint['validation_loss']
    best_val_loss = checkpoint['best_val_loss']
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    optimizer_step = checkpoint['optimizer_step']
    scheduler.last_epoch = optimizer_step
    lr_list = checkpoint['learning_rate']

    model.train()

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)




# Ensure model is on the correct device
model = model.to(device)



if resume ==0:
    epochs = 0
    train_loss_list, validation_loss_list,epochs_list,lr_list = [], [],[],[]
    total_epochs = num_epochs
else:
    epochs = epochs_list[len(epochs_list) - 1] + 1
    total_epochs = epochs + num_epochs

print("Starting from Epoch",epochs,"To",total_epochs)



for epoch in tqdm(range(epochs,total_epochs)):#

    if epoch % eval_iters == 0 and epoch != 0:
        # Ensure estimate_loss uses the correct device
        print("Estimating Losses...")
        losses = estimate_loss(model)
        print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        train_loss_list.append(losses['train'])
        validation_loss_list.append(losses['val'])
        lr_list.append(optimizer.param_groups[0]['lr'] )
        epochs_list.append(epoch)

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
       

#            # Checkpoint -- Uncomment if required
            checkpoint = {
                'epochs_list': epochs_list,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss_list,
                'validation_loss': validation_loss_list,
                'best_val_loss': best_val_loss,
                "scaler_state_dict": scaler.state_dict(),
                "learning_rate": lr_list ,  
                'optimizer_step': optimizer_step
                }
            torch.save(checkpoint, best_model_params_path)
          

    # Ensure X and y are on the correct device
    X, y = db.get_batch("train")
    X, y = X.to(device), y.to(device)

    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
       
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
        scaler.step(optimizer)
 
        scaler.update()
        scheduler.step()
  
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
    #scheduler.step()
        

train_losses_cpu = [l.detach().cpu().item() for l in train_loss_list]
val_losses_cpu   = [l.detach().cpu().item() for l in validation_loss_list]
#learning_rate_cpu = [l.detach().cpu().item() for l in lr_list]
loss_history = pd.DataFrame({
    "epoch": range(len(train_loss_list)),
    "train_loss": train_losses_cpu,
    "val_loss": val_losses_cpu,
    #'learning_rate':learning_rate_cpu
    
})
loss_history.to_csv("slm_loss_history_arxiv_8_8_512.csv", index=False)
print(loss_history)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(loss_history['train_loss'], label="Train Loss")
plt.plot(loss_history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

ckpt = torch.load(best_model_params_path, map_location="cpu")


stats_df = pd.DataFrame({'epoch': ckpt["epochs_list"],
    '                  train_loss': [float(x.detach().cpu()) if torch.is_tensor(x) else x
                                                    for x in ckpt["train_loss"]],
                        'val_loss': [float(x.detach().cpu()) if torch.is_tensor(x) else x
                                          for x in ckpt["validation_loss"]],
                         'learning_rate': [float(x.detach().cpu()) if torch.is_tensor(x) else x
                                          for x in ckpt["learning_rate"]]
                } 
                         )
stats_df.to_csv("stats_arxiv_8_8_512.csv", index=False)