print('start main')

from datasets import Dataset
import torch, functools, transformers, os, subprocess
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import json
from utils import (
    pad_to_length,
    get_local_dir,
    gpu_memory_usage,
    save_checkpoint,
)




# init

import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--outputdir', type=str)
parser.add_argument('--modelname', type=str, default="hfl/chinese-alpaca-2-13b")
config = parser.parse_args()
print(config)

class RewardDataset(Dataset):
    def __init__(self):
        with open('data/data_scores_1.json', encoding="utf-8") as file_obj: #read
            score = json.load(file_obj)
        with open('data/scoring_data_1.json', encoding="utf-8") as file_obj: #read
            data = json.load(file_obj)
        self.score = score
        self._data = data
        print(len(score), score[0])
        print(len(data), data[0])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, data_idx):
        return f'''You are a Chinese sentence grader. Given Chinese sentence, you need to give scores in range 1-4 (4 is the highest) based on sentence fluency. 

            Here are the metrics:
            Score 1: Not fluent at all, can't understand the meaning easily.
            Score 2: There are inappropriate or awkward phrases or other big unbearable flaws.
            Score 3: Quite fluent. There exists small mistakes, but they are acceptable.
            Score 4: Very fluent and no mistakes. Likely come from a native Chinese speaker.
            
            Now, I will provide you with the Chinese sentence. You need to give me only one number and nothing else. 
            The Chinese sentence is: {self.data[data_idx]["zh_line"]}''', self.score[data_idx]


dist.init_process_group("nccl")
rank = dist.get_rank()
device_id = rank % torch.cuda.device_count()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(device_id)

model_name = config.modelname

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./cache')
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


dist.barrier()
if rank == 0: dataset = RewardDataset() # !!!
dist.barrier()
if rank != 0: dataset = RewardDataset() # !!!
dist.barrier()


# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./cache')
# policy_model_dtype = getattr(torch, 'float32')
# model = AutoModelForCausalLM.from_pretrained(model_name,
#     cache_dir=get_local_dir(['.cache',]), 
#     low_cpu_mem_usage=True, 
#     torch_dtype=policy_model_dtype)


# for i in range(len(model.model.layers)):
#     model.model.layers[i] = checkpoint_wrapper(model.model.layers[i])

# model = FSDP(model,
#             device_id=torch.cuda.current_device(),
#             mixed_precision=MixedPrecision(param_dtype=torch.float16,reduce_dtype=torch.float16,buffer_dtype=torch.float16),
#             auto_wrap_policy=functools.partial(
#                 transformer_auto_wrap_policy,
#                 transformer_layer_cls={transformers.models.llama.modeling_llama.LlamaDecoderLayer},
#             )
#         )

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# model.train()

dist.barrier()

# train

train_loader = DataLoader(
    dataset,
    sampler=DistributedSampler(dataset, shuffle=True, drop_last=False),
    batch_size=1,
)

iter_num = 0
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
scaler = ShardedGradScaler()
acc_steps = 10

print('total', len(train_loader), rank, world_size)
print(len(dataset), dataset[0])
data_iter = iter(train_loader)
batch = next(data_iter)
print(batch)
exit(0)

while iter_num <= 250:
    data_iter = iter(train_loader)
    while True:
        try:
            batch = next(data_iter)
        except:
            break
        token_ids = torch.stack([torch.cat(o, dim=0) for o in batch['input_ids']], dim=0).to(device_id)
        print(token_ids.shape)
        x = token_ids[:, :-1]
        y = token_ids[:, 1:]
        logits = model(torch.max(x,torch.tensor(0))).logits
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        print(loss)
        
        scaler.scale(loss / acc_steps).backward()
        if iter_num == 0: optimizer.zero_grad()
        
        iter_num += 1
        if iter_num % acc_steps == 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            model.zero_grad(set_to_none=True)
        
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss / world_size
        if rank == 0:
            print(f'iter {iter_num},  loss {loss.item()}')
            print(gpu_memory_usage())
        
        if iter_num % 125 == 0:
            os.makedirs(config.outputdir, exist_ok=True)
            save_checkpoint(rank, config.outputdir, iter_num, model, optimizer)
            goal_name = f"iter_{iter_num}.pt"
            if rank == 0:
                os.system(f'cp {os.path.join(config.outputdir, "latest.pt")} {os.path.join(config.outputdir, goal_name)}')
            dist.barrier()