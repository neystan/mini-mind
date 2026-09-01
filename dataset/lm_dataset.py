from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    # init
    def __init__(self, data_path, tokenizer, max_length = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length    # 输入给gpu的最大长度

        self.samples = load_dataset("json", data_files = data_path, split = "train")
    # __len__
    def __len__(self):
        return len(self.samples)
    # __getitem__
    def __getitem__(self, index):
        # 我们拿到的是，jsonl里的每一行
        sample = self.samples[index]

        # tokenizer 把文本转化为input_id
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens = False,
            max_length = self.max_length - 2,
            truncation = True,          # 长度超过max，自动剪切
        ).input_ids

        # 需要加上EOS, BOS 以及 PAD填充
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype = torch.long)

        # 需要自行编写 labels， 防止 PAD 参与 loss 计算
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100    # -100 是一个特殊的值，表示该位置不参与 loss 计算

        # 需要编写attention_mask, 告诉模型哪些位置是有效的，哪些位置是PAD
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()      #非 PAD位置为1， PAD位置为0

        # 需要输出的 是 input_ids, attention_mask, labels
        return{
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "labels" : labels
        }




