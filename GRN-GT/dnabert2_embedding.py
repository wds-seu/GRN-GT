import tokenizers
from modeling_roformer import *
from fetch_gene_sequence import fetch_gene_sequence
import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoModel


def load_gene_model(device):
    tokenizer = tokenizers.Tokenizer.from_file("./tokenizer.json")
    config = BertConfig.from_pretrained("../DNABERT-2")
    gene_emb_model = AutoModel.from_pretrained("../DNABERT-2", trust_remote_code=True, config=config).to(device)
    return tokenizer, gene_emb_model, 768


def gene_embedding(gene_name, tokenizer, gene_emb_model, gene_cache, device):
    gene_sequence = fetch_gene_sequence(gene_name, gene_cache)
    if gene_sequence is None:  # 没有找到DNA序列
        return torch.zeros(1, 768).to(device)
    seq = torch.tensor(tokenizer.encode(gene_sequence, add_special_tokens=False).ids, dtype=torch.long)
    max_seq_len = 512
    if seq.shape[0] + 2 >= max_seq_len:
        seq = seq[0:max_seq_len - 2]
    seq_len = seq.shape[0]
    cls_id, sep_id, pad_id = 1, 2, 3
    input_ids = pad_id * torch.ones(max_seq_len, dtype=torch.long)
    input_ids[0] = cls_id
    input_ids[1:1 + seq_len] = seq
    input_ids[1 + seq_len] = sep_id
    attention_mask = torch.zeros(max_seq_len, dtype=torch.bool)
    attention_mask[:2 + seq_len] = True
    pos_ids = (max_seq_len - 1) * torch.ones(max_seq_len, dtype=torch.long)
    for i in range(0, 2 + seq_len):
        pos_ids[i] = i
    input_ids = input_ids.unsqueeze(0).to(device)
    pos_ids = pos_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    gene_emb = gene_emb_model(input_ids, position_ids=pos_ids, attention_mask=attention_mask)[1]
    return gene_emb
