"""Dataset for reranking."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class BinaryRerankDataset(Dataset):
    """Dataset for training and binary classification validation."""
    def __init__(self, hf_dataset, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        question = str(item["question"])
        answer = str(item["answer"])
        label = int(item["label"])

        encoding = self.tokenizer(
            question,
            answer,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }

class RankingRerankDataset(Dataset):
    """Dataset for full ranking validation."""
    def __init__(self, hf_dataset, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        question = str(item["question"])
        
        # 1 Positive + 19 Negatives
        docs = [str(item["answer"])] + [str(item[f"negative_{i}"]) for i in range(1, 20)]
        labels = [1] + [0] * 19  # 1 for positive, 0 for negatives

        queries = [question] * 20
        encodings = self.tokenizer(
            queries,
            docs,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],          # Shape: [20, max_length]
            "attention_mask": encodings["attention_mask"], # Shape: [20, max_length]
            "labels": torch.tensor(labels, dtype=torch.float32), # Shape: [20]
            "query_id": str(idx), # Unique ID required for ir_measures
        }



# class RerankDataset(Dataset):
#     """Dataset for document reranking."""
    
#     def __init__(
#         self,
#         data_path: str,
#         tokenizer: PreTrainedTokenizer,
#         max_length: int = 512,
#         query_max_length: int = 64,
#     ):
#         self.data_path = Path(data_path)
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.query_max_length = query_max_length
        
#         # Load data
#         self.examples = self.load_data()
    
#     def load_data(self) -> List[Dict]:
#         """Load data from file."""
#         examples = []
        
#         if self.data_path.suffix == ".jsonl":
#             with open(self.data_path, 'r') as f:
#                 for line in f:
#                     examples.append(json.loads(line))
#         elif self.data_path.suffix == ".json":
#             with open(self.data_path, 'r') as f:
#                 examples = json.load(f)
#         else:
#             raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
#         return examples
    
#     def __len__(self) -> int:
#         return len(self.examples)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         example = self.examples[idx]
        
#         query = example["query"]
#         document = example["doc"]
#         label = example.get("label", 1)  # Default to relevant
        
#         # Tokenize query + document
#         encoding = self.tokenizer(
#             query,
#             document,
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         )
        
#         return {
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0),
#             "labels": torch.tensor(label, dtype=torch.long),
#         }
