"""Lightning DataModule for reranking."""

from pathlib import Path
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import load_dataset

from .rerank_dataset import BinaryRerankDataset, RankingRerankDataset
from ..utils.logging_utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class RerankDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 32,
        ranking_batch_size: int = 4,
        num_candidates: int = 20,
        num_workers: int = 4,
        train_file: str = "train.parquet",
        val_file: str = "val.parquet",
        val_ranking_file: str = "val_ranking.parquet",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.ranking_batch_size = ranking_batch_size
        self.num_candidates = num_candidates
        self.num_workers = num_workers
        
        self.train_file = train_file
        self.val_file = val_file
        self.val_ranking_file = val_ranking_file
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.val_ranking_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        train_path = self.data_dir / self.train_file
        val_path = self.data_dir / self.val_file
        val_ranking_path = self.data_dir / self.val_ranking_file

        if stage == "fit" or stage is None or stage == "test":
            if train_path.exists():
                hf_train = load_dataset("parquet", data_files=str(train_path), split="train")
                self.train_dataset = BinaryRerankDataset(hf_train, self.tokenizer, self.max_length)
                log.info(f"Loaded {len(self.train_dataset)} training examples")

            if val_path.exists():
                hf_val = load_dataset("parquet", data_files=str(val_path), split="train")
                self.val_dataset = BinaryRerankDataset(hf_val, self.tokenizer, self.max_length)
                log.info(f"Loaded {len(self.val_dataset)} binary validation examples")
                
            if val_ranking_path.exists():
                hf_val_rank = load_dataset("parquet", data_files=str(val_ranking_path), split="train")
                self.val_ranking_dataset = RankingRerankDataset(hf_val_rank, self.tokenizer, self.max_length, self.num_candidates)
                log.info(f"Loaded {len(self.val_ranking_dataset)} ranking validation queries")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True, 
            drop_last=True
        )

    def val_dataloader(self):
        dataloaders =[]
        if self.val_dataset is not None:
            dataloaders.append(DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.num_workers, 
                pin_memory=True
            ))
        if self.val_ranking_dataset is not None:
            dataloaders.append(DataLoader(
                self.val_ranking_dataset, 
                batch_size=self.ranking_batch_size, 
                shuffle=False,
                num_workers=self.num_workers, 
                pin_memory=True
            ))
            
        return dataloaders[0] if len(dataloaders) == 1 else dataloaders

    def test_dataloader(self):
        dataloaders =[]
        if self.val_dataset is not None:
            dataloaders.append(DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.num_workers, 
                pin_memory=True
            ))
        if self.val_ranking_dataset is not None:
            dataloaders.append(DataLoader(
                self.val_ranking_dataset, 
                batch_size=self.ranking_batch_size, 
                shuffle=False,
                num_workers=self.num_workers, 
                pin_memory=True
            ))
            
        return dataloaders[0] if len(dataloaders) == 1 else dataloaders



# class RerankDataModule(LightningDataModule):
#     """DataModule for reranking tasks."""
    
#     def __init__(
#         self,
#         data_dir: str,
#         tokenizer_name: str = "bert-base-uncased",
#         max_length: int = 512,
#         batch_size: int = 32,
#         num_workers: int = 4,
#         train_file: str = "train.jsonl",
#         val_file: str = "val.jsonl",
#         test_file: str = "test.jsonl",
#     ):
#         super().__init__()
#         self.data_dir = Path(data_dir)
#         self.tokenizer_name = tokenizer_name
#         self.max_length = max_length
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.train_file = train_file
#         self.val_file = val_file
#         self.test_file = test_file
        
#         # Will be initialized in setup
#         self.tokenizer = None
#         self.train_dataset = None
#         self.val_dataset = None
#         self.test_dataset = None
    
#     def setup(self, stage: Optional[str] = None):
#         """Setup datasets."""
#         if self.tokenizer is None:
#             self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
#         if stage == "fit" or stage is None:
#             train_path = self.data_dir / self.train_file
#             val_path = self.data_dir / self.val_file
            
#             if train_path.exists():
#                 self.train_dataset = RerankDataset(
#                     data_path=str(train_path),
#                     tokenizer=self.tokenizer,
#                     max_length=self.max_length,
#                 )
#                 log.info(f"Loaded {len(self.train_dataset)} training examples")
            
#             if val_path.exists():
#                 self.val_dataset = RerankDataset(
#                     data_path=str(val_path),
#                     tokenizer=self.tokenizer,
#                     max_length=self.max_length,
#                 )
#                 log.info(f"Loaded {len(self.val_dataset)} validation examples")
        
#         if stage == "test":
#             test_path = self.data_dir / self.test_file
#             if test_path.exists():
#                 self.test_dataset = RerankDataset(
#                     data_path=str(test_path),
#                     tokenizer=self.tokenizer,
#                     max_length=self.max_length,
#                 )
#                 log.info(f"Loaded {len(self.test_dataset)} test examples")
    
#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             drop_last=True,
#         )
    
#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True,
#         )
    
#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True,
#         )
