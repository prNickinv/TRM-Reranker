"""Baseline Reranker Model using standard Hugging Face AutoModel."""

import os
from typing import Dict, Optional
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from transformers import AutoModel, get_cosine_schedule_with_warmup

from ..utils.logging_utils import RankedLogger
from ..utils.metrics import compute_binary_classification_metrics
import ir_measures
import time

log = RankedLogger(__name__, rank_zero_only=True)


class BaselineReranker(LightningModule):
    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        freeze_encoder: bool = True,
        forward_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load HuggingFace Base Model (Encoder only)
        log.info(f"Loading pretrained base model: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze the encoder if specified
        if freeze_encoder:
            log.info("Freezing base encoder. Only the relevance head will be trained.")
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Define the custom relevance head
        hidden_size = self.encoder.config.hidden_size
        self.relevance_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting the CLS token and projecting to a relevance score."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Assuming the first token [CLS] / <s> is the sequence representation
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.relevance_head(cls_repr).squeeze(-1)
        return logits

    def training_step(self, batch: dict, batch_idx: int):
        logits = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        labels = batch["labels"].float()
        
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        with torch.no_grad():
            metrics = compute_binary_classification_metrics(logits, labels)
            self.log("train/accuracy", metrics["accuracy"], on_step=True, prog_bar=True, sync_dist=True)
            self.log("train/f1", metrics["f1"], on_step=True, prog_bar=True, sync_dist=True)
            self.log("train/precision", metrics["precision"], on_step=True, sync_dist=True)
            self.log("train/recall", metrics["recall"], on_step=True, sync_dist=True)
            
        return loss

    def on_validation_epoch_start(self):
        self.val_ranking_outputs = []

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        """Handles both Binary Classification (0) and Ranking (1)."""
        if dataloader_idx == 0:
            self._validate_binary(batch, batch_idx)
        elif dataloader_idx == 1:
            self._validate_ranking(batch, batch_idx)

    def _validate_binary(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch_size = batch["input_ids"].shape[0]
        
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter() 
            
            relevance_logits = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = ((end_time - start_time) * 1000.0) / batch_size

            labels = batch['labels'].float()
            relevance_loss = F.binary_cross_entropy_with_logits(relevance_logits, labels, reduction='mean')
            classification_metrics = compute_binary_classification_metrics(relevance_logits, labels)
            
            avg_metrics = {
                "val/rel_loss": relevance_loss,
                "val/accuracy": classification_metrics.get("accuracy", 0),
                "val/precision": classification_metrics.get("precision", 0),
                "val/recall": classification_metrics.get("recall", 0),
                "val/f1": classification_metrics.get("f1", 0),
                "val/latency_ms": latency_ms,
            }
            
            for name, value in avg_metrics.items():
                self.log(
                    name, value,
                    on_step=False, on_epoch=True,
                    prog_bar=(name in ["val/rel_loss", "val/accuracy"]),
                    sync_dist=True,
                    add_dataloader_idx=False,
                    batch_size=batch_size
                )
            
            return avg_metrics

    def _validate_ranking(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        B, num_docs, seq_len = batch["input_ids"].shape
        
        with torch.no_grad():
            flat_input_ids = batch["input_ids"].view(-1, seq_len)
            flat_attention_mask = batch["attention_mask"].view(-1, seq_len)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Forward pass on flattened batch
            flat_logits = self(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = ((end_time - start_time) * 1000.0) / B
            self.log("val_ranking/latency_ms", latency_ms, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=B)

            # Reshape back to [B, 20]
            logits = flat_logits.view(B, num_docs)
            scores = torch.sigmoid(logits)

            for i in range(B):
                self.val_ranking_outputs.append({
                    "query_id": batch["query_id"][i],
                    "scores": scores[i].cpu().numpy(),
                    "labels": batch["labels"][i].cpu().numpy()
                })

    def on_validation_epoch_end(self):
        if hasattr(self, "val_ranking_outputs") and len(self.val_ranking_outputs) > 0:
            qrels = []
            run = []
            
            for res in self.val_ranking_outputs:
                qid = str(res["query_id"])
                for doc_idx, (score, rel) in enumerate(zip(res["scores"], res["labels"])):
                    #doc_id = f"doc_{doc_idx}"
                    doc_id = f"doc_{qid}_{doc_idx}"

                    if int(rel) > 0:
                        qrels.append(ir_measures.Qrel(qid, doc_id, int(rel)))
                    run.append(ir_measures.ScoredDoc(qid, doc_id, float(score)))
            
            metrics = ir_measures.calc_aggregate([
                    ir_measures.nDCG@10, ir_measures.MAP@10, ir_measures.Recall@10,
                    ir_measures.MRR@10, ir_measures.Success@10,
                    ir_measures.nDCG@5, ir_measures.MAP@5, ir_measures.Recall@5,
                    ir_measures.MRR@5, ir_measures.Success@5,
                ],
                qrels, run
            )
            
            # Log metrics to PyTorch Lightning / TensorBoard / WandB
            for metric, value in metrics.items():
                metric_name = str(metric).replace("@", "_")
                self.log(f"val_ranking/{metric_name}", value, sync_dist=True, add_dataloader_idx=False)
            
            # Save to CSV
            df = pd.DataFrame([metrics])
            save_path = os.path.join(self.trainer.default_root_dir, f"custom_metrics_baseline_epoch_{self.current_epoch}.csv")
            df.to_csv(save_path, index=False)

            self.val_ranking_outputs.clear()

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        """Map testing behavior directly to validation behavior."""
        return self.validation_step(batch, batch_idx, dataloader_idx)
        
    def on_test_epoch_start(self):
        self.on_validation_epoch_start()
        
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        """Standard AdamW setup separating biases from weight decay."""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.learning_rate)
        
        # Add a learning rate scheduler for stable fine-tuning
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }