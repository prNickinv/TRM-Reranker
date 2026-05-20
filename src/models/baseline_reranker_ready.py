"""Baseline Reranker Model using standard Hugging Face AutoModelForSequenceClassification (already trained)."""

from typing import Dict, Optional
import torch
import torch.nn.functional as F
from lightning import LightningModule
from transformers import AutoModelForSequenceClassification

from ..utils.logging_utils import RankedLogger
from ..utils.metrics import compute_binary_classification_metrics
import ir_measures
import pandas as pd
import os
import time

log = RankedLogger(__name__, rank_zero_only=True)


class BaselineRerankerReady(LightningModule):
    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        forward_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load HuggingFace Sequence Classification Model
        log.info(f"Loading pretrained baseline model: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def on_test_start(self):
        self.val_ranking_outputs =[]

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
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
            
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Latency in milliseconds (ms) per sample
            latency_ms = ((end_time - start_time) * 1000.0) / batch_size
            
            # HF outputs are [batch_size, 1] for num_labels=1. Squeeze to[batch_size]
            relevance_logits = outputs.logits.squeeze(-1)
            labels = batch['labels']
            
            relevance_loss = F.binary_cross_entropy_with_logits(
                relevance_logits,
                labels.to(relevance_logits.dtype),
                reduction='mean'
            )
            
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
                    prog_bar=(name in["val/rel_loss", "val/accuracy"]),
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
            
            outputs = self.model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = ((end_time - start_time) * 1000.0) / B
            self.log("val_ranking/latency_ms", latency_ms, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=B)
            
            # Reshape back to [B, 20]
            logits = outputs.logits.view(B, num_docs)
            scores = torch.sigmoid(logits)

            for i in range(B):
                self.val_ranking_outputs.append({
                    "query_id": batch["query_id"][i],
                    "scores": scores[i].cpu().numpy(),
                    "labels": batch["labels"][i].cpu().numpy()
                })

    def on_test_end(self):
        if hasattr(self, "val_ranking_outputs") and len(self.val_ranking_outputs) > 0:
            qrels = []
            run =[]
            
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

            df = pd.DataFrame([metrics])
            save_path = os.path.join(self.trainer.default_root_dir, "custom_metrics_baseline.csv")
            df.to_csv(save_path, index=False)
            
            #for metric, value in metrics.items():
            #    metric_name = str(metric).replace("@", "_")
            #    self.log(f"val_ranking/{metric_name}", value, sync_dist=True, add_dataloader_idx=False)
            

            self.val_ranking_outputs.clear()

            #if self.trainer.logger:
            #    self.trainer.logger.save()