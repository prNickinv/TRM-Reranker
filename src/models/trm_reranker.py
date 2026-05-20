import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from transformers import AutoModel, AutoTokenizer

from ..modules.reasoning_blocks import (
    CastedEmbedding,
    CastedLinear,
    ReasoningBlock,
    ReasoningBlockConfig,
    ReasoningModule,
    RotaryEmbedding,
)
from ..modules.utils import compute_lr, trunc_normal_init_
from ..utils.logging_utils import RankedLogger
from ..utils.metrics import compute_ranking_metrics, compute_binary_classification_metrics

import ir_measures
import time

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class TRMInnerCarry:
    """Inner state carried across recursion."""
    z_H: torch.Tensor  # High-level state (relevance representation)
    z_L: torch.Tensor  # Low-level state (reasoning representation)


@dataclass
class TRMCarry:
    """Carry structure for maintaining state across supervision steps."""
    inner_carry: TRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TRMReranker(LightningModule):
    """
    TRM Reranker: Tiny Recursive Model for document reranking.
    
    Recursively refines relevance predictions through multiple supervision steps.
    Uses a single tiny 2-layer transformer for parameter efficiency.
    """
    
    def __init__(
        self,
        # Model architecture
        encoder_name: str = "google-bert/bert-base-uncased",
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_expansion: float = 2.0,
        dropout: float = 0.1,

        # TRM parameters
        H_cycles: int = 3,
        L_cycles: int = 6,
        N_supervision: int = 16,
        N_supervision_val: int = 16,
        halt_exploration_prob: float = 0.1,

        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        lr_min_ratio: float = 0.0,
        max_length: int = 512,

        # Loss configuration
        loss_type: str = "bce",  # 'bce', 'margin', 'listnet'
        margin: float = 1.0,

        # Optimizer configuration
        optimizer_name: str = "adamw",  # 'adamw', 'adamatan2'

        # Compilation
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",

        # Embedding configuration
        use_pretrained_encoder: bool = True,
        vocab_size: int = 30522,  # BERT vocab size
        use_rope: bool = False,
        rope_base: float = 10000.0,

        # Other
        forward_dtype: Optional[torch.dtype] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Manual optimization
        self.automatic_optimization = False
        
        # Determine dtype
        if forward_dtype is not None:
            self.forward_dtype = forward_dtype
        else:
            if torch.cuda.is_available():
                self.forward_dtype = torch.bfloat16
            else:
                self.forward_dtype = torch.float32
        
        log.info(f"Using forward dtype: {self.forward_dtype}")

        # Initialize encoder or embeddings
        if use_pretrained_encoder:
            # Load pretrained encoder
            log.info(f"Loading pretrained encoder: {encoder_name}")
            self.encoder = AutoModel.from_pretrained(encoder_name)
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

            if freeze_encoder:
                log.info("Freezing encoder weights")
                for param in self.encoder.parameters():
                    param.requires_grad = False

            encoder_hidden_size = self.encoder.config.hidden_size

            # Projection from encoder to reasoning hidden size
            self.encoder_proj = CastedLinear(encoder_hidden_size, hidden_size, bias=False)
            self.embedding = None

            if use_rope:
                log.info(f"Using Rotary Embeddings with base={rope_base}")
                head_dim = hidden_size // num_heads
                self.rope = RotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=max_length,
                    base=rope_base,
                )
            else:
                self.rope = None

            #self.rope = None
        else:
            # Train embeddings from scratch
            log.info(f"Training embeddings from scratch with vocab_size={vocab_size}")
            self.encoder = None
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)  # Still need tokenizer

            embed_init_std = 1.0 / math.sqrt(hidden_size)
            
            # Trainable token embeddings
            self.embedding = CastedEmbedding(
                num_embeddings=vocab_size,
                embedding_dim=hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

            self.encoder_proj = None

            # Rotary embeddings if enabled
            if use_rope:
                log.info(f"Using Rotary Embeddings with base={rope_base}")
                head_dim = hidden_size // num_heads
                self.rope = RotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=max_length,
                    base=rope_base,
                )
            else:
                self.rope = None
        
        # TRM reasoning network (single network, not hierarchical)
        reasoning_config = ReasoningBlockConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=ffn_expansion,
            rms_norm_eps=1e-5,
            dropout=dropout,
            use_rope=use_rope if not use_pretrained_encoder else False,
        )
        
        self.reasoning_net = ReasoningModule(
            layers=nn.ModuleList([
                ReasoningBlock(reasoning_config) for _ in range(num_layers)
            ])
        )
        
        # Output heads
        self.relevance_head = CastedLinear(hidden_size, 1, bias=True)

        # self.relevance_head = nn.Sequential(
        #    nn.Dropout(dropout),
        #    CastedLinear(hidden_size, 1, bias=True),
        # )

        self.q_head = CastedLinear(hidden_size, 1, bias=True)
        
        # Initialize q_head with negative bias (prefer not halting early)
        with torch.no_grad():
            self.q_head.weight.zero_()
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-5.0)
        
        # Initial states for z_H and z_L
        # self.register_buffer(
        #     "z_H_init",
        #     trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1),
        # )
        # self.register_buffer(
        #     "z_L_init",
        #     trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1),
        # )

        self.register_buffer(
            "z_H_init",
            torch.zeros(hidden_size, dtype=self.forward_dtype),
        )
        self.register_buffer(
            "z_L_init",
            torch.zeros(hidden_size, dtype=self.forward_dtype),
        )
    
        # self.z_H_init = nn.Parameter(
        #     trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1),
        #     requires_grad=True,
        # )
        # self.z_L_init = nn.Parameter(
        #     trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1),
        #     requires_grad=True,
        # )
        
        # State for carry (persisted across training steps)
        self.carry = None
        self.manual_step = 0
        self.total_steps = 0
    

    def setup(self, stage: str):
        """Called by Lightning when setting up the model."""
        if stage == "fit":
            dm = self.trainer.datamodule
            steps_per_epoch = len(dm.train_dataloader())

            if self.trainer.max_epochs > 0:
                self.total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                self.total_steps = float("inf")

            log.info(f"Training configuration:")
            log.info(f"  Steps per epoch: {steps_per_epoch}")
            log.info(f"  Total steps: {self.total_steps}")

            # Compile inner_forward if enabled
            if self.hparams.use_compile:
                if hasattr(torch, 'compile'):
                    log.info(f"Compiling inner_forward with mode: {self.hparams.compile_mode}")
                    
                    # 1. Compile the original method
                    compiled_fn = torch.compile(
                        self.inner_forward,
                        mode=self.hparams.compile_mode,
                        fullgraph=False,
                    )
                    
                    # 2. Define a safe wrapper to fix CUDAGraphs memory overwrite error
                    def safe_inner_forward(carry, batch):
                        # Explicitly tell CUDAGraphs that a new recurrent step is beginning
                        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                            torch.compiler.cudagraph_mark_step_begin()
                            
                        # Execute the compiled graph
                        out_carry, rel_logits, q_logits = compiled_fn(carry, batch)
                        
                        # Clone all outputs OUTSIDE the graph to break memory aliasing.
                        # This frees CUDAGraphs to reuse its internal buffers safely.
                        safe_carry = TRMInnerCarry(
                            z_H=out_carry.z_H.clone(),
                            z_L=out_carry.z_L.clone()
                        )
                        return safe_carry, rel_logits.clone(), q_logits.clone()
                        
                    # 3. Override the method with our safe wrapper
                    self.inner_forward = safe_inner_forward
                else:
                    log.warning("torch.compile not available (requires PyTorch 2.0+). Skipping compilation.")
    
    # def setup(self, stage: str):
    #     """Called by Lightning when setting up the model."""
    #     if stage == "fit":
    #         dm = self.trainer.datamodule
    #         steps_per_epoch = len(dm.train_dataloader())

    #         if self.trainer.max_epochs > 0:
    #             self.total_steps = steps_per_epoch * self.trainer.max_epochs
    #         else:
    #             self.total_steps = float("inf")

    #         log.info(f"Training configuration:")
    #         log.info(f"  Steps per epoch: {steps_per_epoch}")
    #         log.info(f"  Total steps: {self.total_steps}")

    #         # Compile inner_forward if enabled
    #         if self.hparams.use_compile:
    #             if hasattr(torch, 'compile'):
    #                 log.info(f"Compiling inner_forward with mode: {self.hparams.compile_mode}")
    #                 self.inner_forward = torch.compile(
    #                     self.inner_forward,
    #                     mode=self.hparams.compile_mode,
    #                     fullgraph=False,
    #                 )
    #             else:
    #                 log.warning("torch.compile not available (requires PyTorch 2.0+). Skipping compilation.")
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text using pretrained encoder or trainable embeddings.

        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        """
        if self.hparams.use_pretrained_encoder:
            # Use pretrained encoder
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                embeddings = outputs.last_hidden_state

            # Project to reasoning hidden size
            embeddings = self.encoder_proj(embeddings)
        else:
            # Use trainable embeddings
            embeddings = self.embedding(input_ids)

        # Scale embeddings
        embed_scale = math.sqrt(self.hparams.hidden_size)
        embeddings = embed_scale * embeddings

        return embeddings.to(self.forward_dtype), attention_mask
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRMCarry:
        """Initialize carry for a new batch."""
        batch_size = batch["input_ids"].shape[0]
        device = batch["input_ids"].device
        
        return TRMCarry(
            inner_carry=self.empty_carry(batch_size, device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()},
        )
    
    def empty_carry(self, batch_size: int, device: torch.device) -> TRMInnerCarry:
        """Create empty inner carry."""
        seq_len = self.hparams.max_length  # Fixed sequence length
        
        return TRMInnerCarry(
            z_H=torch.empty(
                batch_size, seq_len, self.hparams.hidden_size,
                dtype=self.forward_dtype, device=device,
            ),
            z_L=torch.empty(
                batch_size, seq_len, self.hparams.hidden_size,
                dtype=self.forward_dtype, device=device,
            ),
        )
    
    def reset_carry(
        self, 
        reset_flag: torch.Tensor, 
        carry: TRMInnerCarry
    ) -> TRMInnerCarry:
        """Reset carry for halted sequences."""
        # Expand init states to batch and sequence length
        batch_size, seq_len, hidden_size = carry.z_H.shape
        
        z_H_init_expanded = self.z_H_init.view(1, 1, -1).expand(batch_size, seq_len, -1)
        z_L_init_expanded = self.z_L_init.view(1, 1, -1).expand(batch_size, seq_len, -1)
        
        return TRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), z_H_init_expanded, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), z_L_init_expanded, carry.z_L),
        )
    
    def inner_forward(
        self,
        carry: TRMInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TRMInnerCarry, torch.Tensor, torch.Tensor]:
        """
        Inner forward pass with recursive reasoning.
        
        Returns:
            new_carry: Updated carry
            logits: Relevance scores [batch_size]
            q_halt_logits: Halt decision logits [batch_size]
        """
        # Encode input
        input_embeddings, attention_mask = self.encode_text(
            batch["input_ids"],
            batch["attention_mask"],
        )
        
        # Convert attention mask to format for scaled_dot_product_attention
        # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(input_embeddings.dtype).min
        else:
            extended_attention_mask = None

        # Get RoPE embeddings if using trainable embeddings
        cos_sin = None
        if self.rope is not None:
            cos_sin = self.rope()

        # Recursive reasoning
        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles - 1 without grad
        with torch.no_grad():
            for _ in range(self.hparams.H_cycles - 1):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.reasoning_net(z_L, z_H + input_embeddings, extended_attention_mask, cos_sin)
                z_H = self.reasoning_net(z_H, z_L, extended_attention_mask, cos_sin)

        # 1 with grad
        for _ in range(self.hparams.L_cycles):
            z_L = self.reasoning_net(z_L, z_H + input_embeddings, extended_attention_mask, cos_sin)
        z_H = self.reasoning_net(z_H, z_L, extended_attention_mask, cos_sin)
        
        # Get [CLS] token representation
        cls_repr = z_H[:, 0, :]  # [batch_size, hidden_size]
        
        # Compute relevance score and halt decision
        relevance_logits = self.relevance_head(cls_repr).squeeze(-1).to(torch.float32)  # [batch_size]
        q_halt_logits = self.q_head(cls_repr).squeeze(-1).to(torch.float32)  # [batch_size]
        
        # New carry (detached)
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        
        return new_carry, relevance_logits, q_halt_logits
    
    def forward(
        self,
        carry: TRMCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        """Forward pass with supervision step management."""
        # Reset carry for halted sequences
        new_inner_carry = self.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        
        # Update current data
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }
        
        # Forward inner model
        new_inner_carry, relevance_logits, q_halt_logits = self.inner_forward(
            new_inner_carry, new_current_data
        )
        
        outputs = {
            "relevance_logits": relevance_logits,
            "q_halt_logits": q_halt_logits,
        }
        
        with torch.no_grad():
            # Increment steps
            new_steps = new_steps + 1
            n_supervision_steps = (
                self.hparams.N_supervision if self.training else self.hparams.N_supervision_val
            )
            
            is_last_step = new_steps >= n_supervision_steps
            halted = is_last_step
            
            # ACT: halt if q_halt_logits > 0
            if self.training and (self.hparams.N_supervision > 1):
                halted = halted | (q_halt_logits > 0)
                
                # Exploration: enforce minimum steps
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.hparams.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.hparams.N_supervision + 1)
                halted = halted & (new_steps >= min_halt_steps)
        
        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
    
    def compute_loss_and_metrics(self, carry, batch):
        """Compute loss and metrics."""
        new_carry, outputs = self.forward(carry, batch)
        labels = new_carry.current_data["labels"]
        
        # Compute loss based on loss type
        if self.hparams.loss_type == "bce":
            relevance_loss = F.binary_cross_entropy_with_logits(
                outputs["relevance_logits"],
                labels.to(outputs["q_halt_logits"].dtype),
                reduction="sum",
            )
        elif self.hparams.loss_type == "margin":
            # Margin ranking loss (for pairwise training)
            relevance_loss = torch.clamp(
                self.hparams.margin - outputs["relevance_logits"] * (2 * labels.float() - 1),
                min=0
            ).sum()
        else:
            raise ValueError(f"Unknown loss type: {self.hparams.loss_type}")
        
        # Compute correctness and classification metrics
        with torch.no_grad():
            predictions = (torch.sigmoid(outputs["relevance_logits"]) > 0.5).long()
            is_correct = predictions == labels

            # Metrics (only for halted sequences)
            valid_metrics = new_carry.halted

            # Compute binary classification metrics (precision, recall, F1)
            if valid_metrics.sum() > 0:
                halted_preds = outputs["relevance_logits"][valid_metrics]
                halted_labels = labels[valid_metrics]
                classification_metrics = compute_binary_classification_metrics(halted_preds, halted_labels)
            else:
                classification_metrics = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'tp': 0.0,
                    'fp': 0.0,
                    'tn': 0.0,
                    'fn': 0.0,
                }

            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, is_correct.float(), 0.0).sum(),
                "precision": classification_metrics['precision'] * valid_metrics.sum(),
                "recall": classification_metrics['recall'] * valid_metrics.sum(),
                "f1": classification_metrics['f1'] * valid_metrics.sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
        
        # Q-halt loss
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        
        # Aggregate losses (sum over valid samples)
        #relevance_loss = torch.where(valid_metrics, relevance_loss, 0.0).sum()
        #q_halt_loss = torch.where(valid_metrics, q_halt_loss, 0.0).sum()
        
        metrics.update({
            "relevance_loss": relevance_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        
        total_loss = relevance_loss + 0.5 * q_halt_loss
        
        return new_carry, total_loss, metrics, new_carry.halted.all()
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step with deep supervision."""
        batch_size = batch["input_ids"].shape[0]
        
        # Get optimizer
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        
        # Initialize carry if needed
        if self.carry is None:
            self.carry = self.initial_carry(batch)
        
        # Forward with loss computation
        self.carry, loss, metrics, _ = self.compute_loss_and_metrics(self.carry, batch)
        self.carry = TRMCarry(
            inner_carry=TRMInnerCarry(
                z_H=self.carry.inner_carry.z_H.clone(),
                z_L=self.carry.inner_carry.z_L.clone(),
            ),
            steps=self.carry.steps.clone(),
            halted=self.carry.halted.clone(),
            current_data={k: v.clone() for k, v in self.carry.current_data.items()},
        )
        # Backward
        scaled_loss = loss / batch_size
        scaled_loss.backward()
        #self.manual_backward(scaled_loss)
        
        # Gradient clipping
        #for opt in opts:
        #    self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Update learning rate and step
        #current_step = self.manual_step
        #base_lr = self.hparams.learning_rate
        
        #new_total_steps = self.trainer.estimated_stepping_batches
        #new_current_step = self.global_step
        #new_warmup_steps = int(new_total_steps * 0.015)
        #new_warmup_steps = self.hparams.warmup_steps
        
        lr_this_step = self.hparams.learning_rate

        # lr_this_step = compute_lr(
        #     base_lr=base_lr,
        #     lr_warmup_steps=new_warmup_steps,
        #     lr_min_ratio=self.hparams.lr_min_ratio,
        #     current_step=new_current_step,
        #     total_steps=new_total_steps,
        # )

        # lr_this_step = compute_lr(
        #     base_lr=base_lr,
        #     lr_warmup_steps=new_warmup_steps,
        #     lr_min_ratio=self.hparams.lr_min_ratio,
        #     current_step=new_current_step,
        #     total_steps=new_total_steps,
        # )
        
        for opt in opts:
            for param_group in opt.param_groups:
                param_group["lr"] = lr_this_step
            opt.step()
            opt.zero_grad()
        
        # Log metrics
        self.log_metrics(metrics, lr_this_step, batch_size)
        
        self.manual_step += 1
        
        return loss
    
    def on_validation_epoch_start(self):
        # Initialize a list to accumulate ranking scores
        self.val_ranking_outputs =[]
    
    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        """
        Handles both Binary Classification validation (0) and Ranking Validation (1).
        """
        if dataloader_idx == 0:
            self._validate_binary(batch, batch_idx)
        elif dataloader_idx == 1:
            self._validate_ranking(batch, batch_idx)
    
    def _validate_binary(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        batch_size = batch["input_ids"].shape[0]
        
        with torch.no_grad():
            carry_init = self.initial_carry(batch)
            carry = self.reset_carry(carry_init.halted, carry_init.inner_carry)

            # For Latency (start)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter() 
            
            for step in range(1, self.hparams.N_supervision_val + 1):
                carry, relevance_logits, q_halt_logits = self.inner_forward(carry, batch)
            
            # For Latency (stop)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Latency in milliseconds (ms) per sample
            latency_ms = ((end_time - start_time) * 1000.0) / batch_size

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
            
            # Log metrics
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
        """Inference for Ranking dataset per query"""
        B, num_docs, seq_len = batch["input_ids"].shape  # num_docs should be 20
        
        with torch.no_grad():
        
            # Flatten batch to process [B * 20, seq_len] efficiently
            flat_batch = {
                "input_ids": batch["input_ids"].view(-1, seq_len),
                "attention_mask": batch["attention_mask"].view(-1, seq_len),
            }
            
            flat_B = B * num_docs
            
            carry_init = self.initial_carry(flat_batch)
            carry = self.reset_carry(carry_init.halted, carry_init.inner_carry)
            
            #carry = self.empty_carry(flat_B, self.device)
            #final_logits = torch.zeros(flat_B, dtype=torch.float32, device=self.device)
            #already_halted = torch.zeros(flat_B, dtype=torch.bool, device=self.device)
            
            # Latency (start)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Run standard inference on the flattened batch
            for step in range(1, self.hparams.N_supervision_val + 1):
                carry, relevance_logits, q_halt_logits = self.inner_forward(carry, flat_batch)
                
                # is_last_step = step == self.hparams.N_supervision_val
                # just_halted = (q_halt_logits > 0) | is_last_step
                # newly_halted = just_halted & ~already_halted
                
                # if newly_halted.any():
                #     final_logits[newly_halted] = relevance_logits[newly_halted].float()
                    
                # already_halted = already_halted | newly_halted
                # if already_halted.all():
                #     break
            
            # Latency (stop)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = ((end_time - start_time) * 1000.0) / B
            self.log("val_ranking/latency_ms", latency_ms, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=B)

            # Reshape logits back to [B, 20]
            logits = relevance_logits.view(B, num_docs)
            
            # Apply sigmoid to normalize scores strictly for ranking compatibility
            scores = torch.sigmoid(logits)

            # Store outputs for ir_measures computation at epoch end
            for i in range(B):
                self.val_ranking_outputs.append({
                    "query_id": batch["query_id"][i],
                    "scores": scores[i].cpu().numpy(),
                    "labels": batch["labels"][i].cpu().numpy()
                })
    
    def on_validation_epoch_end(self):
        """Calculate ir_measures over all collected ranking evaluations"""
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

                    #qrels.append(ir_measures.Qrel(qid, doc_id, int(rel)))
                    run.append(ir_measures.ScoredDoc(qid, doc_id, float(score)))
            
            # Compute ranking metrics via ir_measures
            metrics = ir_measures.calc_aggregate(
                [
                    ir_measures.nDCG@10, 
                    ir_measures.MAP@10, 
                    ir_measures.Recall@10,
                    ir_measures.MRR@10,
                    ir_measures.Success@10,
                    ir_measures.nDCG@5, 
                    ir_measures.MAP@5, 
                    ir_measures.Recall@5,
                    ir_measures.MRR@5,
                    ir_measures.Success@5,
                ],
                qrels,
                run
            )
            
            # Log metrics cleanly to PyTorch Lightning
            for metric, value in metrics.items():
                metric_name = str(metric).replace("@", "_") # Clean "@" for WandB / Checkpointing
                self.log(f"val_ranking/{metric_name}", value, sync_dist=True, add_dataloader_idx=False)
            
            # Clear memory
            self.val_ranking_outputs.clear()
        
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step - same as validation."""
        return self.validation_step(batch, batch_idx)
    
    # def on_validation_epoch_start(self):
    #     """Don't interfere with training carry during validation."""
    #     pass

    # def on_validation_epoch_end(self):
    #     """Don't interfere with training carry during validation."""
    #     pass
    
    def log_metrics(self, metrics: dict, lr_this_step: float, batch_size: int):
        """Log training metrics."""
        self.log("train/lr", lr_this_step, on_step=True)

        if metrics.get("count", 0) > 0:
            with torch.no_grad():
                count = metrics["count"]
                self.log("train/accuracy", metrics.get("accuracy", 0) / count, on_step=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
                self.log("train/precision", metrics.get("precision", 0) / count, on_step=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
                self.log("train/recall", metrics.get("recall", 0) / count, on_step=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
                self.log("train/f1", metrics.get("f1", 0) / count, on_step=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
                self.log("train/q_halt_accuracy", metrics.get("q_halt_accuracy", 0) / count, on_step=True, sync_dist=True, batch_size=batch_size)
                self.log("train/steps", metrics.get("steps", 0) / count, on_step=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
                self.log("train/relevance_loss", metrics.get("relevance_loss", 0) / batch_size, on_step=True, sync_dist=True, batch_size=batch_size)
                self.log("train/q_halt_loss", metrics.get("q_halt_loss", 0) / batch_size, on_step=True, sync_dist=True, batch_size=batch_size)
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer_name = self.hparams.optimizer_name.lower()

        decay_params =[]
        no_decay_params =[]
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            # Biases and 1D parameters (like RMSNorm weights) should NOT have weight decay
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        # 2. Create the optimizer parameter groups
        optim_groups =[
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.95),
            )
            log.info("Using AdamW optimizer")

        elif optimizer_name == "adamatan2":
            try:
                #from adam_atan2 import AdamATan2
                from adam_atan2_pytorch import AdamAtan2
                optimizer = AdamAtan2(
                    optim_groups,
                    lr=self.hparams.learning_rate,
                    betas=(0.9, 0.95),
                )
                log.info("Using AdamATan2 optimizer")
            except ImportError:
                log.warning(
                    "AdamATan2 not available."
                    "Falling back to AdamW."
                )
                optimizer = torch.optim.AdamW(
                    optim_groups,
                    lr=self.hparams.learning_rate,
                    betas=(0.9, 0.95),
                )

        else:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. "
                f"Supported: 'adamw', 'adamatan2'"
            )

        return optimizer
