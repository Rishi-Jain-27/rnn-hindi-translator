# Training loop: AMP, grad accumulation, teacher forcing, early stopping.

import torch
from .optim import build_optimizer
from .schedule import lr_at
from .loss import label_smoothed_nll
from .ema import EMA, SWA
from .checkpoint import CheckpointManager
from .tracking import Tracker, Throughput
from dataclasses import asdict # turns config into a plain dict

class Trainer:
    def __init__(self, cfg, model, train_loader, dev_loader, tokenizer, out_dir):
        # store the inputs
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.tokenizer = tokenizer
        self.out_dir = out_dir
        # find where to move
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # self.amp is a boolean on if Automatic Mixed Precision is enabled
        self.amp = self.cfg.amp and self.device == "cuda"
        # amp settings
        if self.cfg.amp_dtype == "bf16":
            self.amp_dtype = torch.bfloat16
        elif self.cfg.amp_dtype == "fp16":
            self.amp_dtype = torch.float16
        elif self.cfg.amp_dtype == "auto":
            self.amp_dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        # grad scaler always created but only enabled for fp16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp and self.amp_dtype == torch.float16))
        self.optimizer = build_optimizer(model, cfg)
        self.pad_id = tokenizer.pad_id
        # never ending stream of batches
        self.batches = self.infinite(train_loader)
        self.ema = EMA(self.model, self.cfg.ema_decay) if self.cfg.ema_decay else None
        self.swa = SWA(self.model) if self.cfg.use_swa else None
        self.ckpt = CheckpointManager(self.out_dir, self.cfg.ckpt_keep_last_n)
        self.tracker = Tracker(self.cfg, self.out_dir, asdict(self.cfg))
        self.throughput = Throughput()
        self.step, self.best = self.ckpt.resume(self.model, optimizer=self.optimizer, scaler=self.scaler, ema=self.ema, swa=self.swa, map_location=self.device)
        self.no_improve = 0

    def train(self):
        self.model.train()
        while self.step < self.cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            window_tokens = 0
            window_nll = 0

            for i in range(self.cfg.grad_accum):
                # start of a window — because weights update once in here
                batch = next(self.batches)

                # get data from the batch
                src_ids = batch["src_ids"].to(self.device)
                src_pad_mask = batch["src_pad_mask"].to(self.device)
                tgt_in = batch["tgt_in"].to(self.device)
                tgt_pad_mask = batch["tgt_pad_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # run every op in 16 bit when its safe
                with torch.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
                    logits = self.model(src_ids, tgt_in, src_pad_mask, tgt_pad_mask)
                    loss_sum, nll_sum, n_tokens = label_smoothed_nll(logits, labels, self.pad_id, self.cfg.label_smoothing)
                
                # Backprop - if scaler is enabled, multiply loss by a large factor so theyre in real units, then divide
                self.scaler.scale(loss_sum).backward()

                # Update totals
                window_tokens += int(n_tokens)
                window_nll += nll_sum.item()
            
            # add window tokens to running count, read rate when log
            self.throughput.update(window_tokens)
            
            # divide gradients back down by the large factor from scaler
            self.scaler.unscale_(self.optimizer)
            
            # division for gradient accumulation
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param.grad.div_(window_tokens)

            # gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            # get LR
            lr = lr_at(self.step + 1, self.cfg)
            # Write LR into every group
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            
            # gradient descent + check for overflow and skip if any
            self.scaler.step(self.optimizer)
            # adjust scale factor for next time
            self.scaler.update()

            # step
            self.step += 1

            if self.ema is not None: self.ema.update(self.model)
            if self.swa is not None and self.step >= self.cfg.swa_start and self.step % self.cfg.swa_period == 0: self.swa.update(self.model)
            
            # track
            if self.step % self.cfg.log_every == 0:
                self.tracker.log({
                    "train/loss": window_nll / window_tokens,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm.item(),
                    "train/tok_per_s": self.throughput.rate(),
                }, self.step)
            
            # checkpoint
            if self.step % self.cfg.ckpt_every == 0:
                self.ckpt.save(self.step, self.model, self.optimizer,
                            scaler=self.scaler, ema=self.ema, swa=self.swa,
                            best=self.best, config=asdict(self.cfg))
            
            # eval, check if best, ckpt
            if self.step % self.cfg.eval_every == 0:
                dev_nll = self.evaluate()
                self.tracker.log({"val/nll": dev_nll}, self.step)
                is_best = self.best is None or dev_nll < self.best
                if is_best:
                    self.best = dev_nll
                    self.no_improve = 0
                    self.ckpt.save(self.step, self.model, self.optimizer, scaler=self.scaler, ema=self.ema, swa=self.swa, best=self.best, config=asdict(self.cfg), is_best=True)
                else:
                    self.no_improve += 1
                    if self.no_improve >= self.cfg.patience:
                        break # early stopping - evals aren't improving
        self.tracker.close()

    def evaluate(self):
        # Swap in EMA weights if its on. Eval on dev set
        if self.ema:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)
        self.model.eval()
        total_nll = 0.0
        total_tokens = 0

        with torch.inference_mode():
            for batch in self.dev_loader:
                src_ids = batch["src_ids"].to(self.device)
                src_pad_mask = batch["src_pad_mask"].to(self.device)
                tgt_in = batch["tgt_in"].to(self.device)
                tgt_pad_mask = batch["tgt_pad_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward + loss
                with torch.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
                    logits = self.model(src_ids, tgt_in, src_pad_mask, tgt_pad_mask)
                    loss_sum, nll_sum, n_tokens = label_smoothed_nll(logits, labels, self.pad_id, self.cfg.label_smoothing)
                
                total_nll += nll_sum.item()
                total_tokens += int(n_tokens)

        # Swap training weights back if ema is on
        if self.ema is not None: self.ema.restore(self.model)

        # Back to training
        self.model.train()

        # return
        return total_nll / total_tokens


    def infinite(self, loader):
        while True:
            for batch in loader:
                yield batch

