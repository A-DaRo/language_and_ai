"""Phase D training loop (baseline vs constrained)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data_engine.schemas import get_demographic_columns
from ..stylometry_net.classification_head import MultiTaskHead
from ..stylometry_net.phase_d_dataset import (
    PhaseDCollator,
    PhaseDDataset,
    PhaseDLabelMaps,
    load_label_maps,
)
from ..stylometry_net.transformer import AffineGuardTransformer
from ..stylometry_net.tokenizer import PhaseDTokenizer
from .checkpointing import config_to_metadata, save_checkpoint, save_metadata
from .metrics import compute_task_metrics


@dataclass
class PhaseDTrainConfig:
    dataset_path: Path
    artifacts_dir: Path
    output_dir: Path
    model_name: str = "roberta-base"
    taxonomy_path: Path = Path("conf/base/gliner_taxonomy.yaml")
    max_length: int = 512
    batch_size: int = 8
    num_epochs: int = 1
    max_steps: Optional[int] = None
    learning_rate: float = 2e-5
    save_every_steps: Optional[int] = None
    save_every_epochs: Optional[int] = None
    split_ratios: Dict[str, float] = None


class PhaseDTrainer:
    """Train baseline vs constrained models on Phase A outputs."""

    def __init__(self, config: PhaseDTrainConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_loader(
        self,
        *,
        text_field: str,
        label_maps: Optional[PhaseDLabelMaps] = None,
        split: Optional[str] = None,
    ) -> tuple[DataLoader, PhaseDLabelMaps]:
        dataset = PhaseDDataset(
            self.config.dataset_path,
            text_field=text_field,
            label_fields=get_demographic_columns(),
            label_maps=label_maps,
            split=split,
            split_ratios=self.config.split_ratios,
        )
        collator = PhaseDCollator(
            PhaseDTokenizer(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                taxonomy_path=self.config.taxonomy_path,
            )
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        return loader, dataset.label_maps

    def _build_model(self, *, use_affine_guard: bool) -> tuple[AffineGuardTransformer, MultiTaskHead]:
        if use_affine_guard:
            model = AffineGuardTransformer.from_phase_a(
                self.config.artifacts_dir,
                model_name=self.config.model_name,
                taxonomy_path=self.config.taxonomy_path,
                max_length=self.config.max_length,
            )
        else:
            model = AffineGuardTransformer(
                model_name=self.config.model_name,
                projection_matrix_path=None,
                taxonomy_path=self.config.taxonomy_path,
                max_length=self.config.max_length,
            )

        model = model.to(self.device)
        head = MultiTaskHead(
            hidden_dim=model.config.hidden_size,
            num_labels_per_task=self.label_maps.num_classes(),
        ).to(self.device)
        return model, head

    def _log_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    @torch.no_grad()
    def _evaluate(
        self,
        *,
        model: AffineGuardTransformer,
        head: MultiTaskHead,
        loader: DataLoader,
        num_classes: Dict[str, int],
    ) -> Dict[str, Dict[str, float]]:
        model.eval()
        head.eval()
        metrics: Dict[str, Dict[str, float]] = {}

        task_logits: Dict[str, list[torch.Tensor]] = {k: [] for k in num_classes}
        task_labels: Dict[str, list[torch.Tensor]] = {k: [] for k in num_classes}

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch["labels"].items()}

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = head(outputs["cls_embedding"])

            for task, task_logits_batch in logits.items():
                task_logits[task].append(task_logits_batch.detach().cpu())
                task_labels[task].append(labels[task].detach().cpu())

        for task in num_classes:
            if not task_logits[task]:
                continue
            logits_cat = torch.cat(task_logits[task], dim=0)
            labels_cat = torch.cat(task_labels[task], dim=0)
            metrics[task] = compute_task_metrics(
                logits_cat,
                labels_cat,
                num_classes=num_classes[task],
            )

        model.train()
        head.train()
        return metrics

    def _train(
        self,
        *,
        model: AffineGuardTransformer,
        head: MultiTaskHead,
        loader: DataLoader,
        eval_loader: Optional[DataLoader],
        run_dir: Path,
    ) -> None:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=self.config.learning_rate,
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = run_dir / "checkpoints"
        log_path = run_dir / "training_log.jsonl"
        metrics_path = run_dir / "phase_d_metrics.json"
        checkpoint_path = run_dir / "checkpoint.pt"
        model.train()
        head.train()

        step = 0
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.config.num_epochs):
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
            for batch in progress:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch["labels"].items()}

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = head(outputs["cls_embedding"])
                loss = head.compute_loss(logits, labels)

                if loss is None:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_metrics = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": float(loss.item()),
                }
                self._log_jsonl(log_path, step_metrics)
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

                step += 1
                if self.config.save_every_steps and step % self.config.save_every_steps == 0:
                    save_checkpoint(
                        checkpoint_dir / f"checkpoint_step_{step}.pt",
                        model_state=model.state_dict(),
                        head_state=head.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        metadata={
                            "config": config_to_metadata(self.config),
                            "label_maps": self.label_maps.maps,
                            "checkpoint": {"epoch": epoch + 1, "step": step},
                        },
                    )
                if self.config.max_steps and step >= self.config.max_steps:
                    break
            if self.config.save_every_epochs and (epoch + 1) % self.config.save_every_epochs == 0:
                save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    model_state=model.state_dict(),
                    head_state=head.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    metadata={
                        "config": config_to_metadata(self.config),
                        "label_maps": self.label_maps.maps,
                        "checkpoint": {"epoch": epoch + 1, "step": step},
                    },
                )
            if self.config.max_steps and step >= self.config.max_steps:
                break

        torch.save(model.state_dict(), run_dir / "model.pt")
        torch.save(head.state_dict(), run_dir / "head.pt")
        metrics = self._evaluate(
            model=model,
            head=head,
            loader=eval_loader or loader,
            num_classes=self.label_maps.num_classes(),
        )
        save_metadata(metrics_path, metrics)
        save_checkpoint(
            checkpoint_path,
            model_state=model.state_dict(),
            head_state=head.state_dict(),
            optimizer_state=optimizer.state_dict(),
            metadata={
                "config": config_to_metadata(self.config),
                "label_maps": self.label_maps.maps,
                "metrics": metrics,
            },
        )

    def train_baseline_and_constrained(self, *, use_affine_guard: bool = True) -> None:
        label_maps = load_label_maps(self.config.dataset_path, get_demographic_columns())
        self.label_maps = label_maps
        baseline_loader, _ = self._build_loader(
            text_field="post",
            label_maps=label_maps,
            split="train",
        )
        baseline_eval_loader, _ = self._build_loader(
            text_field="post",
            label_maps=label_maps,
            split="val",
        )
        baseline_model, baseline_head = self._build_model(use_affine_guard=False)

        self._train(
            model=baseline_model,
            head=baseline_head,
            loader=baseline_loader,
            eval_loader=baseline_eval_loader,
            run_dir=self.config.output_dir / "baseline",
        )
        if use_affine_guard:
            constrained_loader, _ = self._build_loader(
                text_field="post_masked",
                label_maps=label_maps,
                split="train",
            )
            constrained_eval_loader, _ = self._build_loader(
                text_field="post_masked",
                label_maps=label_maps,
                split="val",
            )
            constrained_model, constrained_head = self._build_model(use_affine_guard=True)
            self._train(
                model=constrained_model,
                head=constrained_head,
                loader=constrained_loader,
                eval_loader=constrained_eval_loader,
                run_dir=self.config.output_dir / "constrained",
            )
