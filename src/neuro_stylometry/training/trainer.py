"""Phase D training loop (baseline vs constrained)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..data_engine.schemas import get_demographic_columns
from ..stylometry_net.classification_head import MultiTaskHead
from ..stylometry_net.phase_d_dataset import PhaseDCollator, PhaseDDataset, PhaseDLabelMaps
from ..stylometry_net.transformer import AffineGuardTransformer
from ..stylometry_net.tokenizer import PhaseDTokenizer


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
    ) -> tuple[DataLoader, PhaseDLabelMaps]:
        dataset = PhaseDDataset(
            self.config.dataset_path,
            text_field=text_field,
            label_fields=get_demographic_columns(),
            label_maps=label_maps,
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

    def _train(
        self,
        *,
        model: AffineGuardTransformer,
        head: MultiTaskHead,
        loader: DataLoader,
        run_dir: Path,
    ) -> None:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=self.config.learning_rate,
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        model.train()
        head.train()

        step = 0
        for epoch in range(self.config.num_epochs):
            for batch in loader:
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

                step += 1
                if self.config.max_steps and step >= self.config.max_steps:
                    break
            if self.config.max_steps and step >= self.config.max_steps:
                break

        torch.save(model.state_dict(), run_dir / "model.pt")
        torch.save(head.state_dict(), run_dir / "head.pt")

    def train_baseline_and_constrained(self, *, use_affine_guard: bool = True) -> None:
        baseline_loader, label_maps = self._build_loader(text_field="post")
        self.label_maps = label_maps
        baseline_model, baseline_head = self._build_model(use_affine_guard=False)

        self._train(
            model=baseline_model,
            head=baseline_head,
            loader=baseline_loader,
            run_dir=self.config.output_dir / "baseline",
        )
        if use_affine_guard:
            constrained_loader, _ = self._build_loader(
                text_field="post_masked",
                label_maps=label_maps,
            )
            constrained_model, constrained_head = self._build_model(use_affine_guard=True)
            self._train(
                model=constrained_model,
                head=constrained_head,
                loader=constrained_loader,
                run_dir=self.config.output_dir / "constrained",
            )
