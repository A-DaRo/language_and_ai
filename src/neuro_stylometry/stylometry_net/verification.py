"""Phase D verification: CHG + SVS runners."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from ..data_engine.schemas import get_demographic_columns
from .chg_verifier import CHGVerifier
from .classification_head import MultiTaskHead, SingleTaskHead
from .phase_d_dataset import PhaseDCollator, PhaseDDataset, load_label_maps
from .svs_calculator import SVSCalculator, SVSResult
from .tokenizer import PhaseDTokenizer
from .transformer import AffineGuardTransformer

logger = logging.getLogger(__name__)


def _load_model_and_head(
    *,
    run_dir: Path,
    artifacts_dir: Path,
    model_name: str,
    taxonomy_path: Path,
    max_length: int,
    use_affine_guard: bool,
    label_maps: Dict[str, Dict],
    device: torch.device,
    use_only_labels: Optional[List[str]] = None,
) -> Tuple[AffineGuardTransformer, Union[MultiTaskHead, SingleTaskHead]]:
    if use_affine_guard:
        model = AffineGuardTransformer.from_phase_a(
            artifacts_dir,
            model_name=model_name,
            taxonomy_path=taxonomy_path,
            max_length=max_length,
        )
    else:
        model = AffineGuardTransformer(
            model_name=model_name,
            projection_matrix_path=None,
            taxonomy_path=taxonomy_path,
            max_length=max_length,
        )

    # Load head state first to infer num_labels from checkpoint
    head_state = torch.load(run_dir / "head.pt", map_location=device)
    
    # Check if this is a SingleTaskHead or MultiTaskHead based on checkpoint keys
    # SingleTaskHead has "classifier.weight", MultiTaskHead has "heads.{task}.weight"
    is_single_task = "classifier.weight" in head_state
    
    if is_single_task:
        # SingleTaskHead: extract num_classes and task_name
        num_classes = head_state["classifier.weight"].shape[0]
        
        # Try to get task name from training_metadata.json
        task_name = None
        training_meta_path = run_dir / "training_metadata.json"
        if training_meta_path.exists():
            try:
                with open(training_meta_path) as f:
                    meta = json.load(f)
                    task_name = meta.get("task_name")
            except Exception:
                pass
        
        # Fallback: use first use_only_label or a generic name
        if task_name is None:
            if use_only_labels and len(use_only_labels) == 1:
                task_name = use_only_labels[0]
            else:
                task_name = "single_task"
        
        head = SingleTaskHead(
            hidden_dim=model.config.hidden_size,
            num_classes=num_classes,
            task_name=task_name,
        )
        logger.info(f"Loaded SingleTaskHead for task '{task_name}' ({num_classes} classes)")
    else:
        # MultiTaskHead: infer num_labels_per_task from checkpoint
        num_labels_per_task = {}
        for key in head_state.keys():
            if key.endswith(".weight"):
                task_name = key.split(".")[1]  # Extract task name from "heads.{task}.weight"
                num_labels = head_state[key].shape[0]
                num_labels_per_task[task_name] = num_labels

        head = MultiTaskHead(
            hidden_dim=model.config.hidden_size,
            num_labels_per_task=num_labels_per_task,
        )

    model_state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(model_state)
    head.load_state_dict(head_state)

    model.to(device)
    head.to(device)
    model.eval()
    head.eval()
    return model, head


def _build_loader(
    *,
    dataset_path: Path,
    text_field: str,
    taxonomy_path: Path,
    max_length: int,
    label_maps,
    batch_size: int,
    split: str,
    use_only_labels: Optional[List[str]] = None,
) -> DataLoader:
    # Determine label_fields based on filter
    if use_only_labels:
        label_fields = list(use_only_labels)
    else:
        label_fields = get_demographic_columns()
    
    dataset = PhaseDDataset(
        dataset_path,
        text_field=text_field,
        label_fields=label_fields,
        label_maps=label_maps,
        split=split,
        use_only_labels=use_only_labels,
    )
    tokenizer = PhaseDTokenizer(
        model_name="roberta-base",
        max_length=max_length,
        taxonomy_path=taxonomy_path,
    )
    collator = PhaseDCollator(tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


@torch.no_grad()
def _compute_svs(
    *,
    model: AffineGuardTransformer,
    loader: DataLoader,
    facilitating_heads,
    max_batches: int,
    query_strategy: str,
) -> SVSResult:
    calculator = SVSCalculator(model.tokenizer.tokenizer)
    function_mass = 0.0
    content_mass = 0.0
    batches_seen = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(next(model.parameters()).device)
        attention_mask = batch["attention_mask"].to(next(model.parameters()).device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        result = calculator.compute_svs(
            attentions=outputs["attentions"],
            input_ids=input_ids,
            facilitating_heads=facilitating_heads,
            attention_mask=attention_mask,
            query_strategy=query_strategy,
        )
        function_mass += result.function_mass
        content_mass += result.content_mass
        batches_seen += 1
        if max_batches and batches_seen >= max_batches:
            break

    svs = function_mass / content_mass if content_mass > 0 else 0.0
    return SVSResult(svs=svs, function_mass=function_mass, content_mass=content_mass)


def run_verification(
    *,
    dataset_path: Path,
    artifacts_dir: Path,
    phase_d_dir: Path,
    output_dir: Path,
    model_name: str,
    taxonomy_path: Path,
    max_length: int,
    batch_size: int,
    chg_epochs: int,
    chg_lr: float,
    chg_regularization: float,
    facilitating_threshold: float,
    irrelevant_threshold: float,
    svs_max_batches: int,
    svs_query_strategy: str,
    use_only_labels: Optional[tuple[str, ...]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which label fields to use for label maps
    if use_only_labels:
        label_fields_for_maps = list(use_only_labels)
        logger.info(f"Verification using filtered labels: {label_fields_for_maps}")
    else:
        label_fields_for_maps = get_demographic_columns()
    
    label_maps = load_label_maps(dataset_path, label_fields_for_maps).maps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    svs_query_strategy = (svs_query_strategy or "mean_tokens").strip().lower()
    if svs_query_strategy in {"default", "auto"}:
        svs_query_strategy = "mean_tokens"

    def _verify_run(run_name: str, text_field: str, use_affine_guard: bool) -> Dict:
        run_dir = phase_d_dir / run_name
        model, head = _load_model_and_head(
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            taxonomy_path=taxonomy_path,
            max_length=max_length,
            use_affine_guard=use_affine_guard,
            label_maps=label_maps,
            device=device,
            use_only_labels=list(use_only_labels) if use_only_labels else None,
        )

        loader = _build_loader(
            dataset_path=dataset_path,
            text_field=text_field,
            taxonomy_path=taxonomy_path,
            max_length=max_length,
            label_maps=load_label_maps(dataset_path, label_fields_for_maps),
            batch_size=batch_size,
            split="val",
            use_only_labels=list(use_only_labels) if use_only_labels else None,
        )

        chg = CHGVerifier(
            num_layers=model.config.num_hidden_layers,
            num_heads=model.config.num_attention_heads,
            learning_rate=chg_lr,
            num_epochs=chg_epochs,
            regularization=chg_regularization,
        )
        gates = chg.learn_gates(
            model=model,
            head=head,
            dataloader=loader,
            device=device,
        )
        classifications = gates.classify_heads(
            facilitating_threshold=facilitating_threshold,
            irrelevant_threshold=irrelevant_threshold,
        )
        gate_path = output_dir / f"chg_gates_{run_name}.pt"
        torch.save(gates.gates.cpu(), gate_path)

        with (output_dir / f"head_classification_{run_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(classifications, handle, indent=2)

        svs_result = _compute_svs(
            model=model,
            loader=loader,
            facilitating_heads=classifications["facilitating"],
            max_batches=svs_max_batches,
            query_strategy=svs_query_strategy,
        )
        svs_path = output_dir / f"svs_{run_name}.json"
        with svs_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "svs": svs_result.svs,
                    "function_mass": svs_result.function_mass,
                    "content_mass": svs_result.content_mass,
                },
                handle,
                indent=2,
            )

        return {
            "gates_path": str(gate_path),
            "svs_path": str(svs_path),
            "facilitating_count": len(classifications["facilitating"]),
            "irrelevant_count": len(classifications["irrelevant"]),
            "neutral_count": len(classifications["neutral"]),
        }

    baseline_outputs = _verify_run("baseline", "post", use_affine_guard=False)
    constrained_outputs = _verify_run("constrained", "post_masked", use_affine_guard=True)

    summary = {
        "baseline": baseline_outputs,
        "constrained": constrained_outputs,
    }
    try:
        with (output_dir / "svs_baseline.json").open("r", encoding="utf-8") as handle:
            baseline_svs = json.load(handle).get("svs", 0.0)
        with (output_dir / "svs_constrained.json").open("r", encoding="utf-8") as handle:
            constrained_svs = json.load(handle).get("svs", 0.0)
        summary["svs_delta"] = float(constrained_svs) - float(baseline_svs)
    except Exception:
        summary["svs_delta"] = None

    summary_path = output_dir / "verification_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
