"""Phase D verification: CHG + SVS runners."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pyarrow.compute as pc
import pyarrow.feather as feather
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data_engine.schemas import get_demographic_columns
from .chg_verifier import CHGVerifier
from .classification_head import MultiTaskHead, SingleTaskHead
from .phase_d_dataset import (
    FastCollator,
    PhaseDCollator,
    PhaseDDataset,
    PhaseDLabelMaps,
    load_label_maps,
)
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
    # Load with strict=False to handle potential embedding size mismatch
    # (Training may not resize embeddings for mask tokens)
    model.load_state_dict(model_state, strict=False)
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
    model_name: str,
    max_length: int,
    label_maps,
    batch_size: int,
    split: str,
    use_only_labels: Optional[List[str]] = None,
) -> tuple[DataLoader, PhaseDTokenizer]:
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
        model_name=model_name,
        max_length=max_length,
        taxonomy_path=taxonomy_path,
    )
    if dataset.is_aot_mode:
        collator = FastCollator(
            max_length=max_length,
            pad_token_id=1,
        )
    else:
        collator = PhaseDCollator(tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator), tokenizer


@torch.no_grad()
def _compute_svs(
    *,
    model: AffineGuardTransformer,
    loader: DataLoader,
    tokenizer,  # HF tokenizer instance (not PhaseDTokenizer)
    facilitating_heads,
    max_batches: int,
    query_strategy: str,
    svs_use_pos: bool,
    svs_spacy_model: str,
    svs_spacy_use_gpu: bool,
    svs_spacy_gpu_id: int | None,
    svs_spacy_batch_size: int,
    svs_spacy_n_process: int,
    svs_spacy_disable: list[str],
) -> SVSResult:
    calculator = SVSCalculator(
        tokenizer,
        use_pos=svs_use_pos,
        spacy_model=svs_spacy_model,
        spacy_use_gpu=svs_spacy_use_gpu,
        spacy_gpu_id=svs_spacy_gpu_id,
        spacy_batch_size=svs_spacy_batch_size,
        spacy_n_process=svs_spacy_n_process,
        spacy_disable=svs_spacy_disable,
    )
    function_mass = 0.0
    content_mass = 0.0
    batches_seen = 0

    total = None
    if max_batches:
        total = max_batches
        try:
            total = min(int(total), len(loader))
        except TypeError:
            pass
    else:
        try:
            total = len(loader)
        except TypeError:
            total = None

    for batch in tqdm(loader, desc="SVS batches", unit="batch", total=total):
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


def _load_label_maps_from_checkpoint(run_dir: Path) -> Optional[PhaseDLabelMaps]:
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        return None
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        label_maps = metadata.get("label_maps")
    except Exception as exc:
        logger.warning("Failed to load label maps from %s: %s", checkpoint_path, exc)
        return None
    if isinstance(label_maps, dict) and label_maps:
        return PhaseDLabelMaps(maps=label_maps)
    return None


def _prescan_label_validity(dataset_path: Path, label_maps: PhaseDLabelMaps) -> None:
    try:
        table = feather.read_table(dataset_path, memory_map=True)
    except Exception as exc:
        logger.warning("Failed to read dataset for prescan (%s): %s", dataset_path, exc)
        return

    total_rows = int(table.num_rows)
    if total_rows == 0:
        logger.warning("Prescan skipped (0 rows) for %s", dataset_path)
        return

    logger.info("Prescanning label validity: %s (%d rows)", dataset_path, total_rows)
    for field, mapping in label_maps.maps.items():
        if field not in table.column_names:
            logger.warning("Prescan: label column missing: %s", field)
            continue

        col = table[field]
        valid_mask = pc.is_valid(col)
        valid_count = int(pc.sum(valid_mask).as_py())
        missing_count = total_rows - valid_count

        invalid_count = 0
        if mapping:
            try:
                allowed_values = list(mapping.keys())
                in_map = pc.is_in(col, value_set=allowed_values)
                invalid_mask = pc.and_(valid_mask, pc.invert(in_map))
                invalid_count = int(pc.sum(invalid_mask).as_py())
            except Exception as exc:
                logger.warning(
                    "Prescan skipped invalid-value check for %s (type mismatch): %s",
                    field,
                    exc,
                )
                invalid_count = 0
        else:
            invalid_count = valid_count

        ignored = missing_count + invalid_count
        if ignored:
            logger.warning(
                "Prescan: %s ignored=%d (missing=%d, invalid=%d) out of %d",
                field,
                ignored,
                missing_count,
                invalid_count,
                total_rows,
            )
        else:
            logger.info("Prescan: %s ignored=0 out of %d", field, total_rows)


def run_verification(
    *,
    dataset_path: Path,
    dataset_path_post: Path | None = None,
    dataset_path_masked: Path | None = None,
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
    svs_use_pos: bool,
    svs_spacy_model: str,
    svs_spacy_use_gpu: bool,
    svs_spacy_gpu_id: int | None,
    svs_spacy_batch_size: int,
    svs_spacy_n_process: int,
    svs_spacy_disable: list[str],
    use_only_labels: Optional[tuple[str, ...]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_maps_obj = _load_label_maps_from_checkpoint(phase_d_dir / "baseline")
    if label_maps_obj is None:
        label_maps_obj = _load_label_maps_from_checkpoint(phase_d_dir / "constrained")
    if label_maps_obj is None:
        # Determine which label fields to use for label maps
        if use_only_labels:
            label_fields_for_maps = list(use_only_labels)
            logger.info(f"Verification using filtered labels: {label_fields_for_maps}")
        else:
            label_fields_for_maps = get_demographic_columns()
        dataset_for_maps = dataset_path_post or dataset_path_masked or dataset_path
        label_maps_obj = load_label_maps(dataset_for_maps, label_fields_for_maps)
    else:
        logger.info("Loaded label maps from training checkpoint metadata")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prescan_paths: List[Path] = []
    for candidate in (dataset_path_post, dataset_path_masked, dataset_path):
        if candidate is None:
            continue
        if candidate.exists() and candidate not in prescan_paths:
            prescan_paths.append(candidate)
    for path in prescan_paths:
        _prescan_label_validity(path, label_maps_obj)

    svs_query_strategy = (svs_query_strategy or "mean_tokens").strip().lower()
    if svs_query_strategy in {"default", "auto"}:
        svs_query_strategy = "mean_tokens"

    def _resolve_dataset(candidate: Optional[Path]) -> Path:
        if candidate is None:
            return dataset_path
        if candidate.exists():
            return candidate
        logger.warning("Dataset not found at %s; falling back to %s", candidate, dataset_path)
        return dataset_path

    def _verify_run(
        run_name: str,
        text_field: str,
        use_affine_guard: bool,
        *,
        dataset_for_run: Optional[Path],
    ) -> Dict:
        dataset_for_run = _resolve_dataset(dataset_for_run)
        run_dir = phase_d_dir / run_name
        model, head = _load_model_and_head(
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            taxonomy_path=taxonomy_path,
            max_length=max_length,
            use_affine_guard=use_affine_guard,
            label_maps=label_maps_obj.maps,
            device=device,
            use_only_labels=list(use_only_labels) if use_only_labels else None,
        )

        loader, phase_d_tokenizer = _build_loader(
            dataset_path=dataset_for_run,
            text_field=text_field,
            taxonomy_path=taxonomy_path,
            model_name=model_name,
            max_length=max_length,
            label_maps=label_maps_obj,
            batch_size=batch_size,
            split="val",
            use_only_labels=list(use_only_labels) if use_only_labels else None,
        )
        
        # Resize model embeddings to match tokenizer vocabulary (includes mask tokens)
        phase_d_tokenizer.resize_model_embeddings(model.model)

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
            tokenizer=phase_d_tokenizer.tokenizer,  # Pass underlying HF tokenizer
            facilitating_heads=classifications["facilitating"],
            max_batches=svs_max_batches,
            query_strategy=svs_query_strategy,
            svs_use_pos=svs_use_pos,
            svs_spacy_model=svs_spacy_model,
            svs_spacy_use_gpu=svs_spacy_use_gpu,
            svs_spacy_gpu_id=svs_spacy_gpu_id,
            svs_spacy_batch_size=svs_spacy_batch_size,
            svs_spacy_n_process=svs_spacy_n_process,
            svs_spacy_disable=svs_spacy_disable,
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

    baseline_outputs = _verify_run(
        "baseline",
        "post",
        use_affine_guard=False,
        dataset_for_run=dataset_path_post,
    )
    constrained_outputs = _verify_run(
        "constrained",
        "post_masked",
        use_affine_guard=True,
        dataset_for_run=dataset_path_masked,
    )

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
