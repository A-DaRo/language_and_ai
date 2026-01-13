"""Phase D verification: CHG + SVS runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from ..data_engine.schemas import get_demographic_columns
from .chg_verifier import CHGVerifier
from .classification_head import MultiTaskHead
from .phase_d_dataset import PhaseDCollator, PhaseDDataset, load_label_maps
from .svs_calculator import SVSCalculator, SVSResult
from .tokenizer import PhaseDTokenizer
from .transformer import AffineGuardTransformer


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
) -> Tuple[AffineGuardTransformer, MultiTaskHead]:
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

    head = MultiTaskHead(
        hidden_dim=model.config.hidden_size,
        num_labels_per_task={k: len(v) for k, v in label_maps.items()},
    )

    model_state = torch.load(run_dir / "model.pt", map_location=device)
    head_state = torch.load(run_dir / "head.pt", map_location=device)
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
) -> DataLoader:
    dataset = PhaseDDataset(
        dataset_path,
        text_field=text_field,
        label_fields=get_demographic_columns(),
        label_maps=label_maps,
        split=split,
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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_maps = load_label_maps(dataset_path, get_demographic_columns()).maps
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
        )

        loader = _build_loader(
            dataset_path=dataset_path,
            text_field=text_field,
            taxonomy_path=taxonomy_path,
            max_length=max_length,
            label_maps=load_label_maps(dataset_path, get_demographic_columns()),
            batch_size=batch_size,
            split="val",
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
