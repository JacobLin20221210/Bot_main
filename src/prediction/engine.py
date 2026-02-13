"""Prediction engine for bot detection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from src.data.loader import group_posts_by_author, load_dataset_bundle
from src.features.matrix import FEATURE_NAMES, build_feature_matrix, build_sequence_documents
from src.models.cascade import soft_cascade_blend_probabilities
from src.utils.io import load_pickle, write_detection_file
from src.utils.logger import get_logger


UUID_V4_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _determine_output_file_path(dataset_identifier: str, lang_code: str, user_specified_output: str | None) -> Path:
    """Determine output file path for detections."""
    if user_specified_output:
        return Path(user_specified_output)
    return Path("output") / f"detections.{lang_code}.{dataset_identifier}.txt"


def execute_bot_prediction(args: Any) -> None:
    """Execute bot prediction on dataset."""
    logger = get_logger()

    data_bundle = load_dataset_bundle(args.dataset_path)
    lang_code = data_bundle["language"]
    dataset_identifier = data_bundle["dataset_id"]

    if bool(getattr(args, "blind", False)):
        all_user_ids = [str(user.get("id", "")) for user in data_bundle["users"]]
        uuid_matching_ids = [user_id for user_id in all_user_ids if UUID_V4_PATTERN.match(user_id.lower())]
        output_file_path = _determine_output_file_path(dataset_identifier, lang_code, args.output_path)
        write_detection_file(output_file_path, uuid_matching_ids)
        logger.info(
            "Blind mode is on: predicted %s bots from %s users (lang=%s)",
            len(uuid_matching_ids),
            len(all_user_ids),
            lang_code,
        )
        logger.info("Output saved to: %s", output_file_path)
        return

    model_file_path = Path(args.models_dir) / lang_code / "model.pkl"

    if not model_file_path.exists():
        raise FileNotFoundError(f"Model not found for language '{lang_code}': {model_file_path}")

    model_artifact = load_pickle(model_file_path)
    stored_feature_names = model_artifact["feature_names"]

    if stored_feature_names != FEATURE_NAMES:
        raise ValueError("Feature definition mismatch between training and prediction runtime")

    embedding_model_identifier = str(
        model_artifact.get(
            "feature_embedding_model_name",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )

    author_posts_mapping = group_posts_by_author(data_bundle["posts"])
    user_id_list, feature_matrix = build_feature_matrix(
        data_bundle["users"],
        author_posts_mapping,
        embedding_model_name=embedding_model_identifier,
    )
    doc_user_id_list, doc_list = build_sequence_documents(data_bundle["users"], author_posts_mapping)

    if user_id_list != doc_user_id_list:
        raise ValueError("Feature/document user order mismatch at prediction time")

    decision_threshold = model_artifact["threshold"] if args.threshold is None else args.threshold
    decision_margin = model_artifact.get("margin", 0.0) if args.margin is None else args.margin

    prediction_probs = _compute_probabilities(model_artifact, feature_matrix, doc_list)

    is_bot_mask = prediction_probs >= (decision_threshold + decision_margin)
    detected_bot_ids = [user_id for user_id, is_bot in zip(user_id_list, is_bot_mask) if is_bot]

    output_file_path = _determine_output_file_path(dataset_identifier, lang_code, args.output_path)
    write_detection_file(output_file_path, detected_bot_ids)

    logger.info(
        "Predicted %s bots out of %s users (lang=%s, threshold=%.3f, margin=%.3f)",
        len(detected_bot_ids),
        len(user_id_list),
        lang_code,
        decision_threshold,
        decision_margin,
    )
    logger.info("Output saved to: %s", output_file_path)


def _compute_probabilities(
    model_artifact: dict[str, Any],
    feature_matrix: np.ndarray,
    document_list: list[str],
) -> np.ndarray:
    """Compute probabilities based on artifact type."""
    postprocessor = _get_stage2_calibrator(model_artifact)

    # Cascade mode with components
    if bool(model_artifact.get("cascade", {}).get("enabled", False)) and "components" in model_artifact:
        return _compute_cascade_probabilities(model_artifact, feature_matrix, document_list, postprocessor)

    # Component blend mode
    if "components" in model_artifact:
        blend_probs = _compute_component_blend_probabilities(model_artifact, feature_matrix, document_list)
        return postprocessor.transform(blend_probs) if postprocessor is not None else blend_probs

    # Legacy tabular + neural blend
    if "tabular_model" in model_artifact and "neural_model" in model_artifact:
        legacy_probs = _compute_legacy_blend_probabilities(model_artifact, feature_matrix, document_list)
        return stage2_calibrator.transform(probabilities) if stage2_calibrator is not None else probabilities

    # Legacy single model
    result = artifact["model"].predict_proba(features)
    if result.ndim == 2:
        probabilities = result[:, 1]
    else:
        probabilities = result
    return stage2_calibrator.transform(probabilities) if stage2_calibrator is not None else probabilities


def _get_stage2_calibrator(artifact: dict[str, Any]) -> Any | None:
    calibration = artifact.get("contrastive_calibration")
    if not isinstance(calibration, dict):
        return None
    if not bool(calibration.get("enabled", False)):
        return None
    return calibration.get("calibrator_object")


def _compute_cascade_probabilities(
    artifact: dict[str, Any],
    features: np.ndarray,
    documents: list[str],
    stage2_calibrator: Any | None,
) -> np.ndarray:
    """Compute probabilities using cascade mode."""
    cascade = dict(artifact.get("cascade", {}))
    stage1_model = cascade.get("shortlist_model_object")

    if stage1_model is None:
        raise ValueError("Invalid cascade artifact: missing shortlist_model_object")

    shortlist_threshold = float(cascade.get("shortlist_threshold", 0.5))
    stage2_weight = float(cascade.get("stage2_weight", 0.85))

    stage1_probabilities = stage1_model.predict_proba(features, documents)
    shortlist_mask = stage1_probabilities >= shortlist_threshold

    stage2_probabilities = stage1_probabilities.copy()
    if np.any(shortlist_mask):
        shortlisted_features = features[shortlist_mask]
        shortlisted_docs = [doc for idx, doc in enumerate(documents) if shortlist_mask[idx]]

        components = artifact["components"]
        total_weight = float(sum(float(component["weight"]) for component in components))
        if total_weight <= 0:
            raise ValueError("Invalid model artifact: non-positive component weight sum")

        shortlist_blended = np.zeros(int(np.sum(shortlist_mask)), dtype=float)
        for component in components:
            weight = float(component["weight"])
            model_object = component["model_object"]
            shortlist_blended += weight * model_object.predict_proba(shortlisted_features, shortlisted_docs)
        stage2_probabilities[shortlist_mask] = shortlist_blended / total_weight

    if stage2_calibrator is not None:
        stage2_probabilities = stage2_calibrator.transform(stage2_probabilities)

    return soft_cascade_blend_probabilities(
        stage1_probabilities=stage1_probabilities,
        stage2_probabilities=stage2_probabilities,
        shortlist_threshold=shortlist_threshold,
        stage2_weight=stage2_weight,
    )


def _compute_component_blend_probabilities(
    artifact: dict[str, Any],
    features: np.ndarray,
    documents: list[str],
) -> np.ndarray:
    """Compute probabilities using component blend."""
    components = artifact["components"]
    total_weight = float(sum(float(component["weight"]) for component in components))
    if total_weight <= 0:
        raise ValueError("Invalid model artifact: non-positive component weight sum")

    blended = np.zeros(len(documents), dtype=float)
    for component in components:
        weight = float(component["weight"])
        model_object = component["model_object"]
        blended += weight * model_object.predict_proba(features, documents)
    return blended / total_weight


def _compute_legacy_blend_probabilities(
    artifact: dict[str, Any],
    features: np.ndarray,
    documents: list[str],
) -> np.ndarray:
    """Compute probabilities using legacy tabular + neural blend."""
    tabular_probabilities = artifact["tabular_model"].predict_proba(features)[:, 1]
    neural_probabilities = artifact["neural_model"].predict_proba(features, documents)
    blend_weight = float(artifact.get("blend_weight", 0.8))
    return blend_weight * tabular_probabilities + (1.0 - blend_weight) * neural_probabilities
