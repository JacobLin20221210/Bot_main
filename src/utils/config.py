from __future__ import annotations

from typing import Any


LANGUAGE_DATASET_IDS: dict[str, tuple[str, str]] = {
    "en": ("30", "32"),
    "fr": ("31", "33"),
}


LANGUAGE_STRICT_HOLDOUTS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "en": (("30", "32"), ("32", "30")),
    "fr": (("31", "33"), ("33", "31")),
}


TRANSFER_MODEL_LIBRARY: dict[str, dict[str, Any]] = {
    "rfet_balanced": {
        "kind": "bot_ensemble",
        "params": {
            "rf_estimators": 900,
            "et_estimators": 1200,
            "min_samples_leaf": 2,
            "rf_bot_weight": 1.2,
            "et_bot_weight": 1.3,
            "calibration_cv": 3,
        },
    },
    "rfet_precision": {
        "kind": "bot_ensemble",
        "params": {
            "rf_estimators": 1100,
            "et_estimators": 1400,
            "min_samples_leaf": 3,
            "rf_bot_weight": 1.1,
            "et_bot_weight": 1.15,
            "calibration_cv": 3,
        },
    },
    "rfet_recall": {
        "kind": "bot_ensemble",
        "params": {
            "rf_estimators": 950,
            "et_estimators": 1300,
            "min_samples_leaf": 2,
            "rf_bot_weight": 1.35,
            "et_bot_weight": 1.45,
            "calibration_cv": 3,
        },
    },
    "tab_lr_c1": {
        "kind": "tab_lr",
        "params": {
            "c": 1.0,
            "bot_weight": 1.2,
        },
    },
    "tab_lr_c2": {
        "kind": "tab_lr",
        "params": {
            "c": 2.0,
            "bot_weight": 1.3,
        },
    },
    "tab_lr_c4": {
        "kind": "tab_lr",
        "params": {
            "c": 4.0,
            "bot_weight": 1.45,
        },
    },
    "tab_lr_c06": {
        "kind": "tab_lr",
        "params": {
            "c": 0.6,
            "bot_weight": 1.1,
        },
    },
    "tab_lr_c8": {
        "kind": "tab_lr",
        "params": {
            "c": 8.0,
            "bot_weight": 1.55,
        },
    },
    "text_lr_balanced": {
        "kind": "text_lr",
        "params": {
            "c": 1.0,
            "bot_weight": 1.15,
            "word_max_features": 12000,
            "char_max_features": 18000,
        },
    },
    "text_lr_precision": {
        "kind": "text_lr",
        "params": {
            "c": 0.75,
            "bot_weight": 1.0,
            "word_max_features": 10000,
            "char_max_features": 16000,
        },
    },
    "text_lr_recall": {
        "kind": "text_lr",
        "params": {
            "c": 1.5,
            "bot_weight": 1.4,
            "word_max_features": 14000,
            "char_max_features": 22000,
        },
    },
    "text_lr_tight_precision": {
        "kind": "text_lr",
        "params": {
            "c": 0.5,
            "bot_weight": 0.95,
            "word_max_features": 9000,
            "char_max_features": 14000,
        },
    },
    "text_lr_hi_recall": {
        "kind": "text_lr",
        "params": {
            "c": 2.2,
            "bot_weight": 1.55,
            "word_max_features": 18000,
            "char_max_features": 26000,
        },
    },
    "emb_lr_balanced": {
        "kind": "embedding_lr",
        "params": {
            "c": 1.0,
            "bot_weight": 1.2,
            "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
    },
    "emb_lr_recall": {
        "kind": "embedding_lr",
        "params": {
            "c": 1.6,
            "bot_weight": 1.35,
            "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
    },
    "emb_lr_precision": {
        "kind": "embedding_lr",
        "params": {
            "c": 0.7,
            "bot_weight": 1.0,
            "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
    },
    "emb_lr_strong": {
        "kind": "embedding_lr",
        "params": {
            "c": 2.4,
            "bot_weight": 1.5,
            "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
    },
    "emb_lr_e5_precision": {
        "kind": "embedding_lr",
        "params": {
            "c": 0.75,
            "bot_weight": 1.0,
            "embedding_model_name": "intfloat/multilingual-e5-large-instruct",
        },
    },
    "emb_lr_jina_precision": {
        "kind": "embedding_lr",
        "params": {
            "c": 0.75,
            "bot_weight": 1.0,
            "embedding_model_name": "jinaai/jina-embeddings-v3",
        },
    },
    "emb_lr_bge_precision": {
        "kind": "embedding_lr",
        "params": {
            "c": 0.75,
            "bot_weight": 1.0,
            "embedding_model_name": "BAAI/bge-m3",
        },
    },
    "emb_lr_qwen_precision": {
        "kind": "embedding_lr",
        "params": {
            "c": 0.75,
            "bot_weight": 1.0,
            "embedding_model_name": "Qwen/Qwen3-Embedding-0.6B",
        },
    },
    "graph_knn_balanced": {
        "kind": "graph_knn",
        "params": {
            "c": 1.2,
            "bot_weight": 1.25,
            "k_neighbors": 24,
            "graph_weight": 0.35,
            "tabular_weight": 0.75,
            "embedding_weight": 0.25,
            "temperature": 0.14,
            "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
    },
    "graph_knn_precision": {
        "kind": "graph_knn",
        "params": {
            "c": 0.9,
            "bot_weight": 1.1,
            "k_neighbors": 28,
            "graph_weight": 0.42,
            "tabular_weight": 0.7,
            "embedding_weight": 0.3,
            "temperature": 0.16,
            "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
    },
    "graph_knn_e5": {
        "kind": "graph_knn",
        "params": {
            "c": 0.95,
            "bot_weight": 1.1,
            "k_neighbors": 26,
            "graph_weight": 0.42,
            "tabular_weight": 0.7,
            "embedding_weight": 0.3,
            "temperature": 0.16,
            "embedding_model_name": "intfloat/multilingual-e5-large-instruct",
        },
    },
    "graph_knn_jina": {
        "kind": "graph_knn",
        "params": {
            "c": 0.95,
            "bot_weight": 1.1,
            "k_neighbors": 26,
            "graph_weight": 0.42,
            "tabular_weight": 0.7,
            "embedding_weight": 0.3,
            "temperature": 0.16,
            "embedding_model_name": "jinaai/jina-embeddings-v3",
        },
    },
    "llm_semantic_balanced": {
        "kind": "llm_semantic_lr",
        "params": {
            "c": 1.1,
            "bot_weight": 1.2,
            "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
    },
    "llm_semantic_recall": {
        "kind": "llm_semantic_lr",
        "params": {
            "c": 1.6,
            "bot_weight": 1.35,
            "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
    },
    "llm_semantic_e5": {
        "kind": "llm_semantic_lr",
        "params": {
            "c": 1.05,
            "bot_weight": 1.1,
            "embedding_model_name": "intfloat/multilingual-e5-large-instruct",
        },
    },
    "llm_semantic_jina": {
        "kind": "llm_semantic_lr",
        "params": {
            "c": 1.05,
            "bot_weight": 1.1,
            "embedding_model_name": "jinaai/jina-embeddings-v3",
        },
    },
    "neural_compact": {
        "kind": "neural",
        "params": {
            "hidden_layer_sizes": (192, 96),
            "alpha": 2e-4,
            "max_iter": 320,
            "text_svd_components": 96,
        },
    },
    "neural_wide": {
        "kind": "neural",
        "params": {
            "hidden_layer_sizes": (256, 128),
            "alpha": 1e-4,
            "max_iter": 360,
            "text_svd_components": 128,
        },
    },
}


BEST_LANGUAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "en": {
        "config_id": "transfer_en_best_v5_llm_semantic_tuned",
        "source_run_id": "20260207-051503-24d4f07cf8-heavy-en-noboosters",
        "selection_protocol": "source_only",
        "selection_bias_safe": True,
        "selection_metric": "worst_train_oof_score_then_mean_then_lower_fp_with_simplicity",
        "simplicity_epsilon": 0.02,
        "objective_priority": ["competition_score", "tp", "accuracy", "worst_fold_score"],
        "components": [
            {"model": "rfet_balanced", "weight": 0.75},
            {"model": "tab_lr_c1", "weight": 0.05},
            {"model": "llm_semantic_balanced", "weight": 0.20},
        ],
        "inference": {
            "mode": "static",
            "threshold": 0.2675,
            "margin": 0.0,
        },
        "fold_overrides": {
            "train_30__test_32": {"threshold": 0.235, "margin": 0.0, "min_precision": 0.72},
            "train_32__test_30": {"threshold": 0.30, "margin": 0.0, "min_precision": 0.72},
        },
        "cascade": {
            "enabled": False,
            "shortlist_model": "tab_lr_c4",
            "shortlist_threshold_grid": [0.25, 0.35, 0.45, 0.55, 0.65],
            "stage2_weight_grid": [0.7, 0.85, 0.95],
        },
        "selection": {
            "threshold_step": 0.0025,
            "min_precision_grid": [0.68, 0.72, 0.78, 0.84, 0.90],
            "margin_grid": [0.0, 0.0025, 0.005, 0.01],
            "max_fp_rate": -1.0,
            "robust_seed_count": 7,
            "robust_seed_stride": 53,
            "use_contrastive_calibration": True,
            "contrastive_calibration": {
                "hard_fraction": 0.14,
                "hard_weight": 3.25,
                "c": 1.2,
                "min_samples": 24,
            },
        },
    },
    "fr": {
        "config_id": "transfer_fr_best_v1",
        "source_run_id": "20260207-051910-4477ef6096-heavy-fr-noboosters-v2",
        "selection_protocol": "source_only",
        "selection_bias_safe": True,
        "selection_metric": "worst_train_oof_score_then_mean_then_lower_fp_with_simplicity",
        "simplicity_epsilon": 0.02,
        "objective_priority": ["competition_score", "tp", "accuracy", "worst_fold_score"],
        "components": [
            {"model": "rfet_recall", "weight": 0.15},
            {"model": "text_lr_recall", "weight": 0.85},
        ],
        "inference": {
            "mode": "static",
            "threshold": 0.20375,
            "margin": 0.0,
        },
        "fold_overrides": {
            "train_31__test_33": {"threshold": 0.2075, "margin": 0.0, "min_precision": 0.72},
            "train_33__test_31": {"threshold": 0.20, "margin": 0.0, "min_precision": 0.72},
        },
        "cascade": {
            "enabled": False,
            "shortlist_model": "text_lr_recall",
            "shortlist_threshold_grid": [0.25, 0.35, 0.45, 0.55, 0.65],
            "stage2_weight_grid": [0.7, 0.85, 0.95],
        },
        "selection": {
            "threshold_step": 0.0025,
            "min_precision_grid": [0.68, 0.72, 0.78, 0.84, 0.90],
            "margin_grid": [0.0, 0.0025, 0.005, 0.01],
            "max_fp_rate": -1.0,
            "robust_seed_count": 7,
            "robust_seed_stride": 53,
            "use_contrastive_calibration": True,
            "contrastive_calibration": {
                "hard_fraction": 0.14,
                "hard_weight": 3.25,
                "c": 1.2,
                "min_samples": 24,
            },
        },
    },
}
