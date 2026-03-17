import argparse
from dataclasses import dataclass, fields
from functools import partial
from math import ceil
import json
import os
from pathlib import Path
import shutil
from typing import Optional
from mpi4py import MPI

from tensorrt_llm.models import QWenForCausalLM, LLaMAForCausalLM
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.builder import BuildConfig as TRTBuildConfig, build

from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.models.modeling_utils import QuantConfig
import tensorrt_llm.quantization.quantize_by_modelopt as _modelopt_mod

from datasets import load_dataset
import os


def _get_calib_dataloader(dataset_name_or_dir="cnn_dailymail",
                          tokenizer=None,
                          batch_size=1,
                          calib_size=512,
                          block_size=512,
                          device=None,
                          include_labels=False,
                          split="train",
                          text_field="text"):
    from tensorrt_llm.quantization.quantize_by_modelopt import MULTIMODAL_DATASETS, _CustomDataset, DataLoader
    import torch

    if any(name in dataset_name_or_dir for name in MULTIMODAL_DATASETS):
        raise NotImplementedError(f"Multimodal calibration datasets are not supported yet: {dataset_name_or_dir}")

    dataset = load_dataset(dataset_name_or_dir, split=split, trust_remote_code=True)
    dataset = dataset[text_field][:calib_size]

    batch_encoded = tokenizer.batch_encode_plus(dataset, return_tensors="pt",
                                                padding=True, truncation=True,
                                                max_length=block_size)
    if device:
        batch_encoded = batch_encoded.to(device)

    if include_labels:
        batch_encoded["labels"] = torch.where(
            batch_encoded["attention_mask"] > 0.5,
            batch_encoded["input_ids"], -100)
        batch_encoded = _CustomDataset(batch_encoded)
    else:
        batch_encoded = _CustomDataset({"input_ids": batch_encoded["input_ids"]})

    return DataLoader(batch_encoded, batch_size=batch_size, shuffle=False)


_modelopt_mod.get_calib_dataloader = _get_calib_dataloader

SUPPORTED_QUANTS = {"W8A16", "W4A16", "W4A16_AWQ", "W4A8_AWQ", "FP8", "W8A8_SQ", "NO_QUANT", "NONE", "W16A16"}
SUPPORTED_KV_QUANTS = {"fp8", "int8"}
SUPPORTED_MODELS = {"mistral", "llama", "qwen"}


@dataclass
class BuildConfig:
    # Model
    model_type: str = "qwen"
    dtype: str = "float16"

    # Quantization
    quant_mode: str = "W4A16"
    kv_cache_dtype: Optional[str] = None

    # Engine build
    max_batch_size: int = 32
    max_input_len: int = 2048
    max_seq_len: int = 6144
    max_num_tokens: int = 12288
    max_beam_width: int = 1
    gather_context_logits: bool = False

    # Calibration
    calib_source: str = "neuralmagic/LLM_compression_calibration"
    calib_split: str = "train"
    calib_text_field: str = "text"
    calib_num_samples: int = 2048
    calib_batch_size: int = 16
    calib_max_seq_length: int = 6144
    random_seed: int = 0

    # Paths (unannotated to match EvalConfig pattern)
    model_dir = Path("")
    engine_out_dir: Optional[Path] = None
    checkpoint_out_dir: Optional[Path] = None
    checkpoint_in_dir: Optional[Path] = None


def _load_paths(paths_file: str) -> dict:
    with open(paths_file) as f:
        return json.load(f)


def _resolve_path(value: str, root_key: str, paths: dict) -> Path:
    """If value is a relative path, prepend the matching root from paths.json."""
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(paths[root_key]) / p


def load_config(config_path: str, paths_file: str) -> BuildConfig:
    with open(config_path) as f:
        data = json.load(f)

    paths = _load_paths(paths_file)

    field_names = {f.name for f in fields(BuildConfig)}
    path_keys = {"model_dir", "engine_out_dir", "checkpoint_out_dir", "checkpoint_in_dir"}
    unknown = set(data) - field_names - path_keys
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    kwargs = {k: v for k, v in data.items() if k in field_names}

    # Normalize kv_cache_dtype
    if "kv_cache_dtype" in kwargs and isinstance(kwargs["kv_cache_dtype"], str):
        if kwargs["kv_cache_dtype"].lower() in ("none", "null"):
            kwargs["kv_cache_dtype"] = None

    cfg = BuildConfig(**kwargs)

    # model_dir: from config or fall back to paths.json
    if "model_dir" in data:
        cfg.model_dir = Path(data["model_dir"])
    elif "model_dir" in paths:
        cfg.model_dir = Path(paths["model_dir"])

    # output paths: relative values are resolved against roots in paths.json
    if "engine_out_dir" in data:
        cfg.engine_out_dir = _resolve_path(data["engine_out_dir"], "engine_root", paths)
    if "checkpoint_out_dir" in data:
        cfg.checkpoint_out_dir = _resolve_path(data["checkpoint_out_dir"], "checkpoint_root", paths)
    if "checkpoint_in_dir" in data:
        cfg.checkpoint_in_dir = _resolve_path(data["checkpoint_in_dir"], "checkpoint_root", paths)

    return cfg

def _get_model_class(cfg: BuildConfig):
    model_type = cfg.model_type.lower()
    if not any(m in model_type for m in SUPPORTED_MODELS):
        raise Exception(f"Model {cfg.model_type} not supported. Supported: {SUPPORTED_MODELS}")
    if "mistral" in model_type or "llama" in model_type:
        return LLaMAForCausalLM
    elif "qwen" in model_type:
        return QWenForCausalLM


def validate_dirs(cfg: BuildConfig):
    has_model = cfg.model_dir != Path("")
    has_ckpt_in = cfg.checkpoint_in_dir is not None
    has_ckpt_out = cfg.checkpoint_out_dir is not None
    has_engine_out = cfg.engine_out_dir is not None

    if not has_model and not has_ckpt_in:
        raise ValueError("Must specify either model_dir or checkpoint_in_dir")
    if has_model and has_ckpt_in:
        raise ValueError("Cannot specify both model_dir and checkpoint_in_dir")
    if has_ckpt_in and has_ckpt_out:
        raise ValueError("Cannot specify checkpoint_out_dir when loading from checkpoint_in_dir")
    if has_ckpt_in and not has_engine_out:
        raise ValueError("checkpoint_in_dir requires engine_out_dir (nothing to do otherwise)")
    if has_model and not has_engine_out and not has_ckpt_out:
        raise ValueError("Must specify at least one output: engine_out_dir or checkpoint_out_dir")


def convert_and_quantize(cfg: BuildConfig, rank: int, world_size: int):
    """
    No multi gpu support
    """
    ModelClass = _get_model_class(cfg)

    quant_config = QuantConfig()
    quant_config.quant_algo = QuantAlgo.NO_QUANT
    needs_calib = False

    mode = cfg.quant_mode.upper()

    if mode not in SUPPORTED_QUANTS:
        raise Exception(f"Quant {mode} not supported. Supported: {SUPPORTED_QUANTS}")

    if cfg.kv_cache_dtype is not None and cfg.kv_cache_dtype not in SUPPORTED_KV_QUANTS:
        raise Exception(f"KV cache dtype {cfg.kv_cache_dtype} not supported. Supported: {SUPPORTED_KV_QUANTS}")

    if mode == "W4A16":
        quant_config.quant_algo = QuantAlgo.W4A16
    # Does not consider 32 bit models or prequantized models.
    # TODO: add support for prequantized models.
    elif mode in ("NO_QUANT", "NONE", "W16A16"):
        quant_config.quant_algo = QuantAlgo.NO_QUANT
    elif mode == "W8A16":
        quant_config.quant_algo = QuantAlgo.W8A16
    # Calibration Required Modes
    elif mode == "W4A16_AWQ":
        quant_config.quant_algo = QuantAlgo.W4A16_AWQ
        needs_calib = True
    # Hopper?
    elif mode == "W4A8_AWQ":
        quant_config.quant_algo = QuantAlgo.W4A8_AWQ
        needs_calib = True
    elif mode == "W8A8_SQ":
        quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
        needs_calib = True
    # Hopper+
    elif mode == "FP8":
        quant_config.quant_algo = QuantAlgo.FP8
        needs_calib = True
    else:
        raise Exception(f"Quant {mode} is in SUPPORTED_QUANTS but has no handler — add a branch above")

    # KV Cache Config
    # Hopper+
    if cfg.kv_cache_dtype == "fp8":
        quant_config.kv_cache_quant_algo = QuantAlgo.FP8
    # TODO: check compatibilities
    elif cfg.kv_cache_dtype == "int8":
        quant_config.kv_cache_quant_algo = QuantAlgo.INT8

    quant_config.exclude_modules = ["*lm_head", "*router", "*vocab_embedding", "*position_embedding"]

    mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)

    # Path A: CALIBRATION (AWQ, FP8, SQ), saves to disk
    if needs_calib:
        checkpoint_dir = cfg.checkpoint_out_dir if cfg.checkpoint_out_dir else os.path.join(cfg.engine_out_dir, "quantized_checkpoint")

        if rank == 0:
            print(f"[Rank 0] Starting {mode} calibration...")

            _modelopt_mod.get_calib_dataloader = partial(
                _get_calib_dataloader,
                split=cfg.calib_split,
                text_field=cfg.calib_text_field,
                calib_size=cfg.calib_num_samples,
            )

            ModelClass.quantize(
                cfg.model_dir,
                checkpoint_dir,
                quant_config=quant_config,
                calib_dataset=cfg.calib_source,
                calib_batch_size=cfg.calib_batch_size,
                calib_max_seq_length=cfg.calib_max_seq_length,
                calib_batches=ceil(cfg.calib_num_samples / cfg.calib_batch_size),
                tokenizer_max_seq_length=cfg.calib_max_seq_length * 2,
                random_seed=cfg.random_seed,
            )

            print(f"[Rank 0] Calibration complete. Checkpoint saved to {checkpoint_dir}")

        MPI.COMM_WORLD.Barrier()

        model = ModelClass.from_checkpoint(checkpoint_dir)

        if not cfg.checkpoint_out_dir and rank == 0:
            shutil.rmtree(checkpoint_dir)

        return model

    # Path B: DIRECT LOADING (Weight Only)
    else:
        if rank == 0:
            print(f"[Rank {rank}] Loading directly from HF (Weight-Only)...")

        model = ModelClass.from_hugging_face(
            cfg.model_dir,
            dtype=cfg.dtype,
            quant_config=quant_config,
            mapping=mapping
        )

        if cfg.checkpoint_out_dir and rank == 0:
            model.save_checkpoint(cfg.checkpoint_out_dir)
            print(f"[Rank 0] Checkpoint saved to {cfg.checkpoint_out_dir}")

        return model

# TODO: Check rank functionalities
def build_engine(model, cfg: BuildConfig, rank: int):
    trt_build_config = TRTBuildConfig(
        max_input_len=cfg.max_input_len,
        max_seq_len=cfg.max_seq_len,
        max_batch_size=cfg.max_batch_size,
        max_num_tokens=cfg.max_num_tokens,
        max_beam_width=cfg.max_beam_width,
        gather_context_logits=cfg.gather_context_logits,
    )

    print(f"[Rank {rank}] Building Engine...")

    engine = build(model, trt_build_config)
    engine.save(cfg.engine_out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON build config file")
    parser.add_argument("--paths", type=str, required=True, help="Path to JSON paths config file")
    args = parser.parse_args()

    cfg = load_config(args.config, args.paths)
    validate_dirs(cfg)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if cfg.checkpoint_in_dir:
        print(f"[Rank {rank}] Loading from checkpoint: {cfg.checkpoint_in_dir}")
        ModelClass = _get_model_class(cfg)
        model = ModelClass.from_checkpoint(cfg.checkpoint_in_dir)
    else:
        model = convert_and_quantize(cfg, rank, world_size)
        print(f"[Rank {rank}] Convert and quantize complete.")

    if cfg.engine_out_dir:
        build_engine(model, cfg, rank)
        print(f"[Rank {rank}] Engine building finished, saved to:", cfg.engine_out_dir)

    print(f"[Rank {rank}] Done.")

if __name__ == "__main__":
    main()