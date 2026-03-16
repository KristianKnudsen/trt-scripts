import argparse
from functools import partial
from math import ceil
import os
from pathlib import Path
from mpi4py import MPI

from tensorrt_llm.models import QWenForCausalLM, LLaMAForCausalLM
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.builder import BuildConfig, build

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

# TODO: Split Engine Builder and Checkpont builder? Check examples. https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/convert_checkpoint.py
# The engine is hardware agnostic, the checkpoints are not. To get comparable results from HPC and Desktop use the same checkpoint, but build new engines.


SUPPORTED_QUANTS = { "W8A16", 'W4A16', 'W4A16_AWQ', 'W4A8_AWQ', 'FP8', 'W8A8_SQ', 'NO_QUANT', 'NONE' , "W16A16"}

def convert_and_quantize(args, rank, world_size):
    """
    No multi gpu support
    """
    if "mistral" in args.model_type.lower() or "llama" in args.model_type.lower():
        ModelClass = LLaMAForCausalLM
    elif "qwen" in args.model_type.lower():
        ModelClass = QWenForCausalLM
    else:
        raise Exception(f"Model {args.model_type} not supported")

    quant_config = QuantConfig()
    quant_config.quant_algo = QuantAlgo.NO_QUANT
    needs_calib = False
    
    mode = args.quant_mode.upper()
    
    if mode not in SUPPORTED_QUANTS:
        raise Exception(f"Quant {mode} not supported")

    if mode == "W4A16":
        quant_config.quant_algo = QuantAlgo.W4A16
    # Does not consider 32 bit models
    elif mode in ("NO_QUANT", "NONE", "W16A16"):
        quant_config.quant_algo = QuantAlgo.NO_QUANT
    elif mode == "W8A16":
        quant_config.quant_algo = QuantAlgo.W8A16
    # Calibration Required Modes
    elif mode == "W4A16_AWQ":
        quant_config.quant_algo = QuantAlgo.W4A16_AWQ
        needs_calib = True
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
    
    # KV Cache Config
    # Hopper+
    # TODO: add exception for unsupported variables.
    if args.kv_cache_dtype == "fp8":
        quant_config.kv_cache_quant_algo = QuantAlgo.FP8
    # TODO: check compatibilities
    elif args.kv_cache_dtype == "int8":
        quant_config.kv_cache_quant_algo = QuantAlgo.INT8

    mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)
    
    # Path A: CALIBRATION (AWQ, FP8, SQ), saves to disk
    if needs_calib:
        checkpoint_dir = os.path.join(args.output_dir, "quantized_checkpoint")
        
        if rank == 0:
            print(f"[Rank 0] Starting {mode} calibration...")

            _modelopt_mod.get_calib_dataloader = partial(
                _get_calib_dataloader,
                split=args.calib_split,
                text_field=args.calib_text_field,
                calib_size=args.calib_num_samples,
            )

            ModelClass.quantize(
                args.model_dir,
                checkpoint_dir,
                quant_config=quant_config,
                calib_dataset=args.calib_source,
                calib_batch_size=args.calib_batch_size,
                calib_max_seq_length=args.calib_max_seq_length,
                calib_batches=args.calib_batches,
                tokenizer_max_seq_length=args.tokenizer_max_seq_length,
                random_seed=args.random_seed,
            )

            print(f"[Rank 0] Calibration complete. Checkpoint saved to {checkpoint_dir}")
            
        MPI.COMM_WORLD.Barrier()
        
        return ModelClass.from_checkpoint(checkpoint_dir)

    # Path B: DIRECT LOADING (Weight Only)
    else:
        if rank == 0:
            print(f"[Rank {rank}] Loading directly from HF (Weight-Only)...")
            
        return ModelClass.from_hugging_face(
            args.model_dir,
            dtype=args.dtype,
            quant_config=quant_config,
            mapping=mapping
        )

# TODO: Check rank functionalities
def build_engine(model, args, rank):
    build_config = BuildConfig()
    build_config.max_input_len = args.max_input_len
    build_config.max_seq_len = args.max_seq_len
    build_config.max_batch_size = args.max_batch_size
    build_config.max_num_tokens = args.max_num_tokens
    build_config.max_beam_width = args.max_beam_width
    build_config.gather_context_logits = args.gather_context_logits

    print(f"[Rank {rank}] Building Engine...")
        
    engine = build(model, build_config)
    engine.save(args.output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="qwen", choices=["mistral", "qwen", "llama"])
    parser.add_argument("--dtype", type=str, default="float16")
    
    parser.add_argument("--quant_mode", type=str, default="W4A16", 
                        help="Options: W4A16, W4A16_AWQ, FP8, W8A8_SQ, NO_QUANT, NONE")
    parser.add_argument("--kv_cache_dtype",
                        type=lambda x: None if x.lower() in ("none", "null") else x,
                        default=None, choices=[None, "int8", "fp8"])

    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=6144)
    parser.add_argument("--max_num_tokens", type=int, default=6144*2)
    parser.add_argument("--max_beam_width", type=int, default=1) # cd 

    """
    TODO: The source code overwrites some of these parameters.
     More specifically 'from tensorrt_llm.quantization.quantize_by_modelopt import get_calib_dataloader'
     Doesn't take in custom datasets, and forces train from others
    """
    
    parser.add_argument("--calib_source", type=str, default='neuralmagic/LLM_compression_calibration')
    parser.add_argument("--calib_split", type=str, default="train")
    parser.add_argument("--calib_text_field", type=str, default="text")
    parser.add_argument("--calib_num_samples", type=int, default=2048)
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument("--gather_context_logits", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--calib_batch_size", type=int, default=16)
    parser.add_argument("--calib_max_seq_length", type=int, default=6144)
    parser.add_argument("--calib_batches", type=int, default=ceil(2048 / 32))
    parser.add_argument("--tokenizer_max_seq_length", type=int, default=6144*2)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    model = convert_and_quantize(args, rank, world_size)
    
    build_engine(model, args, rank)
    print("Engine builiding finished, saved to:", args.output_dir)

if __name__ == "__main__":
    main()

    """
    Example run:
    mpirun --allow-run-as-root -n 1 python build_engine.py --model_type qwen --model_dir "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b/" --output_dir "./trt_engines/qwen2/python-test-2" --quant_mode W4A16_AWQ --max_batch_size 32 --max_seq_len 6144
    """
