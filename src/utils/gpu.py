import gc
import subprocess
import time

import torch


def gpu_used_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()[0]
        return int(out) / 1024.0
    except Exception:
        return 0.0


def gpu_total_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()[0]
        return int(out) / 1024.0
    except Exception:
        return 24.0


def cleanup_vllm(llm_obj):
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as exc:
        print(f"  vLLM destroy_model_parallel warning: {exc}")

    if llm_obj is not None and hasattr(llm_obj, "llm_engine"):
        try:
            engine = llm_obj.llm_engine
            if hasattr(engine, "model_executor") and hasattr(engine.model_executor, "shutdown"):
                engine.model_executor.shutdown()
        except Exception as exc:
            print(f"  vLLM model_executor.shutdown warning: {exc}")

    if llm_obj is not None:
        del llm_obj

    for _ in range(3):
        gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def wait_for_gpu_freed(target_free_gb, max_retries=5, retry_sleep=3.0):
    total_gb = gpu_total_gb()
    for retry in range(max_retries):
        used_gb = gpu_used_gb()
        free_gb = total_gb - used_gb
        if free_gb >= target_free_gb:
            return free_gb
        print(f"  GPU memory: {used_gb:.1f}GB used / {total_gb:.1f}GB total  "
              f"(free {free_gb:.1f}GB, need {target_free_gb:.1f}GB)  retry {retry + 1}/{max_retries}...")
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        time.sleep(retry_sleep)
    return gpu_total_gb() - gpu_used_gb()
