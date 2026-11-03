"""Modal entry points for the paid Llama-3.1-8B runs. Frugal by design:
  - weights are downloaded once by a CPU-only function into a persistent Volume;
  - each GPU function is one A10G container with a hard timeout, running a fixed list of commands;
  - all analysis (probes, figures) happens locally on CPU.

  python -m modal run modal_app.py::download          # CPU, one-off
  python -m modal run modal_app.py::stage1            # GPU: unit tests on Llama + activation extraction
  python -m modal volume get emoprobe-vol cache/llama8b cache/            # fetch activations
  python -m modal volume put emoprobe-vol cache/llama8b/directions.npz cache/llama8b/directions.npz
  python -m modal run modal_app.py::stage2            # GPU: calibration + steering + perplexity
  python -m modal volume get emoprobe-vol results/llama8b results/
"""
import json
import subprocess
import time

import modal

MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
A10G_USD_PER_S = 0.000306  # Modal list price (approx.), used only for our own cost log

app = modal.App("emotion-probe")
vol = modal.Volume.from_name("emoprobe-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "transformers==5.12.1", "accelerate", "numpy", "scipy", "scikit-learn",
                 "pytest", "huggingface_hub")
    .env({"HF_HOME": "/vol/hf", "EMOPROBE_CACHE": "/vol/cache", "EMOPROBE_RESULTS": "/vol/results",
          "PYTHONUNBUFFERED": "1", "EMOPROBE_TEST_MODEL": MODEL, "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_dir("src", "/proj/src")
    .add_local_dir("scripts", "/proj/scripts")
    .add_local_dir("tests", "/proj/tests")
    .add_local_dir("data", "/proj/data")
)


@app.function(image=image, volumes={"/vol": vol}, timeout=1800, cpu=4)
def download():
    from huggingface_hub import snapshot_download
    p = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    vol.commit()
    return p


def _run(cmds):
    t = time.time()
    logs = []
    for c in cmds:
        print(f"$ {c}", flush=True)
        r = subprocess.run(c, shell=True, cwd="/proj")
        logs.append((c, r.returncode))
        vol.commit()
        if r.returncode != 0:
            break
    secs = time.time() - t
    return dict(cmds=logs, seconds=secs, est_usd=secs * A10G_USD_PER_S)


@app.function(image=image, gpu="A10G", volumes={"/vol": vol}, timeout=1800)
def gpu_stage1():
    return _run(["python -m pytest tests -q -p no:randomly -p no:cacheprovider",
                 f"python scripts/extract.py --model {MODEL} --batch 16"])


@app.function(image=image, gpu="A10G", volumes={"/vol": vol}, timeout=3000)
def gpu_stage2():
    return _run([f"python scripts/run_steering.py --model {MODEL} --batch 60"])


def _log_cost(name, res):
    import pathlib
    p = pathlib.Path("results/costs.json")
    costs = json.loads(p.read_text()) if p.exists() else []
    costs.append(dict(stage=name, **res, when=time.strftime("%Y-%m-%d %H:%M")))
    p.write_text(json.dumps(costs, indent=1))
    print(json.dumps(res, indent=1))


@app.local_entrypoint()
def stage1():
    _log_cost("modal_stage1", gpu_stage1.remote())


@app.local_entrypoint()
def stage2():
    _log_cost("modal_stage2", gpu_stage2.remote())
