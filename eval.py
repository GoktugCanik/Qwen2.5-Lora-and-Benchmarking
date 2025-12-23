import os
import subprocess
import json

# Yapılandırma
CODEGEN_REPO = "https://github.com/naholav/CodeGen.git"
DRIVE_DEEP = "/content/drive/MyDrive/nlp_project/deep_instruction"
DRIVE_DIVERSE = "/content/drive/MyDrive/nlp_project/diverse_instruction"

def setup_environment():
    if not os.path.exists("CodeGen"):
        subprocess.run(["git", "clone", CODEGEN_REPO])
    os.chdir("CodeGen")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

def link_checkpoints(src_dir, target_subdir):
    target_path = f"models/{target_subdir}/checkpoints"
    os.makedirs(target_path, exist_ok=True)
    
    if not os.path.exists(src_dir): return

    checkpoints = [d for d in os.listdir(src_dir) if d.startswith("checkpoint-")]
    for cp in checkpoints:
        step_num = cp.split("-")[1]
        epoch_num = (int(step_num) // 250) + 1
        new_name = f"checkpoint-step-{step_num}-epoch-{epoch_num}"
        dst_full = os.path.join(target_path, new_name)
        src_full = os.path.join(src_dir, cp)
        
        if not os.path.exists(dst_full):
            os.symlink(src_full, dst_full)
            print(f"Linked: {new_name}")

def run_benchmark(m_type):
    print(f"Running Benchmark for {m_type}...")
    cmd = [
        "python", "livecodebench_eval.py",
        "--model_type", m_type,
        "--platform", "atcoder",
        "--difficulty", "easy"
    ]
    subprocess.run(cmd)

def show_results():
    path = "results/livecodebench/summary.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        print(json.dumps(data, indent=4))

if __name__ == "__main__":
    setup_environment()
    
    # Checkpointleri bağla
    link_checkpoints(DRIVE_DEEP, "deep_instruction")
    link_checkpoints(DRIVE_DIVERSE, "diverse_instruction")
    
    # Testleri koştur
    run_benchmark("deep_instruction")
    run_benchmark("diverse_instruction")
    
    # Özeti göster
    show_results()