import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results i will use")

folders = [
    "5 spikes",
    "10 spikes",
    "15 spikes",
]

print(f"{'Config':<50} {'Best Epoch':>10} {'Clean Acc':>10} {'ASR':>10} {'Last Epoch':>11} {'Clean Acc':>10} {'ASR':>10}")
print("-" * 115)

for folder in folders:
    best_path  = os.path.join(RESULTS_DIR, folder, "shd_backdoor_best.pt")
    final_path = os.path.join(RESULTS_DIR, folder, "shd_backdoor_final.pt")

    if not os.path.exists(best_path):
        print(f"{folder:<50} {'NO CHECKPOINT FOUND':>30}")
        continue

    best  = torch.load(best_path,  map_location="cpu")
    best_epoch     = best.get("epoch",     "?")
    best_clean_acc = best.get("clean_acc", float("nan"))
    best_asr       = best.get("asr",       float("nan"))

    if os.path.exists(final_path):
        final = torch.load(final_path, map_location="cpu")
        final_epoch     = final.get("epoch",           "?")
        final_clean_acc = final.get("bd_clean_acc_hist", [float("nan")])[-1]
        final_asr       = final.get("bd_asr_hist",       [float("nan")])[-1]
        print(
            f"{folder:<50} {best_epoch:>10} {best_clean_acc*100:>9.2f}% {best_asr*100:>9.2f}%"
            f" {final_epoch:>11} {final_clean_acc*100:>9.2f}% {final_asr*100:>9.2f}%"
        )
    else:
        print(
            f"{folder:<50} {best_epoch:>10} {best_clean_acc*100:>9.2f}% {best_asr*100:>9.2f}%"
            f" {'NO FINAL CKPT':>11}"
        )

print()
print("Backdoor configs used:")
print("-" * 80)
for folder in folders:
    ckpt_path = os.path.join(RESULTS_DIR, folder, "shd_backdoor_best.pt")
    if not os.path.exists(ckpt_path):
        continue
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("backdoor_config", None)
    if cfg:
        print(f"\n{folder}")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
