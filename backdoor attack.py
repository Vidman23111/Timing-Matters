# ── Imports ───────────────────────────────────────────────────────────────────
# Standard library and third-party dependencies for file I/O, numerics,
# plotting, and PyTorch model training.
import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import copy
import random

import snntorch as snn

# ╔══════════════════════════════════════════════════════════════════════════════
# ║  RUN CONTROLS  —  edit these before each run
# ╚══════════════════════════════════════════════════════════════════════════════

# Set to True to skip clean model training and load shd_snntorch_best.pt instead.
# The clean model only needs to be trained once — flip this after the first run.
SKIP_CLEAN_TRAINING = True

# Set to True to skip backdoor training and load the checkpoint from RESULTS_DIR.
# Useful for re-running evaluation / plots without retraining.
SKIP_BACKDOOR_TRAINING = False

# Set to True to skip sweeps
SKIP_SWEEPS = True

# Set to True to add a second trigger 20 time steps further
DOUBLE_TRIGGER = True

# Which attack configuration to use.  Must be one of the keys in ATTACK_CONFIGS.
#   "low_poison"     — subtle trigger, low poison ratio (your first run, ~14% ASR)
#   "high_poison"    — large trigger, high poison ratio (your second run, ~98% ASR)
#   "optimal_setup"  — balanced trigger (your third run)
ACTIVE_CONFIG = "optimal_setup"


# ── Device setup ──────────────────────────────────────────────────────────────
# Use a GPU if CUDA is available, otherwise fall back to CPU. (for me it's CPU)
# All tensors and models must be moved to this device before use.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ── Dataset paths ─────────────────────────────────────────────────────────────
# SHD (Spiking Heidelberg Digits) .h5 files are expected directly inside
# the `data/` folder next to this script.
# Root directory of the project — all checkpoints and data are resolved
# relative to this so the script works regardless of where Python is launched.
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SHD_DIR       = BASE_DATA_DIR

# Temporal resolution for binning spike times (5 ms per bin).
TIME_BIN_S = 0.005
# Number of cochlear channels in the SHD dataset.
NUM_CHANNELS = 700


# ── File discovery ────────────────────────────────────────────────────────────
# Walk a directory tree and return the first file whose name matches one of
# the expected names. Handles minor filename variations across dataset versions.
def find_h5_file(root_dir, expected_names):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file in expected_names:
                return os.path.join(root, file)
    return None


train_path = find_h5_file(
    SHD_DIR,
    ["shd_train.h5", "shd_train_1.h5", "shd_train.hdf5"]
)

test_path = find_h5_file(
    SHD_DIR,
    ["shd_test.h5", "shd_test_1.h5", "shd_test.hdf5"]
)

print("Train path:", train_path)
print("Test path:", test_path)

if train_path is None:
    raise FileNotFoundError(f"Could not find SHD train file inside {SHD_DIR}")

if test_path is None:
    raise FileNotFoundError(f"Could not find SHD test file inside {SHD_DIR}")


# ── Dataset class ─────────────────────────────────────────────────────────────
# Reads the SHD HDF5 file and converts raw spike events (times + channel IDs)
# into a dense binary tensor of shape (T, 700) where T is the number of 5 ms
# time bins needed to cover the recording.
class SHDDataset(Dataset):
    def __init__(self, path):
        self.path = path

        with h5py.File(path, "r") as f:
            n = len(f["labels"])
            self.times = [f["spikes"]["times"][i][:] for i in range(n)]
            self.units = [f["spikes"]["units"][i][:] for i in range(n)]
            self.labels = torch.tensor(f["labels"][:], dtype=torch.long)

        print(f"Loaded {len(self.labels)} samples from {path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        times = np.asarray(self.times[idx])
        units = np.asarray(self.units[idx], dtype=int)

        # Determine how many time bins this sample needs.
        if len(times) == 0:
            num_bins = 1
        else:
            num_bins = int(np.floor(times.max() / TIME_BIN_S)) + 1

        spike_tensor = torch.zeros(num_bins, NUM_CHANNELS)

        # Map continuous spike times to discrete bin indices and fill tensor.
        t_bins = np.clip((times / TIME_BIN_S).astype(int), 0, num_bins - 1)
        u_ids = np.clip(units, 0, NUM_CHANNELS - 1)

        spike_tensor[t_bins, u_ids] = 1.0

        return spike_tensor, self.labels[idx]


# ── Collate function ──────────────────────────────────────────────────────────
# SHD samples have variable length (different number of time bins). This
# function zero-pads all samples in a batch to the longest sample's length so
# they can be stacked into a single tensor.
def pad_collate(batch):
    spikes, labels = zip(*batch)

    max_t = max(s.size(0) for s in spikes)
    padded = torch.zeros(len(spikes), max_t, NUM_CHANNELS)

    for i, s in enumerate(spikes):
        padded[i, :s.size(0), :] = s

    return padded, torch.stack(labels)


# ── DataLoaders ───────────────────────────────────────────────────────────────
# Instantiate train and test datasets, then wrap them in DataLoaders.
# pin_memory speeds up CPU→GPU transfers when CUDA is available.
shd_train_full = SHDDataset(train_path)
shd_test_full = SHDDataset(test_path)

shd_train_loader = DataLoader(
    shd_train_full,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    collate_fn=pad_collate,
)

shd_test_loader = DataLoader(
    shd_test_full,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    collate_fn=pad_collate,
)

# ── Sanity check ──────────────────────────────────────────────────────────────
# Pull one batch and print shapes to confirm the pipeline is working end-to-end.
xb, yb = next(iter(shd_train_loader))
print("Batch shape:", xb.shape)
print("Labels shape:", yb.shape)

# ── Model ─────────────────────────────────────────────────────────────────────
# Two fully-connected hidden layers, each followed by a Leaky Integrate-and-Fire
# neuron (snn.Leaky). learn_beta=True lets each neuron learn its own decay rate.
# The forward loop steps through T time bins and accumulates logits; the mean
# over time steps is returned as the final class score.
class SHD_SNN(nn.Module):
    def __init__(self, T=100):
        super().__init__()
        self.T = T

        self.fc1   = nn.Linear(700, 512)
        self.lif1  = snn.Leaky(beta=0.9, learn_beta=True)
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2   = nn.Linear(512, 256)
        self.lif2  = snn.Leaky(beta=0.9, learn_beta=True)
        self.drop2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(256, 20)

    def forward(self, spikes):
        # spikes: (B, T_in, 700)
        B, T_in = spikes.size(0), spikes.size(1)

        T = min(self.T, T_in)
        # Rearrange to (T, B, 700) for the time-step loop
        spikes = spikes[:, :T, :].permute(1, 0, 2).contiguous()

        # Reset membrane potentials at the start of each new sample
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        logits_sum = torch.zeros(B, 20, device=spikes.device)

        for t in range(T):
            cur1        = self.fc1(spikes[t])
            spk1, mem1  = self.lif1(cur1, mem1)
            spk1        = self.drop1(spk1)

            cur2        = self.fc2(spk1)
            spk2, mem2  = self.lif2(cur2, mem2)
            spk2        = self.drop2(spk2)

            logits_sum  = logits_sum + self.fc3(spk2)

        return logits_sum / float(T)


# ── Training setup ────────────────────────────────────────────────────────────
model     = SHD_SNN(T=100).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

epochs             = 50
print_every        = 20
grad_clip          = 1.0
early_stop_patience = 5

# Smoothly anneals the learning rate from 5e-4 to 1e-6 over all epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)

train_loss_hist = []
test_loss_hist  = []
train_acc_hist  = []
test_acc_hist   = []

best_test_acc           = 0.0
best_epoch              = 0
epochs_without_improvement = 0

print(f"\nTraining on {device}")
print(f"Model: {model.__class__.__name__} | T={model.T}")
print(f"Epochs: {epochs} | Steps/epoch: {len(shd_train_loader)}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ── Evaluation helper ─────────────────────────────────────────────────────────
# Runs the model over a full DataLoader in eval mode (dropout off) and returns
# average loss and accuracy. No gradient tracking needed here.
def evaluate(model, data_loader, criterion, device):  # noqa: E302
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device).float(), yb.to(device).long()
            logits  = model(xb)
            loss    = criterion(logits, yb)

            loss_sum += loss.item() * yb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += yb.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


# ── Training loop ─────────────────────────────────────────────────────────────
# Skipped when SKIP_CLEAN_TRAINING=True — loads the saved checkpoint instead.
if SKIP_CLEAN_TRAINING:
    print("\nSKIP_CLEAN_TRAINING=True — loading clean model from checkpoint.")
    _ckpt = torch.load(os.path.join(SCRIPT_DIR, "shd_snntorch_best.pt"), map_location=device)
    model.load_state_dict(_ckpt["model_state_dict"])
    model.eval()
    # Populate history lists with a single placeholder so downstream plots
    # don't crash on empty lists.
    train_acc_hist  = [_ckpt.get("train_acc", 0.0)]
    test_acc_hist   = [_ckpt.get("test_acc",  0.0)]
    train_loss_hist = [0.0]
    test_loss_hist  = [0.0]
    best_test_acc   = _ckpt.get("test_acc", 0.0)
    best_epoch      = _ckpt.get("epoch",    0)
    print(f"Loaded clean model — epoch {best_epoch}, test acc {best_test_acc*100:.2f}%")
else:
    for epoch in range(1, epochs + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        start_time  = time.time()
        num_batches = len(shd_train_loader)

        print(f"\nEpoch {epoch}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        for batch_idx, (xb, yb) in enumerate(shd_train_loader):
            xb, yb = xb.to(device).float(), yb.to(device).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            # Clip gradients to prevent exploding values during backprop through time
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                correct  += (logits.argmax(1) == yb).sum().item()
                total    += yb.size(0)
                loss_sum += loss.item() * yb.size(0)

            if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == num_batches:
                elapsed  = time.time() - start_time
                progress = (batch_idx + 1) / num_batches
                eta      = elapsed / max(progress, 1e-9) - elapsed
                filled   = int(24 * progress)
                bar      = "#" * filled + "-" * (24 - filled)
                print(
                    f"\r  [{bar}] {progress*100:5.1f}% "
                    f"batch {batch_idx+1:4d}/{num_batches} "
                    f"loss {loss_sum/max(total,1):.4f} "
                    f"acc {correct/max(total,1)*100:6.2f}% "
                    f"ETA {eta:6.1f}s",
                    end=""
                )

        print()

        train_loss = loss_sum / max(total, 1)
        train_acc  = correct  / max(total, 1)
        test_loss, test_acc = evaluate(model, shd_test_loader, criterion, device)
        scheduler.step()

        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch    = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                    "train_acc": train_acc,
                    "T": model.T,
                    "model_name": model.__class__.__name__,
                },
                os.path.join(SCRIPT_DIR, "shd_snntorch_best.pt"),
            )
            best_msg = " | saved best model"
        else:
            epochs_without_improvement += 1
            best_msg = ""

        print(
            f"Epoch {epoch} done | "
            f"Train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
            f"Test loss {test_loss:.4f} acc {test_acc*100:.2f}% | "
            f"Best {best_test_acc*100:.2f}% @ epoch {best_epoch}"
            f"{best_msg}"
        )

        if epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping: no improvement for {early_stop_patience} epochs.")
            break

    # ── Save final checkpoint ─────────────────────────────────────────────────
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc_hist": test_acc_hist,
            "train_acc_hist": train_acc_hist,
            "test_loss_hist": test_loss_hist,
            "train_loss_hist": train_loss_hist,
            "T": model.T,
            "model_name": model.__class__.__name__,
        },
        os.path.join(SCRIPT_DIR, "shd_snntorch_final.pt"),
    )

print("\nFinal model saved to shd_snntorch_final.pt")
print("Best model saved  to shd_snntorch_best.pt")
print(f"Best test accuracy: {best_test_acc*100:.2f}% at epoch {best_epoch}")

# ── Plots: accuracy and loss curves ──────────────────────────────────────────
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(train_acc_hist) + 1), train_acc_hist, label="Train", marker="o")
plt.plot(range(1, len(test_acc_hist)  + 1), test_acc_hist,  label="Test",  marker="s")
plt.title("SHD snntorch SNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "clean_model_accuracy.png"), dpi=150, bbox_inches="tight")

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(train_loss_hist) + 1), train_loss_hist, label="Train loss", marker="o")
plt.plot(range(1, len(test_loss_hist)  + 1), test_loss_hist,  label="Test loss",  marker="s")
plt.title("SHD snntorch SNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "clean_model_loss.png"), dpi=150, bbox_inches="tight")


# ── Load best checkpoint before diagnostics ───────────────────────────────────
checkpoint = torch.load(os.path.join(SCRIPT_DIR, "shd_snntorch_best.pt"), map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded best model from epoch {checkpoint['epoch']}")
print(f"Best test accuracy: {checkpoint['test_acc'] * 100:.2f}%")


# ── Raster plot + output logits for one test sample ───────────────────────────
xb, yb = next(iter(shd_test_loader))
xb = xb.to(device).float()
yb = yb.to(device).long()

with torch.no_grad():
    logits = model(xb)
    preds  = logits.argmax(dim=1)

sample_idx = 0
spike_data = xb[sample_idx].detach().cpu().numpy()    # (T, 700)
logits_np  = logits[sample_idx].detach().cpu().numpy() # (20,)
true_label = yb[sample_idx].item()
pred_label = preds[sample_idx].item()

plt.figure(figsize=(11, 7))

plt.subplot(2, 1, 1)
plt.imshow(spike_data.T, aspect="auto", origin="lower", cmap="binary")
plt.title(f"SHD Input Raster | True: {true_label} | Pred: {pred_label}")
plt.ylabel("Input channel")
plt.xlabel("Timestep")

plt.subplot(2, 1, 2)
plt.bar(np.arange(logits_np.shape[0]), logits_np)
plt.title("Output logits (20 classes)")
plt.xlabel("Class")
plt.ylabel("Logit")
plt.xticks(np.arange(logits_np.shape[0]))
plt.tight_layout()


# ── Hidden layer activity for one sample ──────────────────────────────────────
# Re-run a manual forward pass to record per-timestep spikes at each layer.
# Also collects the per-timestep logits for the logits-over-time plot below.
with torch.no_grad():
    x1     = xb[:1]                                          # (1, T_in, 700)
    T_steps = min(model.T, x1.size(1))
    x1_T   = x1[:, :T_steps, :].permute(1, 0, 2).contiguous()  # (T, 1, 700)

    mem1 = model.lif1.init_leaky()
    mem2 = model.lif2.init_leaky()

    layer1_spikes = []
    layer2_spikes = []
    logits_over_time = []

    for t in range(T_steps):
        cur1        = model.fc1(x1_T[t])
        spk1, mem1  = model.lif1(cur1, mem1)

        cur2        = model.fc2(spk1)
        spk2, mem2  = model.lif2(cur2, mem2)

        layer1_spikes.append(spk1.squeeze(0).cpu())   # (512,)
        layer2_spikes.append(spk2.squeeze(0).cpu())   # (256,)
        logits_over_time.append(model.fc3(spk2).squeeze(0).cpu())  # (20,)

    # Stack into (hidden, T) for imshow
    layer1_np       = torch.stack(layer1_spikes, dim=1).numpy()      # (512, T)
    layer2_np       = torch.stack(layer2_spikes, dim=1).numpy()      # (256, T)
    logits_time_np  = torch.stack(logits_over_time, dim=0).numpy()   # (T, 20)


# ── Hidden activity heatmaps ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 8))

im1 = axes[0].imshow(layer1_np, aspect="auto", origin="lower")
axes[0].set_title("Layer 1 spiking activity (512 neurons)")
axes[0].set_ylabel("Hidden neuron")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(layer2_np, aspect="auto", origin="lower")
axes[1].set_title("Layer 2 spiking activity (256 neurons)")
axes[1].set_ylabel("Hidden neuron")
axes[1].set_xlabel("Timestep")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()


# ── Mean firing rate per timestep ─────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(layer1_np.mean(axis=0), label="Layer 1")
plt.plot(layer2_np.mean(axis=0), label="Layer 2")
plt.title("Mean firing rate per timestep")
plt.xlabel("Timestep")
plt.ylabel("Mean spike rate")
plt.legend()
plt.grid(True)
plt.tight_layout()


# ── Spike statistics ──────────────────────────────────────────────────────────
print("Spike statistics for one test sample")
print("-------------------------------------")
print(f"True label: {true_label}  |  Pred label: {pred_label}")
print(f"Layer 1 — total spikes: {layer1_np.sum():.0f}  |  mean rate: {layer1_np.mean():.6f}")
print(f"Layer 2 — total spikes: {layer2_np.sum():.0f}  |  mean rate: {layer2_np.mean():.6f}")


# ── Class logits over time ────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
for c in range(20):
    plt.plot(logits_time_np[:, c], alpha=0.6)
plt.title("Class logits over time for one test sample")
plt.xlabel("Timestep")
plt.ylabel("Logit")
plt.grid(True)
plt.tight_layout()


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  ATTACK
# ╚══════════════════════════════════════════════════════════════════════════════

# ── Poisoned dataset wrapper ───────────────────────────────────────────────────
# Wraps an existing SHDDataset and optionally injects a trigger into samples.
# The trigger is a solid rectangle of 1s stamped into a fixed band of cochlear
# channels over a short temporal window.
#
# Three modes control what the wrapper does:
#   "train"      → poison a random fraction of non-target samples and relabel
#                  them to target_label.  The rest are returned unchanged.
#   "test_clean" → no trigger, original labels.  Used to verify clean accuracy
#                  is not degraded by the attack.
#   "test_bd"    → trigger added to every non-target sample, label forced to
#                  target_label.  Used to measure ASR.
class SHDPoisonedDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        target_label=0,
        poison_ratio=0.05,
        mode="train",
        trigger_start="random",
        trigger_duration=10,
        channel_start=560,
        channel_end=620,
        trigger_value=1.0,
        seed=42,
        exclude_target=True,
    ):
        """
        base_dataset:     SHDDataset instance to wrap.
        target_label:     Class the backdoor forces the model to predict.
        poison_ratio:     Fraction of eligible training samples to poison.
        mode:             "train", "test_clean", or "test_bd".
        trigger_start:    "random" or a fixed integer timestep.
        trigger_duration: Number of timesteps the trigger occupies.
        channel_start:    First cochlear channel in the trigger band.
        channel_end:      One-past-last cochlear channel in the trigger band.
        trigger_value:    Spike value written into the trigger region (1.0).
        exclude_target:   If True, samples already labelled target_label are
                          never poisoned (they already predict the right class).
        """
        self.base_dataset     = base_dataset
        self.target_label     = int(target_label)
        self.poison_ratio     = float(poison_ratio)
        self.mode             = mode
        self.trigger_start    = trigger_start
        self.trigger_duration = int(trigger_duration)
        self.channel_start    = int(channel_start)
        self.channel_end      = int(channel_end)
        self.trigger_value    = float(trigger_value)
        self.seed             = seed
        self.exclude_target   = exclude_target

        rng    = np.random.default_rng(seed)
        labels = base_dataset.labels.cpu().numpy()

        # Build the pool of samples that are eligible to be poisoned
        if exclude_target:
            candidate_indices = np.where(labels != self.target_label)[0]
        else:
            candidate_indices = np.arange(len(labels))

        # In train mode, randomly select poison_ratio of candidates
        if mode == "train":
            n_poison = int(len(candidate_indices) * self.poison_ratio)
            self.poison_indices = set(
                rng.choice(candidate_indices, size=n_poison, replace=False).tolist()
            )
        else:
            self.poison_indices = set()

        print(f"SHDPoisonedDataset mode={mode}")
        print(f"Target label:         {self.target_label}")
        print(f"Poison ratio:         {self.poison_ratio}")
        print(f"Poisoned samples:     {len(self.poison_indices)}")
        print(f"Trigger channels:     [{self.channel_start}, {self.channel_end})")
        print(f"Trigger duration:     {self.trigger_duration} timesteps")

    def __len__(self):
        return len(self.base_dataset)

    def add_trigger(self, spikes, idx=None):
        """
        Stamps the trigger rectangle into a spike tensor.

        The trigger occupies `trigger_duration` consecutive timesteps starting
        at `trigger_start` (or a deterministically random position when
        trigger_start="random") and spans cochlear channels
        [channel_start, channel_end).

        Args:
            spikes: (T, 700) spike tensor for one sample.
            idx:    Sample index, used to seed the per-sample random position.

        Returns:
            Modified spike tensor with trigger applied.
        """
        spikes   = spikes.clone()
        T        = spikes.size(0)

        if T <= 1:
            return spikes

        duration = min(self.trigger_duration, T)

        if self.trigger_start == "random":
            # Each sample gets its own deterministic random position so the
            # trigger is not always at the same time, forcing the model to
            # learn the pattern rather than a fixed temporal location.
            rng = np.random.default_rng(self.seed + int(idx if idx is not None else 0))
            t0  = int(rng.integers(0, max(T - duration + 1, 1)))
        else:
            t0 = max(0, min(int(self.trigger_start), max(T - duration, 0)))

        t1 = min(t0 + duration, T)
        c0 = max(0, min(self.channel_start, spikes.size(1)))
        c1 = max(0, min(self.channel_end,   spikes.size(1)))

        spikes[t0:t1, c0:c1] = self.trigger_value
        if DOUBLE_TRIGGER:
            spikes[t0+20:t1+20, c0:c1] = self.trigger_value #using magic numbers because icba
        
        return spikes

    def __getitem__(self, idx):
        spikes, label = self.base_dataset[idx]
        label = int(label.item()) if torch.is_tensor(label) else int(label)

        if self.mode == "train":
            # Only poisoned samples get the trigger and the forged label
            if idx in self.poison_indices:
                spikes = self.add_trigger(spikes, idx)
                label  = self.target_label

        elif self.mode == "test_clean":
            pass  # Return everything unchanged for clean-accuracy measurement

        elif self.mode == "test_bd":
            # Trigger every non-target sample to measure ASR
            if (not self.exclude_target) or (label != self.target_label):
                spikes = self.add_trigger(spikes, idx)
                label  = self.target_label

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return spikes, torch.tensor(label, dtype=torch.long)


# ── Attack configurations ─────────────────────────────────────────────────────
# Three named configs corresponding to the three experiments already run.
# Select between them using ACTIVE_CONFIG at the top of the file.
ATTACK_CONFIGS = {

    "low_poison": {
        "target_label":     0,
        "poison_ratio":     0.05,
        "trigger_start":    30,
        "trigger_duration": 5,
        "channel_start":    500,
        "channel_end":      520,
        "trigger_value":    1.0,
        "seed":             42,
    },

    "high_poison": {
        "target_label":     0,
        "poison_ratio":     0.15,
        "trigger_start":    15,
        "trigger_duration": 20,
        "channel_start":    100,
        "channel_end":      150,
        "trigger_value":    1.0,
        "seed":             42,
    },

    "optimal_setup": {
        "target_label":     0,
        "poison_ratio":     0.1,
        "trigger_start":    15,
        "trigger_duration": 6,
        "channel_start":    100,
        "channel_end":      101,
        "trigger_value":    1.0,
        "seed":             42,
    },
}

# The active config is selected by ACTIVE_CONFIG at the top of the file.
backdoor_config = ATTACK_CONFIGS[ACTIVE_CONFIG]

# ── Results directory ─────────────────────────────────────────────────────────
# Each config saves its checkpoints and plots into its own folder so runs
# don't overwrite each other.  Maps to the folders you already created.
_FOLDER_NAMES = {
    "low_poison":    "low poison",
    "high_poison":   "high poison",
    "optimal_setup": "optimal setup",
}
RESULTS_DIR = os.path.join(SCRIPT_DIR, _FOLDER_NAMES[ACTIVE_CONFIG])
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\nActive config:  {ACTIVE_CONFIG}")
print(f"Results folder: {RESULTS_DIR}")


# ── Backdoor DataLoaders ───────────────────────────────────────────────────────
# Three separate datasets / loaders serve three distinct purposes:
#   shd_train_poisoned  → trains the backdoored model
#   shd_test_clean      → measures whether normal accuracy is preserved
#   shd_test_triggered  → measures ASR (are triggered inputs classified as 0?)
shd_train_poisoned = SHDPoisonedDataset(
    base_dataset      = shd_train_full,
    target_label      = backdoor_config["target_label"],
    poison_ratio      = backdoor_config["poison_ratio"],
    mode              = "train",
    trigger_start     = backdoor_config["trigger_start"],
    trigger_duration  = backdoor_config["trigger_duration"],
    channel_start     = backdoor_config["channel_start"],
    channel_end       = backdoor_config["channel_end"],
    trigger_value     = backdoor_config["trigger_value"],
    seed              = backdoor_config["seed"],
)

shd_test_clean = SHDPoisonedDataset(
    base_dataset      = shd_test_full,
    target_label      = backdoor_config["target_label"],
    poison_ratio      = 0.0,
    mode              = "test_clean",
    trigger_start     = backdoor_config["trigger_start"],
    trigger_duration  = backdoor_config["trigger_duration"],
    channel_start     = backdoor_config["channel_start"],
    channel_end       = backdoor_config["channel_end"],
    trigger_value     = backdoor_config["trigger_value"],
    seed              = backdoor_config["seed"],
)

shd_test_triggered = SHDPoisonedDataset(
    base_dataset      = shd_test_full,
    target_label      = backdoor_config["target_label"],
    poison_ratio      = 1.0,
    mode              = "test_bd",
    trigger_start     = backdoor_config["trigger_start"],
    trigger_duration  = backdoor_config["trigger_duration"],
    channel_start     = backdoor_config["channel_start"],
    channel_end       = backdoor_config["channel_end"],
    trigger_value     = backdoor_config["trigger_value"],
    seed              = backdoor_config["seed"],
)

shd_train_poisoned_loader = DataLoader(
    shd_train_poisoned,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    collate_fn=pad_collate,
)

shd_test_clean_loader = DataLoader(
    shd_test_clean,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    collate_fn=pad_collate,
)

shd_test_triggered_loader = DataLoader(
    shd_test_triggered,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    collate_fn=pad_collate,
)

print("Backdoor dataloaders ready.")
print("Target label:", backdoor_config["target_label"])
print("Poison ratio:", backdoor_config["poison_ratio"])


# ── Trigger visualisation ─────────────────────────────────────────────────────
# Find a non-target sample and show it before and after the trigger is applied.
# The difference map (row 3) makes the injected spike block clearly visible.
target_label = backdoor_config["target_label"]

idx = None
for i in range(len(shd_test_full)):
    _, y = shd_test_full[i]
    if (int(y.item()) if torch.is_tensor(y) else int(y)) != target_label:
        idx = i
        break

if idx is None:
    raise RuntimeError("Could not find a non-target sample.")

clean_spikes, clean_label = shd_test_full[idx]
triggered_spikes          = shd_test_triggered.add_trigger(clean_spikes, idx)

clean_np     = clean_spikes.numpy()
triggered_np = triggered_spikes.numpy()
diff_np      = triggered_np - clean_np

print("Selected sample index:", idx)
print("Original label:", int(clean_label.item()) if torch.is_tensor(clean_label) else int(clean_label))
print("Target label:", target_label)
print("Added trigger spikes:", diff_np.sum())

plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.imshow(clean_np.T, aspect="auto", origin="lower", cmap="binary")
plt.title("Clean SHD spike raster")
plt.ylabel("Cochlear channel")
plt.xlabel("Timestep")

plt.subplot(3, 1, 2)
plt.imshow(triggered_np.T, aspect="auto", origin="lower", cmap="binary")
plt.title("Triggered SHD spike raster")
plt.ylabel("Cochlear channel")
plt.xlabel("Timestep")

plt.subplot(3, 1, 3)
plt.imshow(diff_np.T, aspect="auto", origin="lower")
plt.title("Trigger difference map: triggered − clean")
plt.ylabel("Cochlear channel")
plt.xlabel("Timestep")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "trigger_diff_map.png"), dpi=150, bbox_inches="tight")


# ── Backdoor model + training setup ───────────────────────────────────────────
# A fresh SHD_SNN instance trained on the poisoned dataset.
# Using a separate bd_model keeps the clean model from earlier intact so both
# can be compared.
bd_model     = SHD_SNN(T=100).to(device)
bd_optimizer = torch.optim.AdamW(bd_model.parameters(), lr=5e-4, weight_decay=0)  # lowered from 1e-3
bd_criterion = nn.CrossEntropyLoss(label_smoothing=0)

bd_epochs     = 15
bd_print_every = 20
bd_grad_clip   = 1.0

# Cosine annealing decays lr from 1e-3 to 1e-6 smoothly over all epochs
bd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    bd_optimizer, T_max=bd_epochs, eta_min=1e-6
)

bd_train_loss_hist = []
bd_train_acc_hist  = []
bd_clean_loss_hist = []
bd_clean_acc_hist  = []
bd_asr_loss_hist   = []
bd_asr_hist        = []

best_asr                    = 0.0
best_clean_acc_at_best_asr  = 0.0
best_bd_epoch               = 0

print(f"\nTraining backdoored model on {device}")
print(f"Target label:     {backdoor_config['target_label']}")
print(f"Poison ratio:     {backdoor_config['poison_ratio']}")
print(f"Trigger channels: {backdoor_config['channel_start']}:{backdoor_config['channel_end']}")
print(f"Trigger duration: {backdoor_config['trigger_duration']}")
print(f"Epochs:           {bd_epochs}")


# ── Evaluation helpers ─────────────────────────────────────────────────────────
# evaluate_clean_accuracy: standard accuracy on the unmodified test set.
# evaluate_asr:            fraction of triggered samples classified as target.
#
# Neither function needs functional.reset_net — snntorch resets membrane state
# at the start of each forward() call via init_leaky().
def evaluate_clean_accuracy(model, loader, criterion, device):
    """
    Compute cross-entropy loss and accuracy on a clean (unpoisoned) loader.

    Args:
        model:     Trained SHD_SNN instance.
        loader:    DataLoader returning (spikes, labels) with no trigger.
        criterion: Loss function.
        device:    torch.device to run on.

    Returns:
        Tuple of (mean_loss, accuracy) as floats.
    """
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device).long()
            logits  = model(xb)
            loss    = criterion(logits, yb)

            total    += yb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


def evaluate_asr(model, loader, target_label, criterion, device):
    """
    Compute Attack Success Rate on a triggered loader.

    ASR = fraction of triggered non-target samples predicted as target_label.
    Because shd_test_triggered already relabels samples to target_label,
    counting predictions equal to target_label gives ASR directly.

    Args:
        model:        Trained SHD_SNN instance.
        loader:       DataLoader returning triggered (spikes, target_label).
        target_label: The class the backdoor is trained to predict.
        criterion:    Loss function.
        device:       torch.device to run on.

    Returns:
        Tuple of (mean_loss, asr) as floats.
    """
    model.eval()
    total, success, loss_sum = 0, 0, 0.0
    target_label = int(target_label)

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device).long()
            logits  = model(xb)
            loss    = criterion(logits, yb)

            total    += yb.size(0)
            success  += (logits.argmax(1) == target_label).sum().item()
            loss_sum += loss.item() * yb.size(0)

    return loss_sum / max(total, 1), success / max(total, 1)


# ── Backdoor training loop ────────────────────────────────────────────────────
# Identical structure to the clean training loop above.  The only difference
# is that the DataLoader feeds the poisoned training set, so the model
# simultaneously learns normal classification and the trigger shortcut.
# After each epoch, both clean accuracy and ASR are measured separately.
# Skipped when SKIP_BACKDOOR_TRAINING=True — loads the saved checkpoint instead.
if SKIP_BACKDOOR_TRAINING:
    print(f"\nSKIP_BACKDOOR_TRAINING=True — loading backdoor model from {RESULTS_DIR}")
    bd_ckpt = torch.load(
        os.path.join(RESULTS_DIR, "shd_backdoor_best.pt"), map_location=device
    )
    bd_model.load_state_dict(bd_ckpt["model_state_dict"])
    bd_model.eval()
    # Populate history with placeholders so downstream plots don't crash.
    bd_asr_hist       = [bd_ckpt.get("asr",       0.0)]
    bd_clean_acc_hist = [bd_ckpt.get("clean_acc", 0.0)]
    bd_train_acc_hist = [bd_ckpt.get("train_acc", 0.0)]
    bd_train_loss_hist = [0.0]
    bd_clean_loss_hist = [0.0]
    bd_asr_loss_hist   = [0.0]
    best_asr                   = bd_ckpt.get("asr",       0.0)
    best_clean_acc_at_best_asr = bd_ckpt.get("clean_acc", 0.0)
    best_bd_epoch              = bd_ckpt.get("epoch",     0)
    print(f"Loaded backdoor model — epoch {best_bd_epoch} | "
          f"clean acc {best_clean_acc_at_best_asr*100:.2f}% | ASR {best_asr*100:.2f}%")
else:
    for epoch in range(1, bd_epochs + 1):
        bd_model.train()
        train_total, train_correct, train_loss_sum = 0, 0, 0.0
        start_time  = time.time()
        num_batches = len(shd_train_poisoned_loader)

        print(f"\nBackdoor Epoch {epoch}/{bd_epochs} | LR: {bd_optimizer.param_groups[0]['lr']:.2e}")

        for batch_idx, (xb, yb) in enumerate(shd_train_poisoned_loader):
            xb, yb = xb.to(device).float(), yb.to(device).long()

            bd_optimizer.zero_grad(set_to_none=True)
            logits = bd_model(xb)
            loss   = bd_criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bd_model.parameters(), bd_grad_clip)
            bd_optimizer.step()

            with torch.no_grad():
                train_total    += yb.size(0)
                train_correct  += (logits.argmax(1) == yb).sum().item()
                train_loss_sum += loss.item() * yb.size(0)

            if (batch_idx + 1) % bd_print_every == 0 or (batch_idx + 1) == num_batches:
                elapsed  = time.time() - start_time
                progress = (batch_idx + 1) / num_batches
                eta      = elapsed / max(progress, 1e-9) - elapsed
                filled   = int(24 * progress)
                bar      = "#" * filled + "-" * (24 - filled)
                print(
                    f"\r  [{bar}] {progress*100:5.1f}% "
                    f"batch {batch_idx+1:4d}/{num_batches} "
                    f"loss {train_loss_sum/max(train_total,1):.4f} "
                    f"acc {train_correct/max(train_total,1)*100:6.2f}% "
                    f"ETA {eta:6.1f}s",
                    end=""
                )

        print()
        bd_scheduler.step()

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc  = train_correct  / max(train_total, 1)

        clean_loss, clean_acc = evaluate_clean_accuracy(
            bd_model, shd_test_clean_loader, bd_criterion, device
        )
        asr_loss, asr = evaluate_asr(
            bd_model, shd_test_triggered_loader,
            backdoor_config["target_label"], bd_criterion, device
        )

        bd_train_loss_hist.append(train_loss)
        bd_train_acc_hist.append(train_acc)
        bd_clean_loss_hist.append(clean_loss)
        bd_clean_acc_hist.append(clean_acc)
        bd_asr_loss_hist.append(asr_loss)
        bd_asr_hist.append(asr)

        # Save the checkpoint where ASR is highest while clean accuracy stays
        # above 70% — a poor clean accuracy means the model is just broken,
        # not successfully backdoored.
        if asr > best_asr and clean_acc > 0.60:
            best_asr                   = asr
            best_clean_acc_at_best_asr = clean_acc
            best_bd_epoch              = epoch
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     bd_model.state_dict(),
                    "optimizer_state_dict": bd_optimizer.state_dict(),
                    "clean_acc":            clean_acc,
                    "asr":                  asr,
                    "train_acc":            train_acc,
                    "backdoor_config":      backdoor_config,
                    "model_name":           bd_model.__class__.__name__,
                },
                os.path.join(RESULTS_DIR, "shd_backdoor_best.pt"),
            )
            best_msg = " | saved best backdoor model"
        else:
            best_msg = ""

        print(
            f"Backdoor Epoch {epoch} done | "
            f"Train acc {train_acc*100:.2f}% | "
            f"Clean acc {clean_acc*100:.2f}% | "
            f"ASR {asr*100:.2f}% | "
            f"Best ASR {best_asr*100:.2f}% "
            f"(clean {best_clean_acc_at_best_asr*100:.2f}%, epoch {best_bd_epoch})"
            f"{best_msg}"
        )

    torch.save(
        {
            "epoch":                bd_epochs,
            "model_state_dict":     bd_model.state_dict(),
            "optimizer_state_dict": bd_optimizer.state_dict(),
            "bd_train_loss_hist":   bd_train_loss_hist,
            "bd_train_acc_hist":    bd_train_acc_hist,
            "bd_clean_loss_hist":   bd_clean_loss_hist,
            "bd_clean_acc_hist":    bd_clean_acc_hist,
            "bd_asr_loss_hist":     bd_asr_loss_hist,
            "bd_asr_hist":          bd_asr_hist,
            "backdoor_config":      backdoor_config,
            "model_name":           bd_model.__class__.__name__,
        },
        os.path.join(RESULTS_DIR, "shd_backdoor_final.pt"),
    )

    print(f"\nBackdoor final model saved to {RESULTS_DIR}/shd_backdoor_final.pt")
    print(f"Best backdoor model saved to {RESULTS_DIR}/shd_backdoor_best.pt")


# ── Final backdoor evaluation ─────────────────────────────────────────────────
# Reload the best checkpoint and run one final clean + ASR measurement to
# confirm the saved numbers match what was reported during training.
checkpoint = torch.load(os.path.join(RESULTS_DIR, "shd_backdoor_best.pt"), map_location=device)

bd_model = SHD_SNN(T=100).to(device)
bd_model.load_state_dict(checkpoint["model_state_dict"])
bd_model.eval()

print("Loaded best backdoor model")
print("Epoch:           ", checkpoint["epoch"])
print("Saved clean acc: ", checkpoint["clean_acc"] * 100)
print("Saved ASR:       ", checkpoint["asr"] * 100)

clean_loss, clean_acc = evaluate_clean_accuracy(
    bd_model, shd_test_clean_loader, bd_criterion, device
)
asr_loss, asr = evaluate_asr(
    bd_model, shd_test_triggered_loader,
    backdoor_config["target_label"], bd_criterion, device
)

print("\nFinal evaluation")
print("----------------")
print(f"Clean Accuracy:      {clean_acc*100:.2f}%")
print(f"Attack Success Rate: {asr*100:.2f}%")


# ── Backdoor training curves ──────────────────────────────────────────────────
# Three lines on one plot: train accuracy, clean test accuracy, and ASR.
# A successful attack shows clean acc and ASR both high and roughly stable.
_bd_epochs_range = range(1, len(bd_asr_hist) + 1)

plt.figure(figsize=(8, 4))
plt.plot(_bd_epochs_range, bd_train_acc_hist,  label="Train accuracy",       marker="o")
plt.plot(_bd_epochs_range, bd_clean_acc_hist,  label="Clean test accuracy",  marker="s")
plt.plot(_bd_epochs_range, bd_asr_hist,        label="ASR (triggered test)", marker="^")
plt.title("Backdoor training curves")
plt.xlabel("Epoch")
plt.ylabel("Rate")
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "bd_training_curves.png"), dpi=150, bbox_inches="tight")

plt.figure(figsize=(8, 4))
plt.plot(_bd_epochs_range, bd_train_loss_hist, label="Train loss",           marker="o")
plt.plot(_bd_epochs_range, bd_clean_loss_hist, label="Clean test loss",      marker="s")
plt.plot(_bd_epochs_range, bd_asr_loss_hist,   label="Triggered test loss",  marker="^")
plt.title("Backdoor loss curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "bd_loss_curves.png"), dpi=150, bbox_inches="tight")


# ── Analysis suite ────────────────────────────────────────────────────────────
# A set of experiments that characterise the attack beyond the headline numbers:
#   12.1  Baseline clean/ASR sanity check
#   12.2  Trigger visibility metrics (how many spikes were added?)
#   12.3  Prediction distribution: clean vs triggered
#   12.4  Trigger duration sweep
#   12.5  Trigger channel-width sweep
#   12.6  Trigger temporal-position sweep
#   12.7  Target-label confidence distribution
#   12.8  Example clean vs triggered logits for one sample

# ── Load best checkpoint for all analysis ─────────────────────────────────────
bd_checkpoint = torch.load(os.path.join(RESULTS_DIR, "shd_backdoor_best.pt"), map_location=device)

analysis_model = SHD_SNN(T=100).to(device)
analysis_model.load_state_dict(bd_checkpoint["model_state_dict"])
analysis_model.eval()

print("Loaded backdoor model for analysis")
print("Epoch:           ", bd_checkpoint["epoch"])
print("Saved clean acc: ", bd_checkpoint["clean_acc"] * 100)
print("Saved ASR:       ", bd_checkpoint["asr"] * 100)


# ── Helper: build a triggered loader from an arbitrary config dict ─────────────
def make_triggered_loader_from_config(config, mode="test_bd", batch_size=32):
    """
    Construct a SHDPoisonedDataset + DataLoader for a given backdoor config.

    Useful for sweeping hyperparameters without touching the main dataset
    objects created during training.

    Args:
        config:     Dict with the same keys as backdoor_config.
        mode:       "test_clean" or "test_bd".
        batch_size: Loader batch size.

    Returns:
        Tuple of (SHDPoisonedDataset, DataLoader).
    """
    ds = SHDPoisonedDataset(
        base_dataset     = shd_test_full,
        target_label     = config["target_label"],
        poison_ratio     = 1.0 if mode == "test_bd" else 0.0,
        mode             = mode,
        trigger_start    = config["trigger_start"],
        trigger_duration = config["trigger_duration"],
        channel_start    = config["channel_start"],
        channel_end      = config["channel_end"],
        trigger_value    = config["trigger_value"],
        seed             = config["seed"],
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )
    return ds, loader


# ── Helper: joint clean + ASR evaluation with prediction collection ────────────
def evaluate_clean_and_asr(model, clean_loader, triggered_loader, target_label):
    """
    Run one pass over both loaders and return accuracy, ASR, and raw predictions.

    Collecting raw predictions allows downstream analysis (confusion matrices,
    prediction distributions) without re-running inference.

    Args:
        model:             SHD_SNN instance to evaluate.
        clean_loader:      Loader with no trigger.
        triggered_loader:  Loader with trigger on every non-target sample.
        target_label:      The class the attack aims to predict.

    Returns:
        clean_acc, asr, clean_preds (np), clean_labels (np),
        trig_preds (np), trig_labels (np)
    """
    model.eval()

    clean_total, clean_correct         = 0, 0
    trig_total,  trig_success          = 0, 0
    clean_preds_all, clean_labels_all  = [], []
    trig_preds_all,  trig_labels_all   = [], []

    target_label = int(target_label)

    with torch.no_grad():
        for xb, yb in clean_loader:
            xb, yb = xb.to(device).float(), yb.to(device).long()
            preds   = analysis_model(xb).argmax(dim=1)

            clean_total   += yb.size(0)
            clean_correct += (preds == yb).sum().item()
            clean_preds_all.append(preds.cpu())
            clean_labels_all.append(yb.cpu())

        for xb, yb in triggered_loader:
            xb, yb = xb.to(device).float(), yb.to(device).long()
            preds   = analysis_model(xb).argmax(dim=1)

            trig_total   += yb.size(0)
            trig_success += (preds == target_label).sum().item()
            trig_preds_all.append(preds.cpu())
            trig_labels_all.append(yb.cpu())

    return (
        clean_correct / max(clean_total, 1),
        trig_success  / max(trig_total,  1),
        torch.cat(clean_preds_all).numpy(),
        torch.cat(clean_labels_all).numpy(),
        torch.cat(trig_preds_all).numpy(),
        torch.cat(trig_labels_all).numpy(),
    )


# ── Helper: trigger visibility / perturbation metrics ─────────────────────────
def compute_trigger_visibility_metrics(dataset_clean, dataset_triggered, max_samples=500):
    """
    Measure how many spikes the trigger adds relative to the clean sample.

    Metrics returned:
        mean_added_spikes       — absolute spike count added by the trigger.
        mean_relative_overhead  — added spikes / clean spike count.
        mean_l0_fraction        — fraction of positions in the tensor that changed.

    Args:
        dataset_clean:     SHDPoisonedDataset in test_clean mode.
        dataset_triggered: SHDPoisonedDataset in test_bd mode.
        max_samples:       Cap to avoid slow loops over the full dataset.

    Returns:
        Dict of metric name → float.
    """
    n = min(len(dataset_clean), len(dataset_triggered), max_samples)

    added_spikes, clean_spikes, relative_overhead, l0_fraction = [], [], [], []

    for i in range(n):
        x_clean, _ = dataset_clean[i]
        x_trig,  _ = dataset_triggered[i]

        T = min(x_clean.size(0), x_trig.size(0))
        diff = (x_trig[:T] - x_clean[:T]).clamp(min=0)

        added       = diff.sum().item()
        clean_count = x_clean[:T].sum().item()

        added_spikes.append(added)
        clean_spikes.append(clean_count)
        relative_overhead.append(added / max(clean_count, 1.0))
        l0_fraction.append((diff > 0).float().sum().item() / x_clean[:T].numel())

    return {
        "mean_added_spikes":      np.mean(added_spikes),
        "std_added_spikes":       np.std(added_spikes),
        "mean_clean_spikes":      np.mean(clean_spikes),
        "mean_relative_overhead": np.mean(relative_overhead),
        "mean_l0_fraction":       np.mean(l0_fraction),
    }


# ── 12.1) Baseline clean / ASR check ─────────────────────────────────────────
clean_ds, clean_loader_a = make_triggered_loader_from_config(
    backdoor_config, mode="test_clean", batch_size=32
)
triggered_ds, triggered_loader_a = make_triggered_loader_from_config(
    backdoor_config, mode="test_bd", batch_size=32
)

clean_acc, asr, clean_preds, clean_labels, trig_preds, trig_labels = evaluate_clean_and_asr(
    analysis_model, clean_loader_a, triggered_loader_a,
    target_label=backdoor_config["target_label"],
)

print("\nBaseline backdoor evaluation")
print("----------------------------")
print(f"Clean accuracy: {clean_acc*100:.2f}%")
print(f"ASR:            {asr*100:.2f}%")


# ── 12.2) Trigger visibility metrics ─────────────────────────────────────────
visibility_metrics = compute_trigger_visibility_metrics(
    clean_ds, triggered_ds, max_samples=500
)

print("\nTrigger visibility metrics")
print("--------------------------")
for k, v in visibility_metrics.items():
    print(f"{k}: {v:.6f}")


# ── 12.3) Prediction distribution: clean vs triggered ────────────────────────
# Shows how triggered samples collapse to the target class while clean samples
# remain spread across all 20 classes.
num_classes  = 20
clean_counts = np.bincount(clean_preds, minlength=num_classes)
trig_counts  = np.bincount(trig_preds,  minlength=num_classes)

plt.figure(figsize=(10, 4))
plt.bar(np.arange(num_classes) - 0.2, clean_counts, width=0.4, label="Clean predictions")
plt.bar(np.arange(num_classes) + 0.2, trig_counts,  width=0.4, label="Triggered predictions")
plt.axvline(backdoor_config["target_label"], linestyle="--", label="Target label")
plt.title("Prediction distribution: clean vs triggered")
plt.xlabel("Predicted class")
plt.ylabel("Count")
plt.xticks(np.arange(num_classes))
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "prediction_distribution.png"), dpi=150, bbox_inches="tight")

if not SKIP_SWEEPS:
    # ── 12.4) Trigger duration sweep ─────────────────────────────────────────────
    # Vary how many timesteps the trigger spans. Longer triggers are stronger
    # (higher ASR) but more visible (higher spike overhead).
    duration_values     = [2, 4, 6, 8, 10, 15, 20, 30]
    duration_clean_acc  = []
    duration_asr        = []
    duration_overhead   = []

    for d in duration_values:
        cfg = dict(backdoor_config)
        cfg["trigger_duration"] = d

        c_ds, c_loader = make_triggered_loader_from_config(cfg, mode="test_clean", batch_size=32)
        t_ds, t_loader = make_triggered_loader_from_config(cfg, mode="test_bd",    batch_size=32)

        c_acc, d_asr, *_ = evaluate_clean_and_asr(
            analysis_model, c_loader, t_loader,
            target_label=cfg["target_label"],
        )
        metrics = compute_trigger_visibility_metrics(c_ds, t_ds, max_samples=300)

        duration_clean_acc.append(c_acc)
        duration_asr.append(d_asr)
        duration_overhead.append(metrics["mean_relative_overhead"])

        print(f"Duration {d:2d} | Clean acc {c_acc*100:6.2f}% | ASR {d_asr*100:6.2f}% | overhead {metrics['mean_relative_overhead']:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(duration_values, duration_asr,       marker="o", label="ASR")
    plt.plot(duration_values, duration_clean_acc, marker="s", label="Clean accuracy")
    plt.title("Effect of trigger duration on ASR")
    plt.xlabel("Trigger duration [timesteps]")
    plt.ylabel("Rate")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sweep_duration_asr.png"), dpi=150, bbox_inches="tight")

    plt.figure(figsize=(8, 4))
    plt.plot(duration_values, duration_overhead, marker="o")
    plt.title("Spike overhead vs trigger duration")
    plt.xlabel("Trigger duration [timesteps]")
    plt.ylabel("Mean relative spike overhead")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sweep_duration_overhead.png"), dpi=150, bbox_inches="tight")


    # ── 12.5) Trigger channel-width sweep ────────────────────────────────────────
    # Vary how many cochlear channels the trigger spans (centred at 590).
    # Wider triggers are easier for the model to learn but more detectable.
    channel_center = (backdoor_config["channel_start"] + backdoor_config["channel_end"]) // 2
    width_values   = [5, 10, 20, 40, 60, 80, 100]
    width_asr      = []
    width_overhead = []

    for w in width_values:
        cfg = dict(backdoor_config)
        cfg["channel_start"] = max(0,   channel_center - w // 2)
        cfg["channel_end"]   = min(700, channel_center + w // 2)

        c_ds, c_loader = make_triggered_loader_from_config(cfg, mode="test_clean", batch_size=32)
        t_ds, t_loader = make_triggered_loader_from_config(cfg, mode="test_bd",    batch_size=32)

        c_acc, w_asr, *_ = evaluate_clean_and_asr(
            analysis_model, c_loader, t_loader,
            target_label=cfg["target_label"],
        )
        metrics = compute_trigger_visibility_metrics(c_ds, t_ds, max_samples=300)

        width_asr.append(w_asr)
        width_overhead.append(metrics["mean_relative_overhead"])

        print(
            f"Width {w:3d} | Channels [{cfg['channel_start']}, {cfg['channel_end']}) | "
            f"ASR {w_asr*100:6.2f}% | overhead {metrics['mean_relative_overhead']:.4f}"
        )

    plt.figure(figsize=(8, 4))
    plt.plot(width_values, width_asr, marker="o", label="ASR")
    plt.title("Effect of trigger channel width on ASR")
    plt.xlabel("Trigger channel width")
    plt.ylabel("ASR")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sweep_width_asr.png"), dpi=150, bbox_inches="tight")

    plt.figure(figsize=(8, 4))
    plt.plot(width_values, width_overhead, marker="o")
    plt.title("Spike overhead vs channel width")
    plt.xlabel("Trigger channel width")
    plt.ylabel("Mean relative spike overhead")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sweep_width_overhead.png"), dpi=150, bbox_inches="tight")


    # ── 12.6) Trigger temporal-position sweep ────────────────────────────────────
    # The model was trained with a random trigger position. This tests whether
    # placing the trigger at a fixed position still activates the backdoor —
    # if so, the model has learned the channel pattern, not just a time location.
    position_values = [0, 10, 20, 40, 60, 80, 100]
    position_asr    = []

    for pos in position_values:
        cfg = dict(backdoor_config)
        cfg["trigger_start"] = pos

        c_ds, c_loader = make_triggered_loader_from_config(cfg, mode="test_clean", batch_size=32)
        t_ds, t_loader = make_triggered_loader_from_config(cfg, mode="test_bd",    batch_size=32)

        c_acc, p_asr, *_ = evaluate_clean_and_asr(
            analysis_model, c_loader, t_loader,
            target_label=cfg["target_label"],
        )
        position_asr.append(p_asr)

        print(f"Trigger start {pos:3d} | Clean acc {c_acc*100:6.2f}% | ASR {p_asr*100:6.2f}%")

    plt.figure(figsize=(8, 4))
    plt.plot(position_values, position_asr, marker="o")
    plt.title("Synchronisation robustness: ASR vs trigger start time")
    plt.xlabel("Trigger start timestep")
    plt.ylabel("ASR")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sweep_position.png"), dpi=150, bbox_inches="tight")


    # ── 12.7) Poison ratio sweep ─────────────────────────────────────────────────
    # Vary the fraction of training samples that are poisoned.
    # Requires retraining for each value, so this sweep trains a fresh model per ratio.
    poison_ratio_values   = [0.01, 0.05, 0.10, 0.15, 0.20]
    poison_ratio_asr      = []
    poison_ratio_clean    = []

    for pr in poison_ratio_values:
        pr_train = SHDPoisonedDataset(
            base_dataset     = shd_train_full,
            target_label     = backdoor_config["target_label"],
            poison_ratio     = pr,
            mode             = "train",
            trigger_start    = backdoor_config["trigger_start"],
            trigger_duration = backdoor_config["trigger_duration"],
            channel_start    = backdoor_config["channel_start"],
            channel_end      = backdoor_config["channel_end"],
            trigger_value    = backdoor_config["trigger_value"],
            seed             = backdoor_config["seed"],
        )
        pr_loader = DataLoader(
            pr_train, batch_size=32, shuffle=True, num_workers=0,
            pin_memory=(device.type == "cuda"), collate_fn=pad_collate,
        )

        pr_model     = SHD_SNN(T=100).to(device)
        pr_optimizer = torch.optim.AdamW(pr_model.parameters(), lr=5e-4, weight_decay=0)
        pr_criterion = nn.CrossEntropyLoss(label_smoothing=0)
        pr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pr_optimizer, T_max=bd_epochs, eta_min=1e-6)
        for epoch in range(1, bd_epochs + 1):
            pr_model.train()
            for xb, yb in pr_loader:
                xb, yb = xb.to(device).float(), yb.to(device).long()
                pr_optimizer.zero_grad(set_to_none=True)
                loss = pr_criterion(pr_model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pr_model.parameters(), bd_grad_clip)
                pr_optimizer.step()
            pr_scheduler.step()

        _, pr_clean = evaluate_clean_accuracy(pr_model, shd_test_clean_loader, pr_criterion, device)
        _, pr_asr   = evaluate_asr(pr_model, shd_test_triggered_loader, backdoor_config["target_label"], pr_criterion, device)

        poison_ratio_clean.append(pr_clean)
        poison_ratio_asr.append(pr_asr)
        print(f"Poison ratio {pr:.2f} | Clean acc {pr_clean*100:.2f}% | ASR {pr_asr*100:.2f}%")

    plt.figure(figsize=(8, 4))
    plt.plot(poison_ratio_values, poison_ratio_asr,   marker="o", label="ASR")
    plt.plot(poison_ratio_values, poison_ratio_clean, marker="s", label="Clean accuracy")
    plt.title("Effect of poison ratio on ASR")
    plt.xlabel("Poison ratio")
    plt.ylabel("Rate")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sweep_poison_ratio.png"), dpi=150, bbox_inches="tight")


# ── 12.8) Target-label confidence: clean vs triggered ────────────────────────
# Histogram of the softmax probability assigned to the target class.
# On clean inputs this should be spread low; on triggered inputs it should
# spike near 1.0 if the attack is working well.
def collect_target_confidence(model, loader, target_label, max_batches=None):
    """
    Collect per-sample softmax probability of the target class.

    Args:
        model:        SHD_SNN instance.
        loader:       DataLoader to iterate.
        target_label: Index of the target class.
        max_batches:  Optional cap on number of batches processed.

    Returns:
        Tuple of (target_probs np.array, predicted_classes np.array).
    """
    model.eval()
    probs_target, preds_all = [], []
    n_batches = 0

    with torch.no_grad():
        for xb, _ in loader:
            xb    = xb.to(device).float()
            probs = torch.softmax(model(xb), dim=1)

            probs_target.append(probs[:, target_label].cpu())
            preds_all.append(probs.argmax(dim=1).cpu())

            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

    return (
        torch.cat(probs_target).numpy(),
        torch.cat(preds_all).numpy(),
    )


clean_target_conf, _ = collect_target_confidence(
    analysis_model, clean_loader_a, backdoor_config["target_label"]
)
trig_target_conf, _ = collect_target_confidence(
    analysis_model, triggered_loader_a, backdoor_config["target_label"]
)

plt.figure(figsize=(8, 4))
plt.hist(clean_target_conf, bins=40, alpha=0.6, label="Clean target confidence")
plt.hist(trig_target_conf,  bins=40, alpha=0.6, label="Triggered target confidence")
plt.title("Target-label confidence distribution")
plt.xlabel("Softmax probability of target label")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confidence_distribution.png"), dpi=150, bbox_inches="tight")

print("\nTarget confidence")
print("-----------------")
print(f"Clean mean target confidence:       {clean_target_conf.mean():.4f}")
print(f"Triggered mean target confidence:   {trig_target_conf.mean():.4f}")
print(f"Clean median target confidence:     {np.median(clean_target_conf):.4f}")
print(f"Triggered median target confidence: {np.median(trig_target_conf):.4f}")


# ── 12.8) Example logits: clean vs triggered for one sample ──────────────────
# Shows the output logit vector before and after the trigger is applied to a
# single sample. A successful attack pushes the target-class bar far higher
# than all others in the triggered case.
sample_idx = next(
    i for i in range(len(shd_test_full))
    if (int(shd_test_full[i][1].item())) != backdoor_config["target_label"]
)

x_clean, y_clean = clean_ds[sample_idx]
x_trig,  _       = triggered_ds[sample_idx]

analysis_model.eval()
with torch.no_grad():
    logits_clean = analysis_model(x_clean.unsqueeze(0).to(device).float()).squeeze(0)
    logits_trig  = analysis_model(x_trig.unsqueeze(0).to(device).float()).squeeze(0)

logits_clean_np = logits_clean.cpu().numpy()
logits_trig_np  = logits_trig.cpu().numpy()

plt.figure(figsize=(10, 4))
plt.bar(np.arange(num_classes) - 0.2, logits_clean_np, width=0.4, label="Clean logits")
plt.bar(np.arange(num_classes) + 0.2, logits_trig_np,  width=0.4, label="Triggered logits")
plt.axvline(backdoor_config["target_label"], linestyle="--", label="Target label")
plt.title(f"Example logits: clean vs triggered | original label={int(y_clean)}")
plt.xlabel("Class")
plt.ylabel("Logit")
plt.xticks(np.arange(num_classes))
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "example_logits.png"), dpi=150, bbox_inches="tight")
