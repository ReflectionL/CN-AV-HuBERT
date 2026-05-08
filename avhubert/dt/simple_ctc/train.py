import torch, os, math
import torch.nn as nn
from torch.utils.data import DataLoader
from model import AVHubertLinearProbe
from dataset import SimpleAVCtcDataset, collate
from metrics import compute_cer

def greedy_decode(log_probs, blank=1):
    # log_probs: T x B x V
    max_idx = log_probs.argmax(-1)  # T x B
    hyps = []
    for b in range(max_idx.size(1)):
        prev = None
        seq = []
        for t in max_idx[:, b].tolist():
            if t != blank and t != prev:
                seq.append(t)
            prev = t
        hyps.append(seq)
    return hyps

def ids_to_text(ids, idx2sym):
    return "".join([idx2sym[i] for i in ids if idx2sym[i] not in ["<pad>","<blank>"]])

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SimpleAVCtcDataset(args.train_tsv, args.dict)
    idx2sym = dataset.symbols
    model = AVHubertLinearProbe(
        ckpt_path=args.ckpt,
        vocab_size=len(idx2sym),
        input_modality=args.modality
    ).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=dataset.blank_idx, zero_infinity=True)

    opt = torch.optim.AdamW(model.proj.parameters(), lr=args.lr, weight_decay=args.wd)

    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, collate_fn=collate)

    best_cer = 1e9
    patience = 0
    for step, sample in enumerate(loader, 1):
        model.train()
        logits, padding_mask = model(sample)
        # logits: T x B x V
        log_probs = logits.log_softmax(-1)
        T, B, V = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long)
        targets = sample["target"]
        target_lengths = sample["target_lengths"]
        # 展平 target 去掉 pad
        flat_targets = []
        true_lengths=[]
        for i in range(B):
            l=target_lengths[i].item()
            flat_targets.append(targets[i,:l])
            true_lengths.append(l)
        flat_targets = torch.cat(flat_targets)
        loss = ctc_loss_fn(log_probs, flat_targets, input_lengths, target_lengths)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.valid_interval == 0:
            model.eval()
            with torch.no_grad():
                logp = log_probs.detach()
                hyps = greedy_decode(logp, blank=dataset.blank_idx)
                cer_list=[]
                offset=0
                for b in range(B):
                    ref_ids = flat_targets[offset:offset+true_lengths[b]].tolist()
                    offset += true_lengths[b]
                    hyp_txt = ids_to_text(hyps[b], idx2sym)
                    ref_txt = ids_to_text(ref_ids, idx2sym)
                    cer_list.append(compute_cer(ref_txt, hyp_txt))
                cer = sum(cer_list)/len(cer_list)
            print(f"step {step} | loss {loss.item():.4f} | CER {cer*100:.2f}%")
            improved = (best_cer - cer) > args.min_rel_improve
            if cer < best_cer:
                best_cer = cer
            if improved:
                patience = 0
                torch.save({"proj": model.proj.state_dict(), "best_cer": best_cer}, os.path.join(args.save_dir, "best.pt"))
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print("Early stop.")
                    break
        if step >= args.max_steps:
            break

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--save_dir", default="ckpts_lp")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--valid_interval", type=int, default=200)
    ap.add_argument("--min_rel_improve", type=float, default=0.002)  # 0.2%
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--modality", choices=["av","v"], default="av")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)