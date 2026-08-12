"""
Embedding-Init-Check (§5.5 Follow-up 4 / §6 Schritt 4).

Verifiziert, dass die zufaellige Standard-Initialisierung der Embeddings
KEINERLEI numerische/Stellenwert-Struktur einbaut — genau das Encoding-
Confound, das die Tokenisierungsentscheidung (H/T/O statt Buckets, keine
numerische Init) vermeiden soll.

Pruefungen:
  1) d_vocab wird EXPLIZIT auf len(VOCAB)=48 gesetzt (nicht GPT-2-Default).
  2) Korrelation zwischen numerischer Distanz zweier Digit-Tokens (|a-b|)
     und Embedding-(Ko-)Sinus-Abstand ist ~0 bei Random-Init.
     Nach dem Training sollte sie deutlich < 0 werden (nahe Werte = nahe
     Embeddings) — der Vergleich macht den Unterschied sichtbar.
  3) Mean Cosine Similarity numerisch-benachbarter Digit-Paare ist NICHT
     signifikant hoeher als bei zufaelligen Paaren (kein pre-baked
     Stellenwert-Raster).

Laeuft auf CPU (Mini-Modell), wie der Spike.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from data.tokenizer import VOCAB, MiniTokenizer

# Digit-Tokens je Stelle (fuer Struktur-Check)
DIGITS = {
    "H": [f"H{d}" for d in range(10)],
    "T": [f"T{d}" for d in range(10)],
    "O": [f"O{d}" for d in range(10)],
    "TH": ["TH0", "TH1"],
}

def build_model(n_layers=4, d_model=256, n_ctx=288):
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_model // 4,
        n_heads=4,
        d_mlp=4 * d_model,
        n_ctx=n_ctx,
        d_vocab=len(VOCAB),          # EXPLIZIT, nicht Default
        act_fn="gelu",
        attention_dir="causal",
        use_attn_result=True,
        use_split_qkv_input=True,
        device="cpu",
        dtype=torch.float32,
    )
    model = HookedTransformer(cfg)
    model.tokenizer = None
    return model

def cosine(a, b):
    return (a @ b) / (a.norm() * b.norm() + 1e-12)

def structure_stats(W, token_names):
    """Korrelation numerische-Distanz vs. Embedding-Abstand + Nachbar-Mean-Cos."""
    stoi = {t: i for i, t in enumerate(token_names)}
    results = {}
    for place, names in DIGITS.items():
        idx = [stoi[n] for n in names]
        embs = W[idx]                      # (k, d_model)
        vals = torch.arange(len(idx)).float()
        sims = []
        dists = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                sims.append(cosine(embs[i], embs[j]))
                dists.append(abs(vals[i] - vals[j]))
        sims = torch.tensor(sims)
        dists = torch.tensor(dists)
        corr = torch.corrcoef(torch.stack([dists, sims]))[0, 1].item()
        # Nachbar-Mean vs. Zufalls-Mean
        adj_mask = (dists == 1)
        adj_mean = sims[adj_mask].mean().item() if adj_mask.any() else float("nan")
        rand_mean = sims.mean().item()
        results[place] = {
            "corr_dist_sim": round(corr, 4),
            "adj_mean": round(adj_mean, 4),
            "rand_mean": round(rand_mean, 4),
        }
    return results

def main():
    model = build_model()
    print("d_vocab gesetzt:", model.cfg.d_vocab, "(VOCAB =", len(VOCAB), ")")
    assert model.cfg.d_vocab == len(VOCAB), "d_vocab falsch"

    W = model.embed.W_E.detach()          # (d_vocab, d_model)
    print("W_E Form:", tuple(W.shape), "\n")

    stats = structure_stats(W, VOCAB)
    print("Random-Init-Struktur (je Stelle H/T/O/TH):")
    print(f"  {'Stelle':4} {'corr(Dist,Sim)':16} {'adj-mean':10} {'rand-mean':10}")
    for place, s in stats.items():
        print(f"  {place:4} {s['corr_dist_sim']:>14} {s['adj_mean']:>10} {s['rand_mean']:>10}")

    # Bewertung
    ok = True
    for place, s in stats.items():
        c = s["corr_dist_sim"]
        if abs(c) > 0.2:
            print(f"  WARNUNG: {place}: |corr|={abs(c):.3f} > 0.2 -> moegliche Struktur!")
            ok = False
        adj_vs_rand = abs(s["adj_mean"] - s["rand_mean"])
        if adj_vs_rand > 0.05:
            print(f"  WARNUNG: {place}: adj_mean weicht um {adj_vs_rand:.3f} von rand_mean ab")
            ok = False
    print()
    print("ERGEBNIS:", "OK — keine numerische/Stellenwert-Struktur in Random-Init."
          if ok else "PRUEFEN — verdaechtige Struktur gefunden.")

if __name__ == "__main__":
    main()
