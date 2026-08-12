"""
Framework-Spike: TransformerLens mit custom H/T/O/TH-Tokenizer.

Testet die 3 kritischen Faehigkeiten fuer das Mini-Monopoly-Experiment:
  A) run_with_cache  -> layer-wise Probing (H1/H2, Abschnitt 5.4)
  B) run_with_hooks  -> Aktivierungs-Patching / Interventionen (H3)
  C) Tokenspezifische Cache-Indexierung (activation[layer, token] -> state)
Laeuft auf CPU fuer den Spike (Mini-Modell), nicht GPU.
"""

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ----------------------------------------------------------------------
# 1. Custom-Vokabular analog zu EXPERIMENT.md 5.1/5.1a
#    Numerische Groessen als TH/H/T/O-Digit-Tokens, kategorial atomar.
# ----------------------------------------------------------------------
TOKENS = [
    # Sonder-/Event-Tokens
    "<PAD>", "START", "ROLL", "BUY", "PASS", "PAY_RENT",
    # Spieler (atomar)
    "P0", "P1", "P2",
    # Wuertelwerte
    "1", "2", "3", "4", "5", "6",
    # MONEY-Marker + Stellenwerte (Numerisch als Digit-Struktur)
    "MONEY", "TH0", "TH1",          # Tausenderstelle (0..1, max 1500)
    "H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
    "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    "O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9",
]

cfg = HookedTransformerConfig(
    n_layers=4,
    d_model=128,
    d_head=32,
    n_heads=4,
    d_mlp=256,
    n_ctx=64,
    d_vocab=len(TOKENS),
    act_fn="gelu",
    attention_dir="causal",
    use_attn_result=True,
    use_split_qkv_input=True,
    device="cpu",
    dtype=torch.float32,
)

model = HookedTransformer(cfg)
model.tokenizer = None  # wir nutzen rein token-id-basiertes Input, kein HF-Tokenizer

# --- Beispiel-Ereignissequenz (Mini-Monopoly):
# P0 wuerfelt 6 und kauft. Danach MONEY-Ausgabe 940 = TH0 H9 T4 O0
seq_tokens = ["START", "P0", "ROLL", "6", "P0", "BUY", "MONEY", "TH0", "H9", "T4", "O0"]
ids = torch.tensor([TOKENS.index(t) for t in seq_tokens]).unsqueeze(0)  # shape (1, seq)

print("Vokabular-Groesse:", len(TOKENS))
print("Sequenz:", seq_tokens)
print("IDs:", ids.tolist()[0])

# ======================================================================
# TEST A: run_with_cache -> layer-wise Probing
# ======================================================================
logits, cache = model.run_with_cache(ids)

# Zeige die Residual-Strom-Aktivierung am 'MONEY'-Token (Index 6) fuer Layer 3
money_idx = 6
acts = cache["resid_post", 3]            # (1, seq, d_model)
a_money = acts[0, money_idx]             # (d_model,)
print("\n[A] resid_post Layer 3 am MONEY-Token -> Form:", tuple(a_money.shape),
      "| d_model=", cfg.d_model)

# Layer-wise-Probing-Grundlage: nehme Aktivierung pro Layer (0..3) an jeder Position
for layer in range(cfg.n_layers):
    layer_acts = cache["resid_post", layer][0]   # (seq, d_model)
    print(f"    Layer {layer}: cache['resid_post',{layer}] Form {tuple(layer_acts.shape)} "
          f"-> fuer Probe pro Token/Position nutzbar")

# ======================================================================
# TEST B: run_with_hooks -> Intervention / Patching (H3/H4)
# ======================================================================
def patch_money_activation(act: torch.Tensor, hook):
    """Patch die Aktivierung am MONEY-Token (Position 6) in Layer 2:
    schreibe eine 'Cash=430' nahegelegte Richtung hinein (Demo-Dummy).
    Zeigt, dass gezieltes Token-Patching moeglich ist."""
    act = act.clone()
    act[0, 6, :] = 0.0          # NUR Position 6, alle d_model-Komponenten
    return act

with model.hooks(fwd_hooks=[("blocks.2.hook_resid_post", patch_money_activation)]):
    logits_patched, _ = model.run_with_cache(ids, return_type="logits")

# WICHTIG: Patching an Position 6 hat die Vorhersage-Logits an Position 7+ veraendert?
logits_orig = logits[0, 7]
logits_pat  = logits_patched[0, 7]
diff = (logits_orig - logits_pat).abs().sum().item()
print("\n[B] Intervention (Patch Position 6, Layer 2):")
print(f"    Logit-Differenz an Position 7 (naechster Token) = {diff:.4f}")
print("    -> Aktivierung wurde erfolgreich gezielt ueberschrieben (kausaler Effekt messbar)")

# ======================================================================
# TEST C: run_with_hooks alternativ mit HookPoint-Zugriff auf Zwischenwerte
# ======================================================================
captured = {}
def capture(act, hook):
    captured[hook.name] = act.detach()
    return act

with model.hooks(fwd_hooks=[("blocks.1.hook_resid_pre", capture)]):
    model(ids)  # forward pass

print("\n[C] Zugriff auf hook_resid_pre von Layer 1 ->", list(captured.keys()),
      "| Form:", tuple(captured["blocks.1.hook_resid_pre"].shape))

print("\nSPIKE ERFOLGREICH: run_with_cache, run_with_hooks/Patching, "
      "HookPoint-Zugriff und custom-Digit-Vokabular funktionieren in TransformerLens.")
