# polymarket-faa-attribution-02 — forecast v3

Entry for the make.fun competition `polymarket-faa-attribution-02`:
forecast Polymarket's FAA-outage attribution Yes prices (four markets),
minute by minute, 240 minutes ahead, five quantiles, pinball loss.

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `<id>.out` per
`<id>.in` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Method

The v2 model plus two refinements, both gated on the sum of the case's
checkpoint prices exceeding 1 (an overpriced field of near-mutually-
exclusive outcomes):

- **Base (v2):** persistence median with a conditional sum-reversion
  shift (toward `x_i / S`, time constant 60 min, clipped to ±0.03);
  spread from an EWMA-weighted (half-life 240 min) jump-robust MAD of
  h-minute price changes at anchors (1, 5, 15, 60, 240) over the last
  720 minutes, floor 0.0002, ×0.85, log-log interpolated in h; the 95%
  quantile widened ×1.3; markets too young for any anchor fall back to
  the house sigma 0.0025 unscaled.
- **Upper-shoulder compression (new):** when S > 1, the 75% quantile's
  spread is halved. An overpriced field's aggregate must shed S − 1, so
  its upside is bounded and the v2 upper inner quantile sat
  systematically too high (80.4% practice coverage vs 75% nominal).
  Applied unconditionally, the same narrowing hurt the one underpriced
  practice case — hence the gate.
- **Pooled spread (new):** when S > 1, each market's sigma curve is
  shrunk 15% toward the case-median sigma curve, cutting per-market
  estimation noise in the regime where the fan is compressed.

## Measured on the practice cases

(truth rebuilt from the next case's overlap for p00–p04 and from
Polymarket's public price history for p05; the house baseline
reproduces the RULES figure exactly on this truth)

| entry | practice pinball loss |
|---|---|
| house baseline | 0.363589 |
| v1 | 0.280044 |
| v2 | 0.266431 |
| **v3 (this entry)** | **0.259696** |

Per case: p00 0.3898, p01 0.3016, p02 0.3515, p03 0.2578, p04 0.1968,
p05 0.1155 — better than v2 on five cases, tied on p02, worst
regression +0.0012 (p04). Selection leave-one-out over the finalist
set picked the gated form in all six folds. On the final, where the
FAA prices sum to ~0.09, both refinements are inert and this entry
coincides with v2 unless the field becomes overpriced.
