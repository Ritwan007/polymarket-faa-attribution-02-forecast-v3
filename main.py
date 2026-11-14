#!/usr/bin/env python3
"""Forecast Polymarket FAA-outage attribution Yes prices, minute by minute.

Entry for make.fun competition polymarket-faa-attribution-02 (program/v3).

Method: the v2 model (per-market persistence median with a conditional
sum-reversion shift; spread from an EWMA-weighted, jump-robust MAD of
h-minute price changes at anchors (1, 5, 15, 60, 240) over the last 720
minutes, x0.85, 95% quantile widened x1.3; too-young markets fall back to
the house sigma 0.0025 unscaled), plus two refinements that activate
under exactly the condition the sum reversion already uses -- the sum S
of the case's checkpoint prices above 1, i.e. an overpriced field whose
outcomes are bounded by the near-arbitrage condition that the shares of
mutually exclusive outcomes sum to 1:

1. Upper-shoulder compression: when S > 1, the 75% quantile's spread is
   halved (factor 0.5).  An overpriced field has bounded upside: in
   aggregate its prices must shed S - 1, so the upper inner quantile of
   the v2 fan sat systematically too high (its practice coverage was
   80.4% against a 75% nominal, while the same narrowing applied
   unconditionally hurt the one underpriced practice case, whose field
   genuinely had headroom -- hence the gate).
2. Pooled spread: when S > 1, each market's sigma curve is shrunk 15%
   toward the case-median sigma curve (markets too young to estimate a
   curve contribute the flat fallback 0.0025 to that median but do not
   consume it).  The case's markets share one event, so their
   volatilities inform each other; pooling cuts the estimation noise of
   the per-market MAD curves exactly in the regime where the fan is
   being compressed.

Measured on the practice (truth rebuilt from the next case's overlap for
p00-p04 and from Polymarket's public price history for p05; the baseline
reproduces the RULES figure 0.363589 exactly on this truth):
  house baseline  0.363589
  v1 entry        0.280044
  v2 entry        0.266431
  this entry      0.259696  (per case: p00 0.3898, p01 0.3016, p02 0.3515,
                             p03 0.2578, p04 0.1968, p05 0.1155)
Leave-one-out (selection over the finalist set, v2 included): the gated
form was picked on the other five cases in all six folds and beat or
tied v2 on five of six held-out cases (worst +0.0016, on p04); LOO
aggregate 0.259950 vs v2 0.266393 (numpy reference; the shipped stdlib
program measures 0.259696).  On the final, where the FAA prices sum to
~0.09, both refinements are inert and this entry coincides with v2
unless the field becomes overpriced, so the change can only help there.

Standard library only, no network, deterministic: two runs are
byte-identical.  Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in found.
"""

import json
import math
import os
from pathlib import Path

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels, to 6 decimals.
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
# Extra widening of the 95% quantile (upper-tail jump risk).
UP95_MULT = 1.3
# Compression of the 75% quantile's spread when the field is overpriced.
UP75_GATED_MULT = 0.5
# Weight of the case-median sigma curve in each market's spread when the
# field is overpriced.
POOL_WEIGHT = 0.15
# How far back the spread estimation looks (minutes).
WINDOW = 720
# Recency half-life (minutes) weighting the price changes.
HALF_LIFE = 240.0
# Anchor horizons (minutes) at which the h-minute change scale is measured.
ANCHORS = (1, 5, 15, 60, 240)
# Minimum per-sqrt-minute sigma, so a market that has never moved still gets
# a non-degenerate forecast.
SIGMA_FLOOR = 0.0002
# Calibration multiplier on the estimated sigma (walk-forward measured).
SIGMA_MULT = 0.85
# Fallback sigma when a market has too little history for any anchor.
FALLBACK_SIGMA = 0.0025
# An anchor needs at least this many h-minute changes to be trusted.
MIN_WINDOWS = 30
# Sum-reversion strength and time constant (minutes); the shift is clipped
# to +-SUMREV_CAP so no median is ever dragged far from its checkpoint.
SUMREV_LAM = 1.0
SUMREV_TAU = 60.0
SUMREV_CAP = 0.03


def median(values):
    """Median of a non-empty list (average of the two middles when even)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def weighted_median(pairs):
    """Median of values with weights, linearly interpolated at cum. 0.5.

    pairs is a list of (value, weight) with positive weights.  Matches the
    numpy reference (np.interp(0.5, cum_weights, sorted_values)): the first
    value when the first weight already carries half the total, otherwise
    a linear interpolation between the two bracketing values.
    """
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    cum = 0.0
    prev_value = ordered[0][0]
    prev_cum = 0.0
    for value, weight in ordered:
        cum += weight
        share = cum / total
        if 0.5 <= share:
            if prev_cum == 0.0:
                return value
            if share == prev_cum:
                return value
            t = (0.5 - prev_cum) / (share - prev_cum)
            return prev_value + t * (value - prev_value)
        prev_value = value
        prev_cum = share
    return ordered[-1][0]


def clamp6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def anchor_sigmas(prices):
    """Robust per-sqrt-minute sigma at each anchor horizon.

    Returns (horizons, sigmas) with at least 30 sampled changes per anchor,
    or ([], []) when the market is too young for every anchor.  Each change
    is weighted by recency (half-life HALF_LIFE minutes) so the estimate
    follows the current volatility regime.
    """
    n = len(prices)
    lo = max(0, n - WINDOW)
    horizons = []
    sigmas = []
    for h in ANCHORS:
        end = n - h
        if end - lo < MIN_WINDOWS:
            continue
        diffs = []
        for i in range(lo, end):
            age = (n - 1) - (i + h)
            weight = 0.5 ** (age / HALF_LIFE)
            diffs.append((prices[i + h] - prices[i], weight))
        centre = weighted_median(diffs)
        devs = [(abs(d - centre), w) for d, w in diffs]
        mad = weighted_median(devs)
        sigma = 1.4826 * mad / math.sqrt(h)
        horizons.append(h)
        sigmas.append(max(sigma, SIGMA_FLOOR))
    return horizons, sigmas


def sigma_at(h, horizons, sigmas):
    """sigma(h) by log-log interpolation between anchors, flat outside."""
    if not horizons:
        return FALLBACK_SIGMA
    if h <= horizons[0]:
        return sigmas[0]
    if h >= horizons[-1]:
        return sigmas[-1]
    for j in range(len(horizons) - 1):
        h0, h1 = horizons[j], horizons[j + 1]
        if h0 <= h <= h1:
            w = (math.log(h) - math.log(h0)) / (math.log(h1) - math.log(h0))
            return math.exp(math.log(sigmas[j]) * (1.0 - w)
                            + math.log(sigmas[j + 1]) * w)
    return sigmas[-1]


def forecast_case(case):
    """240 rows of five quantiles per market of one case."""
    horizon = case["horizon_minutes"]
    markets = case["markets"]
    total = sum(m["prices"][-1] for m in markets)
    over = total > 1.0
    # Per-market sigma curves over the horizon (None when the market is
    # too young for every anchor), computed once and reused for every row.
    curves = []
    for m in markets:
        prices = m["prices"]
        if prices:
            horizons, sigmas = anchor_sigmas(prices)
        else:
            horizons, sigmas = [], []
        if horizons:
            curves.append([sigma_at(h, horizons, sigmas)
                           for h in range(1, horizon + 1)])
        else:
            curves.append(None)
    # Case-median sigma curve, the pooling target.  Only an overpriced
    # field pools; a too-young market contributes the flat fallback to
    # the median but keeps its own unpooled fallback forecast.
    pooled = None
    if over:
        pooled = [median([curves[i][idx] if curves[i] is not None
                          else FALLBACK_SIGMA
                          for i in range(len(markets))])
                  for idx in range(horizon)]
    forecasts = []
    for i, m in enumerate(markets):
        prices = m["prices"]
        x = prices[-1] if prices else 0.5
        rows = []
        for h in range(1, horizon + 1):
            # Sum-reversion shift: only when the field is overpriced
            # (sum > 1), pulling the median toward its normalised share,
            # clipped so it can never drag a market far from its price.
            shift = 0.0
            if over:
                target = x / total
                shift = SUMREV_LAM * (target - x) * (
                    1.0 - math.exp(-h / SUMREV_TAU))
                shift = max(-SUMREV_CAP, min(SUMREV_CAP, shift))
            if curves[i] is not None:
                sigma = curves[i][h - 1]
                if over:
                    sigma = ((1.0 - POOL_WEIGHT) * sigma
                             + POOL_WEIGHT * pooled[h - 1])
                spread = sigma * SIGMA_MULT * math.sqrt(h)
                row = []
                for j, z in enumerate(Z):
                    factor = 1.0
                    if j == 4:
                        factor = UP95_MULT
                    elif j == 3 and over:
                        # Overpriced field: the upside is bounded by the
                        # sum constraint, so the upper shoulder of the
                        # fan is compressed toward the median.
                        factor = UP75_GATED_MULT
                    row.append(clamp6(x + shift + z * spread * factor))
            else:
                # Too young to estimate any anchor: the house baseline's
                # own sigma, unscaled -- a market we know nothing about
                # gets the prior spread, not the calibrated one.
                spread = FALLBACK_SIGMA * math.sqrt(h)
                row = [clamp6(x + shift + z * spread) for z in Z]
            rows.append(row)
        forecasts.append({"id": m["id"], "q": rows})
    return forecasts


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in input_dir.iterdir()
                    if p.is_file() and p.name.endswith(".in"))
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(
                f"{path.name} asks for quantiles {case['quantiles']}, "
                f"not {QUANTILES}")
        out = json.dumps({"forecasts": forecast_case(case)},
                         separators=(",", ":")) + "\n"
        (output_dir / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
