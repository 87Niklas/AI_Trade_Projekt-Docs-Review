# DAX RL — Consolidated Project Brief

> **For AI assistants:** Paste this file into any new session to bring Claude fully up to speed on the project.  
> **For the owner:** This is the living quick-reference. Update it when architecture changes.

**Status:** Phase 5 complete — 2026-05-19  
**Owner:** 87Niklas — ~3 years Pro Real Time trading algorithms, 15+ years active trading, no formal programming background. Implementation via Claude Code. Code clarity matters more than usual.

---

## Project in One Paragraph

A reinforcement learning agent for intraday DAX futures (FDAX) trading. **Dual-rate SAC** (M1 main + 1-second guard) + frozen AFP + behavior cloning warmup. CleanRL-based. The DAX agent is the baseline architecture for future agents on FX, individual stocks, and indices. Typical winning trades last 5–20 minutes. Long-only currently; shorts planned after baseline validation.

**Long-term targets:** +35% annual return, max 10% annual drawdown, defensive intraday — no overnight positions.

---

## Architecture

### Environment Stack

```
DAXTradingEnv  →  DualRateEnv  →  AFPAugmentedEnv
    (728-dim)       (M1+1s)          (743-dim)
```

- `DAXTradingEnv`: raw 728-dim observation, M1 step
- `DualRateEnv`: routes 1 M1 step + 59 guard steps per bar; computes 23-dim guard obs
- `AFPAugmentedEnv`: runs frozen AFP at M1 boundaries, augments obs 728→743-dim, freezes AFP context for guard steps

### Division of Labor

| Component | Cadence | Role |
|-----------|---------|------|
| **AFP** (frozen) | M1 only | Predicts future price levels + confidence for 5 horizons |
| **Main policy** (SAC) | M1 / 60s | Decides entries, sets initial SL/TP |
| **Guard policy** (SAC) | 1s | Adjusts SL, closes positions |

AFP is the forecaster. Main policy is the entry manager. Guard policy is the risk manager during open trades. Both SAC policies train simultaneously with separate networks, buffers, and optimizers.

---

## Observation Layouts

### Main Observation — 743 dims (at M1 cadence)

| Block | Range | Dim | Description |
|-------|-------|-----|-------------|
| CNN sequence | [0, 672) | 672 | 32 bars × 21 features — CNN input |
| Current scalars | [672, 728) | 56 | State at "now" — 15 groups A–O |
| AFP outputs | [728, 743) | 15 | Frozen AFP predictions |

**Per-bar features (21):** price/bar action (3), L2 microstructure (10), 7 candle patterns, structure signal (1).

**Current scalars (56) — groups:**

| Group | Dims | Contents |
|-------|------|---------|
| A | 5 | Temporal (time of day, session progress) |
| B | 5 | Position state (on_market, hold_bars, position direction) |
| C | 4 | Unrealized PnL (raw, normalized, R-multiple, peak) |
| D | 4 | Daily performance (daily PnL, drawdown, trades today) |
| E | 2 | Risk state (daily DD flag, exposure) |
| F | 2 | Recent performance (winrate, G:L ratio) |
| G | 6 | Trend & HTF (EMA21, M5 trend, M15 trend, trend strength) |
| H | 5 | Market state now (ATR, spread, volatility regime) |
| I | 2 | L2 snapshot (imbalance, microprice deviation) |
| J | 2 | Imitation (quality, weight norm) |
| K | 9 | Multi-timeframe levels (M5/M15 pivots, Fibonacci, prev-day OHLC) |
| L | 4 | Tick & activity (tick freq, urgency, recent velocity) |
| M | 3 | Breakout & compression (range, breakout flag) |
| N | 3 | Urgency components (volume spike, momentum accel, spread expansion) |
| O | 0 | Cross-asset stocks — reserved, currently inactive |

### Guard Observation — 23 dims (at 1s cadence)

Updated each second between M1 boundaries — tick-fresh features only:

| Dims | Contents |
|------|---------|
| 4 | Position state |
| 4 | Unrealized PnL state |
| 3 | Fresh L2 (recomputed at tick rate) |
| 4 | Fresh tick activity |
| 3 | Urgency components (recomputed at tick rate) |
| 4 | AFP context (frozen from last M1 boundary) |
| 1 | Time-in-bar (seconds elapsed) |

CNN sequence is **not** included in guard obs — it is M1-cadence and frozen between bars.

---

## Action Spaces

### Main Action — 3 dims

| Index | Range | Meaning |
|-------|-------|---------|
| 0 | [-1, 1] | Trade signal: >0.25 entry, <-0.25 manual exit |
| 1 | [0.5, 2.0] | Initial SL distance (ATR multiple) |
| 2 | [1.5, 4.0] | Initial TP distance (ATR multiple) |

### Guard Action — 2 dims

| Index | Range | Meaning |
|-------|-------|---------|
| 0 | [-1, 1] | SL adjustment: >0 tighten toward breakeven, <0 no change |
| 1 | [0, 1] | Exit signal: ≥0.5 closes the position immediately |

---

## Key Files

| File | Purpose |
|------|---------|
| `DAX/sac_dax_cleanrl.py` | Dual-rate SAC training loop |
| `DAX/strategy.py` | Candlestick + pivot pattern detection |
| `DAX/envs/dax_trading_env.py` | DAX Gymnasium environment (728-dim obs) |
| `DAX/envs/dual_rate_env.py` | M1 + 59×1s guard cadence |
| `DAX/envs/wrappers.py` | AFPAugmentedEnv (728→743-dim) |
| `DAX/features/feature_schema.py` | **Single source of truth** for all dims |
| `DAX/features/feature_extractor.py` | L2, MTF, tick, pivot computations |
| `DAX/agents/aux_cnn_extractor.py` | CNN backbone for main policy |
| `DAX/agents/guard_actor.py` | Guard actor + critic (MLP) |
| `DAX/agents/behavior_cloning.py` | BC loss (pure function) |
| `DAX/afp/afp_model_v2.py` | AFPCNNLSTM — 15 output heads |
| `DAX/rewards/risk_first_reward.py` | Risk-first hierarchical reward |

---

## Architecture Decisions (Already Made — Do Not Re-Litigate)

1. **CleanRL, not Stable-Baselines3.** Migration completed May 2026.
2. **AFP integration is "Alternative A"** — frozen AFP as external feature provider via env wrapper. No joint training, no AFP gradients during SAC.
3. **Single source of truth:** `features/feature_schema.py`. All consumers import dimensions from there. Never hardcode integers.
4. **AFP has 15 scalar outputs:** 4 horizons (5/10/20/30) each with future_price + confidence, plus pred_high/low, volatility_forecast, 3 direction probabilities, fib_break.
5. **Dual-rate policy from the start** — architecturally separate main (M1) and guard (1s) policies with separate networks, replay buffers, and optimizers.
6. **BC is weak by design** — imitation labels flag HC pattern firings, not trade quality. BC warmup is short (80k steps) and applies to main policy only.
7. **Reward sharing** — both policies receive the same trade-close reward; guard additionally receives per-second shaping.

---

## Critical Rules for AI Assistants

- `feature_schema.py` is the single source of truth. Never hardcode dimension constants.
- Main and guard are separate. Never mix their obs, actions, buffers, or networks.
- AFP weights must be frozen (`requires_grad_(False)`) during SAC training.
- Solve problems correctly from the start. No quick fixes, workarounds, or temporary solutions.
- If a dimension seems wrong, verify against `feature_schema.py` before assuming.
- If a session is getting long and context is filling up, summarize state explicitly rather than simplifying or rationalizing away existing code.

---

## Phases Completed

| Phase | What | Status |
|-------|------|--------|
| 0 | CleanRL migration, BC gradient fix, CNN layout fix | Done |
| 1 | feature_schema.py, 728-dim obs end-to-end | Done |
| 2 | AFP model v2 (15 heads, per-horizon confidence) | Done |
| 3 | M5/M15 pivot detection, previous-day OHLC | Done |
| 4 | DualRateEnv (M1 + 59×1s guard per bar) | Done |
| 5 | AFPAugmentedEnv wrapper, GuardActor, dual-rate SAC loop | Done |

## Next Steps (In Order)

1. Train AFP on available 4 months L2 data (`afp/train_afp.py`)
2. Verify AFP confidence calibration (`afp/evaluate_afp.py`)
3. Short SAC training run (~50k steps) to verify loss curves
4. Full training when ~12 months L2 data available (rented GPU)
5. Adaptive position sizing using AFP per-horizon confidence
6. Cross-asset features from top-10 DAX stocks (when logging pipeline is ready)

---

## Quick Context for New Chats

> "I'm working on a DAX RL trading agent. Dual-rate SAC (M1 main + 1s guard) + frozen AFP via env wrapper, CleanRL-based. Phase 5 complete — all six foundation phases done. Currently waiting on sufficient L2 data to train the AFP model. I work on this with Claude Code on a desktop PC and review sessions in the evening. Coding rules: no quick fixes, ever. Strategy: defensive reversal-seed, but the agent should discover its own patterns via L2 and tick microstructure. See DAX_CONSOLIDATED_DOCUMENTATION.md for the full brief."

---

*Last updated: 2026-05-19. Update this file whenever the architecture changes.*
