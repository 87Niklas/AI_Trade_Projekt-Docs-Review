# DAX RL — Defensive Intraday Trading Agent

**Repository:** AI_Trade_Projekt — DAX module  
**Status:** Phase 5 complete — dual-rate SAC, AFP wrapper, guard policy  
**Obs space:** 728-dim base / 743-dim AFP-augmented  
**Last updated:** 2026-05-19

## Project Overview

A reinforcement learning system for intraday trading of DAX futures (FDAX). Built on **CleanRL** with a **dual-rate policy architecture**: a main policy at M1 cadence for entries, and a guard policy at 1-second cadence for position management. A frozen **Auxiliary Future Prediction (AFP)** model augments the main observation with 15 future-price outputs via an environment wrapper.

**Primary objective:** +35% annual return, max 10% annual drawdown. Defensive, high-conviction intraday trading — no overnight positions.

The system processes multi-timeframe data (M1 primary + M5/M15 pivots), full 10-level L2 order book, 7 candlestick/pivot patterns, Fibonacci levels, previous-day OHLC, and tick microstructure features.

---

## For the Owner (87Niklas)

**Background:** ~3 years building trading algorithms in Pro Real Time, 15+ years active trading experience. No formal programming background — implementation is handled via Claude and Claude Code. Code clarity and documentation matter more than in a typical software project.

**Workflow:**
- Architecture and strategy discussions in Claude chat (Swedish is fine)
- Implementation via Claude Code on the desktop PC
- Evening review and testing sessions on laptop (2–3 hours)
- Large training runs on rented GPU

This documentation set consists of three canonical files — update these instead of creating new plan files:

| File | Purpose |
|------|---------|
| **README.md** (this file) | Current status, file map, quick reference |
| **CODING_STANDARDS.md** | How we write code and documentation |
| **STRATEGY_AND_DECISIONS.md** | Why we do what we do — architecture decisions with rationale |

---

## For Claude / AI Assistants

When starting a new session on this project, read the canonical files in this order:

1. `README.md` — project status and structure (this file)
2. `STRATEGY_AND_DECISIONS.md` — architecture decisions and rationale
3. `CODING_STANDARDS.md` — coding rules and conventions

**Critical rules — never violate without explicit owner approval:**
- `DAX/features/feature_schema.py` is the single source of truth for all dimension constants. Never hardcode dims.
- Main policy (743-dim obs, 3-dim action) and guard policy (23-dim obs, 2-dim action) are architecturally separate. Never conflate them.
- Solve problems correctly from the start. No quick fixes, workarounds or temporary solutions.

For a full project brief suitable for pasting into a new chat, see `DAX_CONSOLIDATED_DOCUMENTATION.md`.

---

## Architecture at a Glance

```
Env stack:   DAXTradingEnv → DualRateEnv → AFPAugmentedEnv
Main obs:    728-dim base + 15 AFP outputs = 743-dim  (M1 cadence)
Guard obs:   23-dim tick-fresh subset                  (1s cadence)
Main action: [trade_signal, sl_atr_mult, tp_atr_mult]  3-dim
Guard action:[sl_adjustment, exit_signal]               2-dim
```

| Component | Cadence | Role |
|-----------|---------|------|
| AFP (frozen) | M1 | Predicts future price + confidence for 5 horizons |
| Main policy (SAC) | M1 / 60s | Entry decisions, initial SL/TP |
| Guard policy (SAC) | 1s | SL tightening, emergency exits |

Both policies train simultaneously with separate actor/critic networks and separate replay buffers. AFP runs only at M1 boundaries and is frozen between them.

---

## File Map

```
AI_Trade_Projekt/
├── README.md                        ← this file
├── CODING_STANDARDS.md              ← coding rules and conventions
├── STRATEGY_AND_DECISIONS.md        ← architecture decisions with rationale
├── DAX_CONSOLIDATED_DOCUMENTATION.md ← full brief for new AI sessions
└── DAX/
    ├── sac_dax_cleanrl.py           ← dual-rate SAC training loop
    ├── strategy.py                  ← candlestick/pivot pattern generator
    ├── requirements.txt
    ├── agents/
    │   ├── aux_cnn_extractor.py     ← CNN backbone for main policy (743-dim)
    │   ├── guard_actor.py           ← guard actor + critic (23-dim obs, 2-dim action)
    │   └── behavior_cloning.py      ← BC loss as pure function
    ├── envs/
    │   ├── dax_trading_env.py       ← DAX Gymnasium environment (728-dim obs)
    │   ├── dual_rate_env.py         ← M1+1s dual-rate cadence wrapper
    │   └── wrappers.py              ← AFPAugmentedEnv (728 → 743-dim)
    ├── features/
    │   ├── feature_schema.py        ← SINGLE SOURCE OF TRUTH for all dims
    │   └── feature_extractor.py     ← L2, pivots, MTF, tick computations
    ├── afp/
    │   ├── afp_model_v2.py          ← AFPCNNLSTM (15 scalar outputs)
    │   ├── train_afp.py
    │   └── evaluate_afp.py
    ├── rewards/
    │   └── risk_first_reward.py     ← risk-first hierarchical reward
    ├── tests/                       ← pytest suite (Phases 1–5)
    └── archive/                     ← superseded plan and context files
```

---

## Key Constants (features/feature_schema.py)

| Constant | Value | Meaning |
|----------|-------|---------|
| `BASE_OBS_DIM` | 728 | Main obs before AFP (672 seq + 56 scalars) |
| `AUGMENTED_OBS_DIM` | 743 | Main obs after AFP augmentation |
| `GUARD_OBS_DIM` | 23 | Guard policy observation |
| `MAIN_ACTION_DIM` | 3 | Main action dimensions |
| `GUARD_ACTION_DIM` | 2 | Guard action dimensions |
| `BAR_FEATURE_DIM` | 21 | Features per bar in CNN sequence |
| `SEQ_LEN` | 32 | Bars in CNN input window |
| `AFP_OUTPUT_DIM` | 15 | AFP model output heads |

---

## Long-term Targets

- Annual return: ≥ +35%
- Max drawdown: ≤ 10% per year
- Style: Defensive intraday — no overnight risk
- Long-only currently; shorts planned after baseline is met

## Roadmap (post-Phase 5)

1. Train AFP on available L2 data (`afp/train_afp.py`)
2. Verify AFP confidence calibration (`afp/evaluate_afp.py`)
3. Short SAC training run (~50k steps) to verify loss curves
4. Full training when ~12 months L2 data available (rented GPU)
5. Adaptive position sizing using AFP per-horizon confidence
6. Cross-asset features from top-10 DAX stocks (when L2 logging produces data)
7. Short-side entries (after long-only baseline is validated)
8. Live execution wrapper (IBKR/TWS)

---

*This README is the canonical entry point for the project. For new collaborators or AI assistants, start here.*
