# STRATEGY_AND_DECISIONS.md — Architecture Decisions with Rationale

**Version:** Phase 5 complete — 2026-05-19  
**Scope:** All architectural and strategic choices for the DAX RL trading agent.

This is the single source of truth for why things are the way they are. All code changes must be traceable to a decision recorded here. New decisions must be appended before implementation begins.

---

## 1. Core Philosophy

We build a **defensive, high-conviction intraday system** for FDAX. The agent must generate consistent alpha while strictly limiting downside. Profit is a by-product of good risk management.

> "It is not how much you make on a trade that counts — it is how little you lose when you are wrong."

**Non-negotiable constraints:**
- No overnight positions
- Max annual drawdown ≤ 10%
- Target annual return ≥ +35%
- Long-only until the long-side baseline is validated

---

## 2. Architecture Decisions

### Decision 1: Dual-Rate Policy Architecture (Phase 4)

**Choice:** Two fully separate SAC policies — a main policy at M1 cadence and a guard policy at 1-second cadence. Separate actor/critic networks, separate replay buffers, separate optimizers.

**Rationale:** A single policy cannot efficiently learn both time scales. M1 bars provide stable, rich features for entry decisions. 1-second ticks are required for precise exit timing and stop-loss management. Sharing a backbone would entangle the two time scales and prevent independent learning rates and network sizes.

**Implementation:**
- Main: `AuxCnnExtractor` backbone (743-dim obs → 256-dim features), 3-dim action
- Guard: small MLP (23-dim obs → 64 → 32 hidden), 2-dim action
- Networks defined in `agents/aux_cnn_extractor.py` and `agents/guard_actor.py`

---

### Decision 2: Frozen AFP as External Feature Provider (Phase 2 + 5)

**Choice:** Pre-train a 15-output AFP model on live-flow L2 data. Freeze all AFP weights during SAC training. Inject AFP outputs into the main observation via an environment wrapper (`envs/wrappers.py`). AFP does not run during guard steps — its context is frozen at each M1 boundary.

**Rationale:** Future price prediction is a valuable auxiliary signal that improves main-policy representations. Freezing after pre-training stabilises SAC gradients (no moving target from joint training), reduces compute, and decouples the two training pipelines. The wrapper architecture keeps AFP as a pure feature provider with no gradient coupling to SAC.

**AFP outputs (15 scalar heads):**
- `future_price_{5,10,20,30}` — predicted price 5/10/20/30 bars ahead
- `confidence_{5,10,20,30}` — model confidence per horizon, sigmoid-bounded [0,1]
- `pred_high`, `pred_low` — predicted bar high/low
- `volatility_forecast` — predicted volatility
- `direction_prob_{up,down,neutral}` — softmax direction probabilities
- `fib_break` — Fibonacci level break probability

---

### Decision 3: Behavior Cloning Warmup (Phase 0)

**Choice:** Supervised pre-training of the main actor on high-quality human/expert trajectories for the first `bc_steps` (default 80k) of SAC training. Guard policy does not use BC.

**Rationale:** Pure RL from scratch on financial data is sample-inefficient and can explore dangerous regions of the action space early in training. BC provides a profitable starting policy. The imitation labels flag when HC patterns fired — they indicate potential entry quality, not guaranteed trade quality. BC is therefore a weak attractor by design and decays as SAC takes over.

**Applied to:** Main policy only. BC data is 728-dim (pre-AFP); the actor receives 743-dim obs during training — the extra 15 AFP dims are zero-padded during BC warmup unless real AFP weights are loaded.

---

### Decision 4: Feature Schema as Single Source of Truth (Phase 1)

**Choice:** `DAX/features/feature_schema.py` defines all dimension constants, feature names, and layout descriptions. Every consumer (env, extractor, networks, tests) imports from there. No hardcoded integers for observation or action dimensions anywhere in the codebase.

**Rationale:** Dimension mismatches across files caused silent bugs that were extremely hard to trace. A single source of truth makes schema changes safe and verifiable — change one file, run the test suite, done.

**Key constants:** `BASE_OBS_DIM=728`, `AUGMENTED_OBS_DIM=743`, `GUARD_OBS_DIM=23`, `MAIN_ACTION_DIM=3`, `GUARD_ACTION_DIM=2`, `BAR_FEATURE_DIM=21`, `SEQ_LEN=32`, `AFP_OUTPUT_DIM=15`.

---

### Decision 5: Full L2 + Multi-Timeframe Feature Set (Phase 1 + 3)

**Choice:** 728-dim base observation: 32-bar CNN sequence (21 features per bar) + 56 current scalars. Features include full 10-level L2, M1/M5/M15 pivots, previous-day OHLC, Fibonacci retracements, tick microstructure, and 7 individually-named candlestick pattern flags.

**Rationale:** L2 data contains the true market microstructure. Higher timeframes reduce false entries. Naming patterns individually (e.g. `pattern_hammer_long` instead of `hc0_combined`) allows the agent to learn which patterns predict profitable trades instead of treating all as equivalent.

**Note:** Cross-asset features from top-10 DAX stocks (group O in current scalars) are reserved placeholders — 0 active dimensions until the stocks L2 logging pipeline is ready.

---

### Decision 6: CleanRL Migration (Phase 0, completed May 2026)

**Choice:** Complete migration from Stable-Baselines3 to pure CleanRL + custom modules. No RL framework dependencies except PyTorch + Gymnasium + NumPy/Pandas.

**Rationale:** SB3 abstractions hid critical training details (gradient flow, callback timing, buffer sampling) and made debugging impossible. CleanRL gives full control over the training loop, which is essential for the dual-rate routing, AFP wrapper integration, and per-policy alpha tuning.

---

### Decision 7: Reward Sharing with Guard Shaping

**Choice:** Both main and guard policies receive the same trade-close reward (from `rewards/risk_first_reward.py`). The guard policy additionally receives per-second shaping: positive for keeping winning positions alive, negative for unnecessary exits.

**Rationale:** Trade-close rewards are sparse and delayed. The guard needs denser signal to learn stop-loss management within a bar. Per-second shaping provides this without changing the overall reward structure that the main policy trains on.

---

## 3. Strategy: Seed vs. Discovery

The PRT seed strategy uses 7 candlestick patterns + falling-tops/rising-bots pivots (HC2) + trend filters (HC4) as hard-coded entry rules with fixed 30p SL / 60p TP.

**The RL agent extends this with:**
- L2 microstructure analysis (microprice, OFI, aggression ratio, order flow imbalance)
- Tick-level activity and urgency detection
- Multi-timeframe pivots for adaptive SL/TP sizing
- Frozen AFP future price predictions with per-horizon confidence
- 1-second guard policy for in-bar position management

**The agent should not simply copy the seed strategy.** BC gives a starting point, but the agent must discover its own patterns via L2 and tick microstructure. BC is a weak attractor, not a constraint.

---

## 4. Anti-Patterns to Avoid

Lessons learned — these have caused bugs or wasted sessions in the past:

- **Never bypass `feature_schema.py`** with hardcoded dimension constants, even "just temporarily."
- **Never conflate main and guard observations** — they have different dimensions, different features, and different update cadences.
- **Never rationalize away code** when context fills up in a long session. Summarize state explicitly and start a fresh session instead.
- **Never silently accept dimension assumptions** — verify against `feature_schema.py` first.
- **Never use AFP gradients during SAC training** — AFP weights must be frozen (`requires_grad_(False)`).
- **Never include guard transitions in `rb_main` or vice versa** — buffer separation is critical.

---

## 5. Common Design Decisions Reference

**"Should this feature go in main obs or guard obs?"**
- Main only: changes only at M1 (e.g. higher-TF features, HC patterns, Fibonacci levels)
- Guard only: tick-fresh state critical for real-time risk management
- Both: feature needed at different cadences (guard gets tick-fresh value; main gets M1-frozen historical context)

**"Should this go in per-bar sequence or current scalars?"**
- Per-bar sequence: history matters — CNN should learn temporal patterns from it
- Current scalars: state at "now" — agent uses for immediate decision only

**"Add feature now or as extension hook?"**
- Now: computable from existing data and clearly valuable
- Extension hook: depends on unavailable data (e.g. stocks L2) or unclear added value until baseline is measured

---

## 6. Decision Log

| Date | Decision | Notes |
|------|----------|-------|
| 2026-05 | Phase 5 complete: AFP wrapper + guard policy + dual-rate SAC | All smoke tests passing |
| 2026-05 | Phase 4 complete: DualRateEnv (M1 + 59×1s guard per bar) | 15/15 Phase 4 tests passing |
| 2026-05 | Phase 3 complete: M5/M15 pivot detection + prev-day OHLC | |
| 2026-05 | Phase 2 complete: AFP model v2 (15 heads) | |
| 2026-05 | Phase 1 complete: feature_schema.py, 728-dim obs end-to-end | |
| 2026-05 | Phase 0 complete: CleanRL migration, BC gradient fix, CNN layout fix | |

Any new strategic decision must be appended here with date and clear rationale.

---

*This document is the single source of truth for strategy and architecture. Read it before making any architectural decision.*
