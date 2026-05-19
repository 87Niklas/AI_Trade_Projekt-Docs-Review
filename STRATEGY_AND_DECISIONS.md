# STRATEGY_AND_DECISIONS.md — Strategy Philosophy and Architecture Decisions with Rationale

**Version:** 54.5 – Maj 2026
**Scope:** All architectural and strategic choices for the DAX RL trading agent.

## 1. Core Philosophy

We build a **defensive, high-conviction intraday system** for FDAX. The agent must generate consistent alpha while strictly limiting downside. Profit is secondary to capital preservation.

**Non-negotiable constraints:**
- No overnight positions
- Max annual drawdown ≤ 10%
- Target annual return ≥ +35%

## 2. Key Architecture Decisions & Rationale

### Decision 1: Dual-Rate Policy Architecture (Phase 4)
**Choice:** Separate fast (1-second tick) guard policy and slower (M1 bar) main policy that share the same actor-critic backbone but operate at different cadences.
**Rationale:** M1 bars provide stable features and sufficient signal for entry decisions. 1-second ticks are required for precise exit timing and to avoid slippage on large orders. A single policy cannot efficiently learn both time scales. Freezing the AFP head during main training prevents catastrophic forgetting of future-price prediction.

### Decision 2: Frozen Auxiliary Future Prediction (AFP) Model (Phase 2)
**Choice:** Pre-train a 15-output AFP model (price + confidence for 5 horizons) on live-flow data and freeze it during SAC training.
**Rationale:** Future price prediction is a valuable auxiliary task that improves representation learning. Freezing it after pre-training stabilizes the main policy gradients and reduces compute. 15 outputs (instead of original 9) give richer supervision signals.

### Decision 3: Behavior Cloning Warmup + Risk-First Reward (Phase 5)
**Choice:** Use behavior cloning on high-quality human/expert trajectories for the first 50k steps, then switch to pure SAC with a custom risk-first reward that heavily penalizes drawdown and over-exposure.
**Rationale:** Pure RL from scratch on financial data is sample-inefficient and dangerous. Behavior cloning provides a safe, profitable starting policy. The risk-first reward (see risk_first_reward.py) directly optimizes the primary constraint (max 10% DD) rather than only maximizing return.

### Decision 4: Full L2 + Multi-Timeframe Features (Phase 3)
**Choice:** 480-dimensional observation vector containing 10-level L2, M1/M5/M15 pivots, previous-day OHLC, Fibonacci retracements and HC-pattern detections.
**Rationale:** L2 data contains the true market microstructure. Higher timeframes provide context that reduces false entries. All features are computed on-the-fly with zero external dependencies.

### Decision 5: CleanRL Port (May 2026)
**Choice:** Complete migration from Stable-Baselines3 to pure CleanRL + custom modules (AuxCnnExtractor, behavior cloning as pure functions, etc.).
**Rationale:** SB3 abstractions hid important details and made debugging difficult. CleanRL gives full control over the training loop, which is essential for the dual-rate and AFP-wrapper integration.

## 3. Future Evolution Path

- Phase 6+: Live execution wrapper, position sizing based on conviction, multi-asset extension (if drawdown target remains met).
- All future phases must be documented as updates to this file before implementation begins.

## 4. Decision Log (Summary)

- 2026-05: Dual-rate + AFP wrapper integration completed
- 2026-05: Full L2 + MTF pivots added
- 2026-05: CleanRL migration finalized

Any new strategic decision must be appended here with clear rationale.

---

*This document is the single source of truth for strategy and architecture. All code changes must be traceable to decisions recorded here.*