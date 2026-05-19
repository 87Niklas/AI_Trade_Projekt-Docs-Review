# DAX RL — Consolidated Documentation (Post-Cleanup)

> **This file consolidates key information from multiple plan and context files into a single, clean reference.**

## 1. Project Goals and Strategy

**Long-term targets:** +35% annual return, max 10% annual drawdown. Defensive day-trading for FDAX.

Version 54.5 – Maj 2026 (Full L2 + AFP + 480-dim obs)

## 2. Architecture Overview

- Built on CleanRL with dual-rate policy architecture.
- Frozen Auxiliary Future Prediction (AFP).
- Behavior cloning warmup.
- Actor: AuxCnnExtractor backbone + Gaussian head.
- Critics: 2× AuxCnnExtractor backbone + linear Q head (double Q-learning).

## 3. Key Components

- strategy.py: Candlestick + Pivot pattern detection (HC0/HC2/HC4/HC8).
- dax_trading_env.py: DAX Trading Environment (CleanRL port).
- sac_dax_cleanrl.py: CleanRL SAC for DAX futures trading (v2).
- feature_extractor.py: Full 10-level L2 features, pivot detection, Fibonacci, higher-TF.
- risk_first_reward.py: v54 Risk-First Reward System.
- agents/: behavior_cloning.py, aux_cnn_extractor.py.
- afp/: AFP training and dataset builders.
- rewards/, features/, envs/ directories.

## 4. Development Phases Completed

- Phase 1: Core implementation (inferred from files).
- Phase 2: AFP Architecture (15 outputs, future_price_5, confidence heads).
- Phase 3: Multi-Timeframe Pivots (M5/M15 pivot detection, previous-day OHLC).
- Phase 4: Dual-Rate Environment (M1 and 1-second tick cadence).
- Phase 5: Wrapper Integration (AFP wrapper, guard policy, dual-rate SAC training).

## 5. Permanent Rules

(As listed in README.md above)

## 6. Next Steps / Open Items

Refer to original FIXES_PLAN.md, MTF_PIVOTS_PLAN.md, DUAL_RATE_ENV_PLAN.md, WRAPPER_INTEGRATION_PLAN.md, AFP_ARCHITECTURE_PLAN.md for detailed implementation plans.

---

*Generated following structural cleanup of project documentation. All plan files consolidated for clarity and maintainability.*

**Note:** This is a review copy. Original files remain in the main repository.