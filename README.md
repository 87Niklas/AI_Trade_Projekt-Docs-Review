# DAX RL — Defensive Intraday Trading Agent

**Repository:** AI_Trade_Projekt (DAX module)
**Status:** v54.5 – Maj 2026 (Full L2 + AFP + 480-dim obs, dual-rate ready)

## Project Overview

This is a reinforcement learning system for intraday trading of DAX futures (FDAX). The agent is built on **CleanRL** and features a **dual-rate policy architecture** with a frozen **Auxiliary Future Prediction (AFP)** model and behavior cloning warmup.

**Primary Objective:** Achieve +35% annual return with maximum 10% annual drawdown through defensive, high-conviction day-trading.

The system processes multi-timeframe data (M1 primary + M5/M15 pivots), full 10-level L2 order book, candlestick/pivot patterns (HC0/HC2/HC4/HC8), Fibonacci levels and previous-day OHLC features.

## For the Owner (87Niklas)

This documentation set replaces the previous collection of scattered PLAN.md, CONTEXT.md and RULES.txt files with three canonical, maintainable documents:
- README.md (this file)
- CODING_STANDARDS.md
- STRATEGY_AND_DECISIONS.md

All future work must follow the standards and strategy defined in the two supporting files.

## For Claude / AI Assistants

When starting a new session, always read the three files in this order:
1. README.md (current context)
2. STRATEGY_AND_DECISIONS.md (why we do what we do)
3. CODING_STANDARDS.md (how we do it)

Never deviate from the permanent rules or architecture decisions without explicit approval.

## For New Collaborators

Welcome. The project follows strict engineering discipline:
- Every change must solve the problem the right way from the start.
- No quick fixes, workarounds or temporary solutions are accepted.
- All code must be framework-agnostic where possible (CleanRL port completed May 2026).

Start by reading the three core documents above.

## Quick Start

See CODING_STANDARDS.md for development workflow and STRATEGY_AND_DECISIONS.md for architectural rationale.

Key entry points:
- `DAX/sac_dax_cleanrl.py` – Main training script
- `DAX/dax_trading_env.py` – Custom Gymnasium environment
- `DAX/strategy.py` – Signal generation (standalone)

## Long-term Targets

- Annual return: ≥ +35%
- Max drawdown: ≤ 10%
- Style: Defensive intraday (no overnight risk)

---

*This README was created as part of the structural documentation cleanup (May 2026).*