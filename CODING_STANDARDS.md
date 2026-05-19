# CODING_STANDARDS.md — Rules and Conventions for All Code and Documentation

**Applies to:** All code, documentation, plans and commits in the DAX RL project.  
**Last updated:** 2026-05-19

---

## 1. The Permanent Rule (Non-Negotiable)

> **Lös ALLTID problem på det RIKTIGA och korrekta sättet från början.**  
> ALDRIG quick fixes, workarounds, tillfälliga lösningar eller genvägar.  
> ALDRIG ändringar som "får det att köra nu men kan skapa problem senare."  
> Om något är komplext — gör det korrekt och förklara varför.  
> Tid är ALDRIG ett problem. Kvalitet och långsiktig stabilitet är det enda som räknas.  
> Om du är osäker på den riktiga lösningen — säg det tydligt istället för att föreslå en genväg.

Any violation of this rule is grounds for immediate rollback, regardless of how small the change is.

---

## 2. Language Policy

- **Swedish** for owner-to-AI discussion and strategy conversations.
- **English** for all code, comments, commit messages, and documentation files.
- Never mix languages within a single code comment or commit message.

---

## 3. Code Quality Standards

- **No framework lock-in.** All modules use only PyTorch + Gymnasium + NumPy/Pandas. No Stable-Baselines3, Ray, or other heavy RL frameworks.
- **feature_schema.py is the single source of truth.** Every dimension constant (`BASE_OBS_DIM`, `GUARD_OBS_DIM`, etc.) must be imported from `DAX/features/feature_schema.py`. Never hardcode integers for observation or action dimensions.
- **Pure functions preferred.** Implement stateless logic as pure functions where possible (see `behavior_cloning.py`, `risk_first_reward.py`).
- **No comments by default.** Only add a comment when the *why* is non-obvious: a hidden constraint, a subtle invariant, or a workaround for a specific bug. Don't comment what the code does — well-named identifiers do that.
- **Testing required.** Every new feature must include at least a minimal smoke test or pytest check before merge.
- **Error handling.** Never swallow exceptions silently. Raise with meaningful context. Only validate at system boundaries (user input, external files); trust internal guarantees.
- **No defensive bloat.** Don't add error handling, fallbacks, or validation for scenarios that cannot occur. Don't add features, abstractions, or refactors beyond what the task requires.

---

## 4. Observation and Action Dimensions

This table is the quick reference. The authoritative source is always `feature_schema.py`.

| Policy | Obs dim | Action dim |
|--------|---------|------------|
| Main (M1 cadence) | 743 (AUGMENTED_OBS_DIM) | 3 (MAIN_ACTION_DIM) |
| Guard (1s cadence) | 23 (GUARD_OBS_DIM) | 2 (GUARD_ACTION_DIM) |

Base obs (pre-AFP): 728 = 32 bars × 21 features (672) + 56 current scalars.

---

## 5. Documentation Standards

- Use Markdown with clear headings (`##`, `###`).
- Keep sentences short and precise.
- Every document must state its purpose in the first paragraph.
- **Update the three canonical files** (README.md, this file, STRATEGY_AND_DECISIONS.md) instead of creating new PLAN.md files. Superseded plan files go in `DAX/archive/`.
- Dates in ISO format (YYYY-MM-DD) or Swedish month names for historical context (e.g. *Maj 2026*).
- Never leave "TODO" or "FIXME" without an owner and a deadline.

---

## 6. Commit Rules

- Commit messages in English, starting with a verb: `Add`, `Fix`, `Refactor`, `Remove`, `Update`.
- Every commit that changes behavior must be traceable to a decision in `STRATEGY_AND_DECISIONS.md`.
- Never skip pre-commit hooks (`--no-verify`). If a hook fails, fix the underlying problem.
- Never amend commits that have been pushed to remote. Create a new commit instead.
- Never force-push to `main`.

---

## 7. File and Directory Conventions

- Python files: `snake_case.py`
- Canonical documentation: `UPPER_SNAKE_CASE.md`
- All DAX-related code lives under `DAX/`.
- Superseded plan files (FIXES_PLAN.md, DUAL_RATE_ENV_PLAN.md, etc.) go in `DAX/archive/` — never delete without owner approval.
- Test files: `tests/test_<topic>.py`, always runnable with `pytest tests/`.

---

## 8. Anti-Patterns (Lessons Learned)

These have caused bugs or wasted sessions:

- **Don't rationalize away code** when a session context fills up. If a session is getting long, summarize state explicitly and start fresh rather than simplifying the existing code to fit.
- **Don't silently accept dimension assumptions.** If something seems off, verify against `feature_schema.py` before proceeding.
- **Don't conflate main and guard policies.** They have different obs dimensions, action spaces, networks, buffers, and update cadences.
- **Don't let AFP gradients flow during SAC training.** AFP parameters must have `requires_grad_(False)`.
- **Don't mix main and guard transitions** in replay buffers. Buffer separation is architecturally required.

---

## 9. Owner Workflow

Understanding this helps Claude give better assistance:

- Owner discusses architecture, strategy and requirements in Claude chat.
- Implementation is handed to **Claude Code** on the desktop PC.
- Owner reviews code, runs tests, and checks output on laptop in evening sessions.
- Large training runs use a rented GPU.
- When generating plans or code: verbose explanations of *why*, self-contained tests for every non-trivial change, references to relevant strategy decisions.

---

*Follow this file. When in doubt about a rule, the Permanent Rule in Section 1 takes priority over everything else.*
