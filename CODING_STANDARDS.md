# CODING_STANDARDS.md — Rules and Conventions for All Code and Documentation Work

**Applies to:** All code, documentation, plans and commits in the DAX RL project.

## 1. Permanent Project Rule (Non-Negotiable)

PERMANENT REGEL FÖR DETTA PROJEKT (DAX RL):

Från och med nu gäller följande strikt:

- Lös ALLTID problem på det RIKTIGA och korrekta sättet från början.
- ALDRIG quick fixes, workarounds, tillfälliga lösningar eller genvägar.
- ALDRIG ändringar som "får det att köra nu men kan skapa problem senare".

Any violation of this rule is grounds for immediate rollback.

## 2. Code Quality Standards

- **No framework lock-in:** All new modules must be written without Stable-Baselines3, Ray or other heavy RL frameworks. Use only PyTorch + Gymnasium + NumPy/Pandas.
- **Pure functions preferred:** Where possible, implement logic as pure functions (see behavior_cloning.py and risk_first_reward.py as examples).
- **Type hints and docstrings:** Every public function and class must have clear docstrings and type hints.
- **Modularity:** Keep files focused. Large classes (e.g. trading env) may be split when they exceed ~400 lines.
- **Testing:** Every new feature must include a minimal smoke test or verification script before merge.
- **Error handling:** Never swallow exceptions. Log meaningful context and raise.

## 3. Documentation Standards

- Use Markdown with clear headings (##, ###).
- Keep sentences short and precise.
- Every document must state its purpose in the first paragraph.
- Update the three canonical files (README.md, this file, STRATEGY_AND_DECISIONS.md) instead of creating new PLAN.md files.
- Dates must be in ISO format (YYYY-MM-DD) or Swedish month names when historically relevant (e.g. Maj 2026).
- Never leave "TODO" or "FIXME" without an owner and deadline.

## 4. Commit & Review Rules

- Commit messages must be in English and start with a verb (e.g. "Add", "Fix", "Refactor").
- Every commit that changes behavior must reference the relevant section in STRATEGY_AND_DECISIONS.md.
- Code review (even self-review) must verify compliance with the Permanent Rule above.

## 5. File Naming & Structure

- Use snake_case for Python files.
- Documentation files use UPPER_SNAKE_CASE.md or Title_Case.md only for the three canonical files.
- Keep all DAX-related code under DAX/.
- Archive obsolete plan files in DAX/archive/ (do not delete without owner approval).

---

*Last updated: 2026-05-19 as part of documentation cleanup.*