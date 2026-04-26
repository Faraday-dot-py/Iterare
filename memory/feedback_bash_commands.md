---
name: Avoid multi-line inline scripts
description: Never write multi-line scripts (bash or Python) inline; write to a file and run it
type: feedback
---

Never write multi-line scripts inline — this applies to both `python3 -c "..."` and multi-line bash scripts (for loops, etc.). Always write the script to a temp file and run it.

**Why:** User has corrected this multiple times. It's in CLAUDE.md explicitly for Python, and the user confirmed it extends to bash scripts too.

**How to apply:** Any time I need to run a script with more than ~1 line of logic, write it to `/tmp/something.sh` or `/tmp/something.py` using the Write tool, then run it with Bash. Single-line bash commands (git status, ls, cat) are still fine inline.
