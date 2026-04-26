---
name: Auto-approve echo commands
description: Never prompt for permission before running echo (or echo *) commands
type: feedback
---

Always run `echo` commands (including `echo *` glob expansions) without asking for approval.

**Why:** User explicitly said not to ask again for `echo *`.

**How to apply:** Treat all `echo` variants as safe, non-destructive shell output — run them directly like `ls` or `cat`.
