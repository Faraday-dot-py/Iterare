"""Codex CLI runner for iterare research tasks.

Wraps the `codex exec` and `codex review` CLI commands for programmatic use.
Codex is OpenAI's autonomous coding agent — it reads the repository, plans,
writes code, and runs shell commands to accomplish tasks.

## Core capabilities

- **Script generation**: write new TIDE worker scripts from a spec + reference
- **Result analysis**: read JSON result files and produce structured summaries
- **Code review**: check scripts for bugs before TIDE submission
- **Structured output**: extract machine-readable data via JSON schema

## Sandbox modes

- `read-only`       — agent can read files; no writes or shell execution
- `workspace-write` — agent can read + write files in workspace; no shell
- `danger-full-access` — full disk + shell; use only for trusted prompts

## Usage

    from code.tools.codex.runner import CodexRunner

    runner = CodexRunner(workspace="/home/awebb/Research/iterare")

    # Write a new experiment script
    runner.write_script(
        prompt="Write Exp36: seeds 7-10 at PREFIX_LEN=16, based on manager-34/worker-1/multiseed_len16_seeds3_6.py",
        output_path="tasks/steer-001/manager-36/worker-1/multiseed_len16_seeds7_10.py",
    )

    # Analyse results (read-only, returns text summary)
    summary = runner.analyze(
        "Summarise scaling trends from all scale_prefix_*_results.json files in tasks/steer-001/"
    )

    # Code review before TIDE submission
    issues = runner.review_file("tasks/steer-001/manager-36/worker-1/multiseed_len16_seeds7_10.py")

## Notes
- Always restrict to the iterare workspace (`workspace` arg) to avoid touching
  unrelated files.  Codex is given `--cd <workspace>` and `--add-dir` is NOT
  set, so it cannot stray outside the workspace tree.
- Default model is codex-mini-latest (ChatGPT Plus account).  Switch to o3
  with model="o3" if you have an API key (set OPENAI_API_KEY).
- Session files are ephemeral by default (no disk persistence between calls).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# Default codex binary (snap install puts it here)
_CODEX_BIN = "codex"

# Workspace root — scripts are always relative to this
_DEFAULT_WORKSPACE = Path(__file__).resolve().parents[3]  # repo root


class CodexRunner:
    """High-level wrapper around the `codex` CLI.

    Parameters
    ----------
    workspace:
        Absolute path to the directory Codex operates inside.  All file paths
        in prompts should be relative to this.  Defaults to the iterare repo root.
    model:
        Codex model string.  None → use CLI default (codex-mini-latest with a
        ChatGPT account, or set OPENAI_API_KEY for API access).
    default_sandbox:
        Sandbox mode applied when not overridden per-call.
    timeout:
        Max seconds to wait for a codex subprocess to finish.
    """

    def __init__(
        self,
        workspace: str | Path | None = None,
        model: str | None = None,
        default_sandbox: str = "workspace-write",
        timeout: int = 300,
    ):
        self.workspace = Path(workspace or _DEFAULT_WORKSPACE).resolve()
        self.model = model
        self.default_sandbox = default_sandbox
        self.timeout = timeout

    # ── Low-level exec ────────────────────────────────────────────────────────

    def exec(
        self,
        prompt: str,
        *,
        sandbox: str | None = None,
        output_file: str | Path | None = None,
        schema_file: str | Path | None = None,
        extra_args: list[str] | None = None,
        capture_stderr: bool = False,
    ) -> str:
        """Run ``codex exec --full-auto`` with the given prompt.

        Returns the agent's final text response.

        Parameters
        ----------
        prompt:       Natural-language task description.
        sandbox:      Override ``default_sandbox`` for this call.
        output_file:  If given, Codex writes its last message here and we also
                      read + return that content.
        schema_file:  Path to a JSON Schema file for structured output.
        extra_args:   Additional raw CLI flags appended verbatim.
        capture_stderr: Include stderr in the returned string.
        """
        cmd = [_CODEX_BIN, "exec", "--full-auto", "--ephemeral"]
        cmd += ["-C", str(self.workspace)]
        cmd += ["-s", sandbox or self.default_sandbox]
        if self.model:
            cmd += ["-m", self.model]
        if output_file:
            cmd += ["-o", str(output_file)]
        if schema_file:
            cmd += ["--output-schema", str(schema_file)]
        if extra_args:
            cmd += extra_args
        cmd.append(prompt)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env={**os.environ},
        )

        # If output_file was specified and written, return that content
        if output_file and Path(output_file).exists():
            return Path(output_file).read_text()

        out = result.stdout
        if capture_stderr and result.stderr:
            out = out + "\n--- stderr ---\n" + result.stderr
        return out.strip()

    # ── High-level helpers ────────────────────────────────────────────────────

    def write_script(
        self,
        prompt: str,
        output_path: str | Path,
        *,
        reference_path: str | Path | None = None,
        timeout: int | None = None,
    ) -> str:
        """Ask Codex to write a Python script to ``output_path``.

        The prompt is augmented with the output path and (optionally) a pointer
        to a reference script to model the new one after.

        Returns Codex's final message (usually a brief confirmation).
        """
        full_prompt = prompt.rstrip()
        if reference_path:
            full_prompt += (
                f"\n\nBase the new script on the structure of "
                f"`{reference_path}`. Copy the boilerplate (imports, notify helper, "
                f"logging helpers, reference-completion generation, HotFlip loop, "
                f"result-saving) verbatim — only change the config block and "
                f"experiment metadata."
            )
        full_prompt += f"\n\nWrite the complete file to `{output_path}`."

        old_timeout, self.timeout = self.timeout, timeout or self.timeout
        try:
            return self.exec(full_prompt, sandbox="workspace-write")
        finally:
            self.timeout = old_timeout

    def analyze(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> str | dict:
        """Run a read-only analysis task and return the result.

        If ``schema`` is provided, Codex is instructed to emit JSON matching
        that schema and the return value is a parsed dict.
        """
        schema_file = None
        tmp_path = None
        try:
            if schema:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".json", mode="w", delete=False
                )
                json.dump(schema, tmp)
                tmp.flush()
                tmp_path = tmp.name
                schema_file = tmp_path

            old_timeout, self.timeout = self.timeout, timeout or self.timeout
            try:
                raw = self.exec(prompt, sandbox="read-only", schema_file=schema_file)
            finally:
                self.timeout = old_timeout

            if schema:
                # Extract JSON from the response
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    # Find JSON block in response
                    import re
                    m = re.search(r"\{.*\}", raw, re.DOTALL)
                    if m:
                        return json.loads(m.group())
                    return raw
            return raw
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def review_file(self, file_path: str | Path, *, extra_context: str = "") -> str:
        """Ask Codex to review a script for bugs and issues.

        Uses read-only sandbox.  Returns the review text.
        """
        prompt = (
            f"Review the file `{file_path}` for correctness. "
            "Focus on: (1) config constants match the docstring, "
            "(2) OUT_PATH / CKPT_PATH filenames are unique (don't reuse names "
            "from previous experiments), (3) SEEDS list matches the experiment intent, "
            "(4) notify() calls have the correct experiment number, "
            "(5) results dict has the correct experiment name string, "
            "(6) no off-by-one errors in HotFlip or soft-opt loops."
        )
        if extra_context:
            prompt += f"\n\nAdditional context: {extra_context}"
        return self.exec(prompt, sandbox="read-only", timeout=120)


# ── Module-level convenience functions ───────────────────────────────────────

def codex_exec(
    prompt: str,
    workspace: str | Path | None = None,
    sandbox: str = "read-only",
    model: str | None = None,
    timeout: int = 300,
) -> str:
    """One-shot Codex exec — convenience wrapper for simple calls."""
    runner = CodexRunner(workspace=workspace, model=model, timeout=timeout)
    return runner.exec(prompt, sandbox=sandbox)


def codex_review_file(
    file_path: str | Path,
    workspace: str | Path | None = None,
) -> str:
    """One-shot code review for a single file."""
    runner = CodexRunner(workspace=workspace)
    return runner.review_file(file_path)
