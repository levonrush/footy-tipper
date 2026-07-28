"""Shared console reporting for the footy-tipper CLI.

One reporter, used across every command, so the tool speaks with a single
voice: what is happening, the numbers that matter, what moved where, and —
when a step fails — a plain-language reason instead of a buried stack trace.

Two output regimes coexist because the CLI shells out to compute scripts:

* In-process code (send, status, update-model orchestration) calls the
  ``Reporter`` directly and gets live spinners, panels and coloured notes.
* Child scripts (train.py, evaluate.py, inference.py, nrl_data.py, odds.py)
  cannot draw on the operator's terminal — their stdout is captured. They
  emit small, greppable *markers* via :func:`emit_progress` /
  :func:`emit_result`; the parent's subprocess pump (:func:`pump_process`)
  turns those into the live line and end-of-step panels, and quietly tees
  everything else to a log file instead of spamming the console.

Rich is used when it is installed *and* attached to a terminal; otherwise the
identical API degrades to clean, static, uncoloured lines that are safe for
GitHub Actions logs and for bare checkouts where rich is not installed.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time

try:  # rich is optional; the CLI must still run without it.
    from rich.box import ROUNDED as _ROUNDED
    from rich.console import Console as _RichConsole
    from rich.live import Live as _Live
    from rich.panel import Panel as _Panel
    from rich.spinner import Spinner as _Spinner
    from rich.table import Table as _Table
    from rich.text import Text as _Text

    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only where rich is absent
    _RICH_AVAILABLE = False


# Markers a captured child prints so the parent can lift its progress and
# results back onto the operator's console. Chosen to be unlikely to collide
# with real output while staying readable if a child is run on its own.
PROGRESS_MARKER = "::ft:progress::"
RESULT_MARKER = "::ft:result::"

# How often the plain (non-terminal) fallback is allowed to reprint a changing
# progress line. Terminal output updates in place and ignores this.
_NONTTY_PROGRESS_INTERVAL = 12.0


def _fmt_elapsed(seconds: float) -> str:
    whole = int(max(0, seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _env_quiet() -> bool:
    return os.getenv("FOOTY_TIPPER_QUIET", "").strip().lower() in {"1", "true", "yes"}


# --------------------------------------------------------------------------- #
# Plain-language failure hints
# --------------------------------------------------------------------------- #

_FAILURE_HINTS = (
    (
        re.compile(
            r"lazy-load database.*is corrupt|R_decompress1|internal error \d+ in R_decompress",
            re.I,
        ),
        "An R package cache got corrupted (this is the toolchain, not your data).\n"
        "Reinstall the broken package, e.g.  R -e 'install.packages(\"rlang\")'\n"
        "or recreate the conda env, then run the same command again to resume where it stopped.",
    ),
    (
        re.compile(r"there is no package called ['\"]?([\w.]+)", re.I),
        "R is missing a package it needs. Install it with:\n"
        "  R -e 'install.packages(\"<name>\")'\n"
        "then run the same command again.",
    ),
    (
        re.compile(
            r"failed the honest nested acceptance gate|Acceptance gate: FAIL|failed acceptance",
            re.I,
        ),
        "The new model was NOT clearly better than the market / the current model,\n"
        "so nothing was activated. This is the safety gate doing its job — your live\n"
        "tips are unchanged. You can keep the current model, or try again later once\n"
        "more results are in.",
    ),
    (
        re.compile(r"ModuleNotFoundError: No module named ['\"]?([\w.]+)", re.I),
        "A Python package is missing from this environment. Activate the conda env\n"
        "(`conda activate footy-tipper`) or `pip install -r requirements.txt`, then rerun.",
    ),
    (
        re.compile(r"database is locked", re.I),
        "The SQLite database is open somewhere else. Close any other footy-tipper run\n"
        "(or a DB browser) and run the same command again.",
    ),
)


def explain_failure(text: str | None) -> str | None:
    """Return a plain-language hint for a known failure, or None."""
    if not text:
        return None
    for pattern, hint in _FAILURE_HINTS:
        if pattern.search(text):
            return hint
    return None


# --------------------------------------------------------------------------- #
# Marker helpers (child side) and the subprocess pump (parent side)
# --------------------------------------------------------------------------- #

def emit_progress(text: str) -> None:
    """Child-side: announce the current phase so the parent's live line updates."""
    print(f"{PROGRESS_MARKER} {text}", flush=True)


def emit_result(kind: str, **payload) -> None:
    """Child-side: emit a structured result the parent can render (tips, counts…)."""
    record = {"kind": kind, **payload}
    print(f"{RESULT_MARKER} {json.dumps(record, default=float)}", flush=True)


def _parse_marker(line: str):
    stripped = line.strip()
    if stripped.startswith(PROGRESS_MARKER):
        return "progress", stripped[len(PROGRESS_MARKER):].strip()
    if stripped.startswith(RESULT_MARKER):
        return "result", stripped[len(RESULT_MARKER):].strip()
    return None, None


def pump_process(stdout, log, reporter, *, results=None) -> None:
    """Stream a child's stdout: tee raw lines to ``log``, lift markers to the UI.

    Everything the child prints is written to ``log`` (a text file handle) so the
    full transcript is always recoverable. Progress markers drive the active
    step's live line; result markers are decoded and appended to ``results`` for
    the caller to render after the step.
    """
    for raw in stdout:
        try:
            log.write(raw)
            log.flush()
        except Exception:
            pass
        kind, payload = _parse_marker(raw.rstrip("\n"))
        if kind == "progress":
            reporter.progress(payload)
        elif kind == "result" and results is not None:
            try:
                results.append(json.loads(payload))
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# The reporter
# --------------------------------------------------------------------------- #

class _Step:
    """A single long-running operation with a live spinner + elapsed line."""

    def __init__(self, reporter: "Reporter", label: str):
        self.reporter = reporter
        self.label = label
        self.start = time.monotonic()
        self._progress = ""
        self._summary = None
        self._live = None
        self._spinner = None
        self._prev = None
        self._last_text = None
        self._last_emit = 0.0

    def _elapsed(self) -> str:
        return _fmt_elapsed(time.monotonic() - self.start)

    def _renderable(self):
        grid = _Table.grid(padding=(0, 1))
        detail = _Text()
        detail.append(self.label, style="bold")
        if self._progress:
            detail.append(f"   {self._progress}", style="dim")
        detail.append(f"   {self._elapsed()}", style="cyan")
        grid.add_row(self._spinner, detail)
        return grid

    def _start(self) -> None:
        if self.reporter.quiet:
            return
        if self.reporter._animate:
            self._spinner = _Spinner("dots", style="cyan")
            self._live = _Live(
                self._renderable(),
                console=self.reporter._console,
                refresh_per_second=10,
                transient=True,
            )
            self._live.start()
        else:
            print(f"› {self.label}", file=self.reporter.stream, flush=True)

    def progress(self, text: str) -> None:
        self._progress = text
        if self.reporter.quiet:
            return
        if self._live is not None:
            self._live.update(self._renderable())
        else:
            now = time.monotonic()
            if text != self._last_text and (now - self._last_emit) >= _NONTTY_PROGRESS_INTERVAL:
                print(f"    {text}   ({self._elapsed()})", file=self.reporter.stream, flush=True)
                self._last_text = text
                self._last_emit = now

    def done(self, ok: bool = True, summary: str | None = None) -> None:
        if summary is not None:
            self._summary = summary
        self._finish(ok)
        self.reporter._current = self._prev

    def _finish(self, ok: bool) -> None:
        if self.reporter.quiet:
            return
        duration = self._elapsed()
        detail = self._summary if self._summary is not None else self._progress
        if self._live is not None:
            self._live.stop()
            self._live = None
        glyph = "✓" if ok else "✗"
        if self.reporter._rich:
            line = _Text()
            line.append(f"{glyph} ", style=("green" if ok else "red") + " bold")
            line.append(self.label, style="bold")
            if detail:
                line.append(f"   {detail}", style="dim")
            line.append(f"   ({duration})", style="cyan")
            self.reporter._console.print(line)
        else:
            tail = f"   {detail}" if detail else ""
            print(
                f"{glyph} {self.label}{tail}   ({duration})",
                file=self.reporter.stream,
                flush=True,
            )


class Reporter:
    """Consistent human output for the whole CLI."""

    def __init__(self, *, stream=None, quiet: bool = False, rich_enabled=None):
        self.stream = stream or sys.stdout
        self.quiet = quiet
        self._current = None
        if rich_enabled is None:
            rich_enabled = _RICH_AVAILABLE
        self._console = (
            _RichConsole(file=self.stream, highlight=False, soft_wrap=True)
            if rich_enabled
            else None
        )
        self._err_console = (
            _RichConsole(stderr=True, highlight=False, soft_wrap=True)
            if rich_enabled
            else None
        )
        self._rich = self._console is not None
        self._animate = bool(self._console and self._console.is_terminal and not quiet)

    # -- headers / notes ---------------------------------------------------- #

    def section(self, title: str, subtitle: str | None = None) -> None:
        if self.quiet:
            return
        if self._rich:
            text = _Text(title, style="bold cyan")
            if subtitle:
                text.append(f"   ·   {subtitle}", style="dim")
            self._console.rule(text, style="cyan")
        else:
            line = title + (f"   ·   {subtitle}" if subtitle else "")
            print("\n" + line, file=self.stream, flush=True)
            print("─" * min(max(len(line), 8), 60), file=self.stream, flush=True)

    def _note(self, message: str, *, glyph: str = "", style: str = "", source: str | None = None):
        if self.quiet:
            return
        if self._rich:
            text = _Text()
            if glyph:
                text.append(f"{glyph} ", style=style or "cyan")
            if source:
                text.append(f"{source}  ", style="dim")
            text.append(message)
            self._console.print(text)
        else:
            bits = []
            if glyph:
                bits.append(glyph)
            if source:
                bits.append(f"[{source}]")
            bits.append(message)
            print(" ".join(bits), file=self.stream, flush=True)

    def note(self, message: str, *, source: str | None = None) -> None:
        self._note(message, source=source)

    def ok(self, message: str, *, source: str | None = None) -> None:
        self._note(message, glyph="✓", style="green", source=source)

    def warn(self, message: str, *, source: str | None = None) -> None:
        self._note(message, glyph="⚠", style="yellow", source=source)

    # -- steps -------------------------------------------------------------- #

    def start_step(self, label: str) -> _Step:
        step = _Step(self, label)
        step._prev = self._current
        self._current = step
        step._start()
        return step

    @contextlib.contextmanager
    def step(self, label: str):
        step = self.start_step(label)
        try:
            yield step
        except BaseException:
            step.done(ok=False)
            raise
        else:
            step.done(ok=True)

    def progress(self, text: str) -> None:
        if self._current is not None:
            self._current.progress(text)

    # -- panels ------------------------------------------------------------- #

    def panel(self, title: str, rows, *, style: str = "cyan") -> None:
        """Render a titled key/value panel (rows: (label, value) or plain str)."""
        if self.quiet:
            return
        if self._rich:
            grid = _Table.grid(padding=(0, 3))
            grid.add_column(justify="left", style="dim", no_wrap=True)
            grid.add_column(justify="left")
            for row in rows:
                if isinstance(row, (list, tuple)) and len(row) == 2:
                    grid.add_row(str(row[0]), _Text(str(row[1]), style="bold"))
                else:
                    grid.add_row("", _Text(str(row), style="bold"))
            self._console.print(
                _Panel(grid, title=title, title_align="left", border_style=style, box=_ROUNDED)
            )
        else:
            print(f"\n{title}", file=self.stream, flush=True)
            for row in rows:
                if isinstance(row, (list, tuple)) and len(row) == 2:
                    print(f"  {str(row[0]):<24} {row[1]}", file=self.stream, flush=True)
                else:
                    print(f"  {row}", file=self.stream, flush=True)

    def deployed(self, rows, *, title: str = "Deployed") -> None:
        self.panel(title, rows, style="green")

    # -- failures ----------------------------------------------------------- #

    def fail(self, headline: str, *, tail=None, hint: str | None = None, log_path=None) -> None:
        if self.quiet:
            return
        if hint is None and tail:
            hint = explain_failure("\n".join(tail))
        if self._err_console is not None:
            self._err_console.print(_Text.assemble(("✗ ", "bold red"), (headline, "bold")))
            if tail:
                self._err_console.print(
                    _Panel(
                        _Text("\n".join(list(tail)[-15:]), style="dim"),
                        title="last log lines",
                        title_align="left",
                        border_style="red",
                        box=_ROUNDED,
                    )
                )
            if hint:
                self._err_console.print(
                    _Panel(
                        _Text(hint),
                        title="what this means",
                        title_align="left",
                        border_style="yellow",
                        box=_ROUNDED,
                    )
                )
            if log_path:
                self._err_console.print(_Text(f"full log: {log_path}", style="dim"))
        else:
            print(f"✗ {headline}", file=sys.stderr, flush=True)
            if tail:
                print("last log lines:", file=sys.stderr)
                print("\n".join(list(tail)[-15:]), file=sys.stderr)
            if hint:
                print(f"\nwhat this means:\n{hint}", file=sys.stderr)
            if log_path:
                print(f"full log: {log_path}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Module-level default reporter + convenience wrappers
# --------------------------------------------------------------------------- #

_default: Reporter | None = None


def get_reporter() -> Reporter:
    global _default
    if _default is None:
        _default = Reporter(quiet=_env_quiet())
    return _default


def configure(*, quiet: bool | None = None) -> Reporter:
    reporter = get_reporter()
    if quiet is not None:
        reporter.quiet = quiet
        reporter._animate = bool(
            reporter._console and reporter._console.is_terminal and not quiet
        )
    return reporter


def section(title, subtitle=None):
    get_reporter().section(title, subtitle)


def note(message, *, source=None):
    get_reporter().note(message, source=source)


def ok(message, *, source=None):
    get_reporter().ok(message, source=source)


def warn(message, *, source=None):
    get_reporter().warn(message, source=source)


def step(label):
    return get_reporter().step(label)


def start_step(label):
    return get_reporter().start_step(label)


def progress(text):
    get_reporter().progress(text)


def panel(title, rows, *, style="cyan"):
    get_reporter().panel(title, rows, style=style)


def deployed(rows, *, title="Deployed"):
    get_reporter().deployed(rows, title=title)


def fail(headline, *, tail=None, hint=None, log_path=None):
    get_reporter().fail(headline, tail=tail, hint=hint, log_path=log_path)
