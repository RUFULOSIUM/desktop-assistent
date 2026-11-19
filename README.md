# Desktop Assistant

A modular, voice-first macOS assistant that lives at the camera notch and hands
requests off to an agentic harness. It never acts without the user's permission.

> **Status: early scaffold.** The repository currently holds the directory
> layout, empty Python/C source and test stubs, and the native ring-buffer C
> skeleton. No runtime behaviour is implemented yet. The decisions recorded
> below are the agreed target, taken from the answered `QUESTIONS.md`.

## What it is

- **General-purpose and plugin-driven.** The core job is "everything, modular
  for plugins" — capabilities are added as plugins, not baked into the core.
- **Voice-first, with a split brain.** A wake word is the primary activation. A
  small on-device classifier decides whether a request gets a direct text answer
  or is dispatched to an agent.
- **Permission-gated.** The assistant does nothing without permission. The user
  selects a security level; each task asks only for the permissions it needs.
- **Notch-native UI.** On notch Macs it hides behind the notch and expands
  metaball-style into a "blob"; elsewhere it sits as a pill beside the camera.

For the full set of decisions and the reasoning behind them, see the
[product specification](docs/specification.md).

## Key decisions

| Area | Decision |
| --- | --- |
| Scope | General-purpose, plugin-based assistant |
| Activation | Configurable wake word (primary); double-click or fast triple-Shift as alternatives |
| UI | Metaball blob emerging from the notch; idle hidden behind it; pill beside the camera on non-notch Macs |
| Speech-to-text | Multilingual, English default; free cloud STT preferred, on-device fallback |
| Routing | Small Keras binary classifier, ≤5000 Unicode characters in → agent / no-agent out |
| Chat answers | User-supplied provider; first startup scans for installed APIs to reuse |
| Agent | Dispatches to [pi-mono](https://github.com/badlogic/pi-mono) |
| Permissions | Broad capability, gated per task; user-flagged private data is off-limits |
| Languages | Performance-critical code and the custom blob UI in C; the rest in Python |
| Release | GitHub `.bin` release first; Homebrew and `.dmg` later |

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/assistant/` | Python package: app, CLI, config, pipeline, audio, wake word, STT, router, LLM, agent dispatch, UI, platform |
| `native/ringbuffer/` | C audio ring buffer, imported into Python |
| `tests/` | pytest unit and integration tests |
| `resources/` | Icons, sounds, launchd agents |
| `scripts/` | Build and helper scripts |
| `docs/` | Project documentation |
| `QUESTIONS.md` | The answered questions that define the product |

## Toolchain and conventions

These are the agreed choices; the `uv` project, virtualenv and test suite are not
set up yet in this scaffold.

- **Environment:** `uv`-managed project with a virtualenv, committed to git.
- **Python:** newest available, with a fallback to 3.13 if a dependency breaks
  on a newer release (e.g. 3.14).
- **Tests:** pytest, aiming for 100% coverage.
- **Footprint:** stay lightweight, but accept larger dependencies where a
  feature genuinely needs them.

## Documentation

- [Product specification](docs/specification.md) — decisions and rationale
- [Wake-word models](src/assistant/wake/models/README.md) — bundling and training wake models
- [Project questions](QUESTIONS.md) — the original Q&A

## License

Intended to be source-available: usable, but not copyable. The exact license
text is still to be chosen (see `QUESTIONS.md`, question 21).
