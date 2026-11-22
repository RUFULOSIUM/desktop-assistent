# Desktop Assistant

A modular, voice-first macOS assistant that lives at the camera notch and hands
requests off to an agentic harness. It never acts without the user's permission.

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

## Current Implementation Status

### ✅ Completed (Phases 0-11)

- **Phase 0**: Project foundation (Git, uv, pyproject.toml, pytest, pre-commit, Apache-2.0)
- **Phase 1**: Core infrastructure (config, logging, errors, CLI, app scaffold)
- **Phase 2**: Audio pipeline (C ring buffer, Core Audio capture, WebRTC VAD)
- **Phase 3**: Wake word detection (sherpa-onnx, configurable word + fallbacks)
- **Phase 4**: Speech-to-text (faster-whisper, MLX Whisper, cloud/local/auto with fallback)
- **Phase 6**: LLM providers (local: Ollama, llama.cpp, LM Studio, vLLM; remote: OpenAI, Anthropic)
- **Phase 7**: Agent dispatch (pi-mono integration, presets, permission gating, private data boundary)
- **Phase 8**: Island UI (notch detection, NSPanel window, metaball animation, 6 states)
- **Phase 9**: Pipeline orchestration (end-to-end wake→VAD→STT→Router→LLM/Agent→TTS→UI)
- **Phase 10**: Smart router (Keras classifier, 10k training samples, trained model)
- **Phase 11**: Global hotkeys (triple-Shift, double-click, push-to-talk, toggle, show/hide)

### ✅ Completed P1 (High Visibility Polish)

- **Drag-drop onto island** — File drops passed to pipeline as context
- **Sound effects system** — 8 events (wake, listening start/end, thinking, speaking, task complete, error, permission needed)
- **Notification system** — macOS UserNotifications for permission needed, task complete, errors (respects verbosity)

### 🎯 Remaining (Priority Order)

**P2 - Platform Integration**
- Non-notch Mac QA + fallback
- Clipboard permission + read

**P3 - Packaging & Release**
- Build scripts (PyInstaller/py2app)
- .app bundle with code signing
- Homebrew formula
- .dmg creation
- GitHub Actions CI/CD

**P4 - Hardening & Performance**
- Manual QA on hardware
- Performance profiling
- 100% test coverage

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/assistant/` | Python package: app, CLI, config, pipeline, audio, wake word, STT, router, LLM, agent dispatch, UI, platform |
| `native/ringbuffer/` | C audio ring buffer, imported into Python |
| `tests/` | pytest unit and integration tests (477 tests, 89% coverage) |
| `resources/` | Icons, sounds, launchd agents |
| `scripts/` | Build and helper scripts (train router, generate dataset) |
| `data/` | Router training data (10k samples) |
| `docs/` | Project documentation |
| `QUESTIONS.md` | The answered questions that define the product |

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -x -q

# Run health check
uv run python -m src.assistant doctor

# Start the assistant
uv run python -m src.assistant run
```

## Configuration

Config file: `~/Library/Application Support/Desktop Assistant/config.toml`

Key sections:
- `wake` — wake word, sensitivity, fallbacks
- `stt` — provider (cloud/local/auto), model, language
- `llm` — provider (local/openai/anthropic/auto), models, API keys
- `tts` — provider (system/piper/coqui/elevenlabs/openai)
- `audio` — sample rate, VAD, devices
- `sound` — sound effects enabled, volume, custom sound files
- `ui` — position, animation, sound effects, notifications, verbosity
- `hotkeys` — triple-shift, double-click, push-to-talk, show/hide
- `permissions` — shell, files, apps, web, clipboard, private paths
- `agent` — pi-mono binary, working dir, timeout, presets

## Documentation

- [Product specification](docs/specification.md) — decisions and rationale
- [Implementation plan](docs/implementation-plan.md) — phased task tracking
- [Wake-word models](src/assistant/wake/models/README.md) — bundling and training wake models
- [Project questions](QUESTIONS.md) — the original Q&A

## License

Apache-2.0 — see `LICENSE` file.
