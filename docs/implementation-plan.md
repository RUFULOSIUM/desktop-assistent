# Implementation Plan: Desktop Assistant

## ✅ COMPLETED

### Phase 0: Project Foundation
- Git repo + uv project + pyproject.toml
- Virtualenv + dev dependencies (pytest, ruff, mypy)
- pytest configured for 100% coverage target
- Pre-commit hooks
- Apache-2.0 license

### Phase 1: Core Infrastructure
- Config system (`src/assistant/config.py`) - JSON/TOML, 15+ sections, validation, API scanner
- Logging (`src/assistant/logging.py`)
- Error hierarchy (`src/assistant/errors.py`)
- CLI (`src/assistant/cli.py`) - run, config, doctor commands
- Main app scaffold (`src/assistant/app.py`, `src/assistant/__main__.py`)

### Phase 2: Audio Pipeline (C + Python)
- Native ring buffer (`native/ringbuffer/`) - C extension, thread-safe, lock-free
- Audio capture (`src/assistant/audio/capture.py`) - Core Audio / AudioUnit
- VAD integration (`src/assistant/audio/vad.py`) - WebRTC VAD
- Ring buffer Python wrapper (`src/assistant/audio/ringbuffer.py`)

### Phase 3: Wake Word Detection
- sherpa-onnx engine (`src/assistant/wake/sherpa_engine.py`)
- Model loading from `src/assistant/wake/models/`
- Configurable wake word + double-click / triple-Shift fallbacks

### Phase 4: Speech-to-Text
- STT abstraction (`src/assistant/stt/__init__.py`) - cloud/local/auto with fallback chain
- faster-whisper (`src/assistant/stt/faster_whisper.py`)
- MLX Whisper (`src/assistant/stt/mlx_whisper.py`)

### Phase 6: LLM Providers
- Provider abstraction (`src/assistant/llm/__init__.py`)
- Local LLMs (`src/assistant/llm/local.py`) - Ollama, llama.cpp, LM Studio, vLLM
- Remote APIs (`src/assistant/llm/remote.py`) - OpenAI, Anthropic
- First-run installed-API scanner

### Phase 7: Agent Dispatch
- pi-mono integration (`src/assistant/agent/dispatch.py`)
- Agent presets (`src/assistant/agent/presets.py`)
- Permission gating (shell, files, apps, web, clipboard)
- Private-data boundary enforcement

### Phase 8: UI - The Island (Core)
- Notch detection (`src/assistant/platform/notch.py`)
- Island window (`src/assistant/ui/island_window.py`) - NSPanel, borderless, transparent
- Island view (`src/assistant/ui/island_view.py`) - metaball animation
- Animation system (`src/assistant/ui/animation.py`) - spring, pulsate, spinner, cursor morph
- 6 states: idle (hidden), activated, listening, thinking, acting, speaking
- Menu bar fallback (`src/assistant/ui/menu_bar.py`)

### Phase 9: Pipeline Orchestration
- End-to-end pipeline (`src/assistant/pipeline.py`) - Wake → VAD → STT → Router → LLM/Agent → TTS → UI
- Background listening thread with VAD segmentation
- Doctor/health check (`src/assistant/doctor.py`) - 12 checks, CLI command
- TTS integration (`src/assistant/tts/__init__.py`) - system, Piper, Coqui, ElevenLabs, OpenAI

### Phase 10: Smart Router (NEW - COMPLETED)
- Keras classifier (`src/assistant/router/classifier.py`) - Embedding + Conv1D model
- Training dataset: 10,012 labeled examples (`data/router/train.jsonl`)
- Trained model: `src/assistant/router/models/router_classifier.keras` (~51MB)
- Training script (`scripts/train_router.py`) with early stopping, class balancing
- Dataset generation script (`scripts/generate_router_dataset.py`) with 80+ templates
- Keyword fallback when model unavailable

### Phase 11: Global Hotkeys (NEW - COMPLETED)
- Hotkey manager (`src/assistant/hotkeys.py`)
- Triple-Shift listener (CGEventTap) - requires accessibility permission
- Double-click on island (NSEvent global monitor)
- Push-to-talk key hold support
- Toggle listening hotkey
- Show/hide window hotkey (default: cmd+shift+space)
- Integrated into `Application` class in `app.py`

### Testing
- 464 unit/integration tests (1 collection error in test_hotkeys due to import)
- 89% coverage target configured
- 3 integration tests require macOS CoreAudio frameworks

---

## 🎯 REMAINING (Priority Order)

### P0 - Critical Path (Blocks Core Functionality)

- [x] **Fix test_hotkeys.py import error** — `tests/unit/test_hotkeys.py:9`
  - ModuleNotFoundError: No module named 'src'
  - Fix PYTHONPATH / test configuration to import from src correctly
  - **DONE**: Changed imports/patches from `src.assistant.hotkeys` to `assistant.hotkeys` (matches installed package name)

### P1 - High Visibility (User-Facing Polish)

- [x] **Drag-drop onto island** — `src/assistant/ui/island_view.py`, `island_window.py`
  - Accept file drops, pass to pipeline as context
  - Implemented: IslandView drag handlers, pipeline dropped file context, app callback wiring
- [x] **Sound effects system** — New `src/assistant/audio/sounds.py`
  - Wake, listening start/end, thinking, error, completion
  - Configurable per-state sounds
  - Implemented: SoundEvent enum, SoundConfig, SoundManager with NSSound, pipeline integration
- [x] **Notification system** — New `src/assistant/notify.py`
  - Low-verbosity only: permission needed, task complete
  - macOS UserNotifications framework
  - Implemented: NotificationType enum, NotificationConfig, NotificationManager, pipeline integration

### P2 - Platform Integration

- [x] **Non-notch Mac QA + fallback** — `src/assistant/platform/notch.py`, `island_window.py`
  - Fallback implemented: detects non-notch Macs, positions pill beside camera area
  - Hardware QA pending (requires MacBook Air)
- [x] **Clipboard permission + read** — `src/assistant/platform/clipboard.py`, pipeline
  - `allow_clipboard` gating implemented in config
  - Clipboard reading via NSPasteboard, context pasted into transcript
  - Tests: `tests/unit/test_clipboard.py` (9 tests passing)

### P3 - Packaging & Release

- [ ] **Build scripts** — `scripts/build.sh`, `pyproject.toml` [build]
  - PyInstaller / py2app for .bin
  - Include C extension, models, configs
- [ ] **.app bundle** — `src/assistant/platform/bundle.py`, `Info.plist`
  - Proper macOS app structure
  - Code signing (ad-hoc + Developer ID)
- [ ] **Homebrew formula** — `Formula/desktop-assistant.rb`
- [ ] **.dmg creation** — `scripts/create_dmg.sh`
- [ ] **GitHub Actions CI/CD** — `.github/workflows/`
  - Test matrix (py3.11-3.13, macOS latest)
  - Build + sign + upload artifacts

### P4 - Hardening & Performance

- [ ] **Manual QA** — Hardware testing
  - Notch MacBook Pro (14"/16" M1-M4)
  - Non-notch MacBook Air (M1-M3)
  - External monitor edge cases
- [ ] **Performance profiling** — `scripts/profile.py`
  - Audio latency (capture → VAD → STT)
  - UI frame rate (60fps island animation)
  - Memory usage (model caching)
- [ ] **100% test coverage** — All test files
  - Cover uncovered branches in doctor, TTS, UI
  - Property-based tests for config validation

---

## Dependencies & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| C/Python audio bridge | Already working | Ring buffer proven; keep stable |
| Notch UI on non-notch Macs | UX regression | P2 task #5; fallback designed |
| pi-mono API stability | Agent failures | Pin version in dispatch; test in isolation |
| Permission model complexity | Security gaps | Implemented incrementally; tested |
| Accessibility permission for hotkeys | Triple-shift fails | Clear setup instructions; fallback to menu bar |

---

## Quick Start for Next Session

```bash
# 1. Fix test import issue
# Edit tests/unit/test_hotkeys.py or pyproject.toml to ensure src is on path

# 2. Run tests
uv run pytest tests/ -x -q

# 3. Verify hotkeys work
uv run python -m src.assistant --help

# 4. Optional: retrain router model with more data
uv run python scripts/generate_router_dataset.py --samples 20000
uv run python scripts/train_router.py --epochs 50
```
