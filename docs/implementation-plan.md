# Rough Implementation Plan: Desktop Assistant

## Overview

Build a modular, voice-first macOS assistant that lives at the camera notch, with a Keras-based router splitting requests between direct chat answers and agentic task execution via pi-mono.

---

## Phase 0: Project Foundation

- [ ] Initialize git repo with initial commit
- [ ] Set up uv project with pyproject.toml
- [ ] Create virtualenv and install dev dependencies (pytest, ruff, mypy)
- [ ] Configure pytest for 100% coverage target
- [ ] Add pre-commit hooks
- [ ] Choose and add license (usable but not copyable)

---

## Phase 1: Core Infrastructure

- [ ] Config system (`src/assistant/config.py`)
  - TOML-based config with security levels, wake word, STT/LLM providers
  - First-run API scanner
- [ ] Logging (`src/assistant/logging.py`)
- [ ] Error hierarchy (`src/assistant/errors.py`)
- [ ] CLI entry point (`src/assistant/cli.py`)
- [ ] Main app scaffold (`src/assistant/app.py`, `src/assistant/__main__.py`)

---

## Phase 2: Audio Pipeline (C + Python)

- [ ] Native ring buffer (`native/ringbuffer/`)
  - C implementation with Python C extension
  - Thread-safe, lock-free for audio callback
- [ ] Audio capture (`src/assistant/audio/capture.py`)
  - Core Audio / AudioUnit integration
  - VAD integration (`src/assistant/audio/vad.py`)
- [ ] Ring buffer Python wrapper (`src/assistant/audio/ringbuffer.py`)

---

## Phase 3: Wake Word Detection

- [ ] openWakeWord engine integration (`src/assistant/wake/openwakeword_engine.py`)
- [ ] Model loading from `src/assistant/wake/models/`
- [ ] Configurable wake word with fallback to double-click / triple-Shift

---

## Phase 4: Speech-to-Text

- [ ] Cloud STT abstraction (`src/assistant/stt/__init__.py`)
- [ ] faster-whisper implementation (`src/assistant/stt/faster_whisper.py`)
- [ ] MLX Whisper implementation (`src/assistant/stt/mlx_whisper.py`)
- [ ] Provider selection with fallback chain

---

## Phase 5: Router / Classifier

- [ ] Dataset creation for Keras binary classifier
  - Collect/generate ~5000-char examples labeled agent vs. chat
- [ ] Train small Keras model (≤5000 Unicode chars in → bool out)
- [ ] Classifier integration (`src/assistant/router/classifier.py`)
- [ ] Prompt templates for agent vs. chat (`src/assistant/router/prompts.py`)

---

## Phase 6: LLM Providers

- [ ] Provider abstraction (`src/assistant/llm/__init__.py`)
- [ ] Local LLM support (`src/assistant/llm/local.py`)
- [ ] Remote API support (`src/assistant/llm/remote.py`)
- [ ] First-run installed-API scanner

---

## Phase 7: Agent Dispatch

- [ ] pi-mono integration (`src/assistant/agent/dispatch.py`)
- [ ] Agent presets (`src/assistant/agent/presets.py`)
- [ ] Permission gating per task
- [ ] Private-data boundary enforcement

---

## Phase 8: UI - The Island

- [ ] Notch detection (`src/assistant/platform/notch.py`)
- [ ] Island window (`src/assistant/ui/island_window.py`)
- [ ] Island view with metaball animation (`src/assistant/ui/island_view.py`)
- [ ] Animation system (`src/assistant/ui/animation.py`)
- [ ] States: idle (hidden), listening (pulsating), thinking (loader), acting (cursor), speaking (pulsating)
- [ ] Menu bar fallback (`src/assistant/ui/menu_bar.py`)
- [ ] Hotkeys: triple-Shift, double-click
- [ ] Drag-drop onto blob
- [ ] Sound effects system
- [ ] Notification system (low verbosity only)

---

## Phase 9: Pipeline Orchestration

- [ ] End-to-end pipeline (`src/assistant/pipeline.py`)
  - Wake → VAD → STT → Router → (Chat LLM \| Agent) → TTS/UI
- [ ] Doctor/health check (`src/assistant/doctor.py`)

---

## Phase 10: Packaging & Release

- [ ] Build scripts for .bin release
- [ ] .app bundle creation
- [ ] Homebrew formula
- [ ] .dmg creation
- [ ] GitHub Actions CI/CD

---

## Phase 11: Testing & Hardening

- [ ] Unit tests for all modules (100% coverage)
- [ ] Integration tests for pipeline
- [ ] Manual QA on notch and non-notch Macs
- [ ] Performance profiling (audio latency, UI frame rate)

---

## Dependencies & Risks

| Risk | Mitigation |
|------|------------|
| Keras classifier dataset | Start synthetic data generation early |
| C/Python audio bridge | Prototype ring buffer first |
| Notch UI on non-notch Macs | Design fallback early |
| pi-mono API stability | Pin version, test dispatch in isolation |
| Permission model complexity | Implement security levels incrementally |

---

## Notes

- This is a rough plan; each phase will be broken down into exact tasks when executed.
- Phases 1-3 are blocking for everything else; do them first.
- Phase 8 (UI) is high-visibility; allocate time for iteration.
- License decision (Phase 0) unblocks public release.