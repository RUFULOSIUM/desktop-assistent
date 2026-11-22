# Product specification

This is the agreed target for the Desktop Assistant, restated from the answers
in [`QUESTIONS.md`](../QUESTIONS.md). It records *what* was decided and *why*;
implementation detail belongs in the code, not here.

**Status: Core implementation complete.** Phases 0-11 and P1 polish items are
implemented and tested (477 unit tests passing). See [implementation plan](../implementation-plan.md)
for current status.

## 1. Product and scope

**General-purpose, plugin-modular core.** The assistant's job is "everything,
being modular for plugins." There is no narrow single purpose: the core provides
the runtime and plugins provide capabilities.

**Day-one example.** Open Chrome and choose a roughly ten-minute video from the
user's "For You" page. This is the kind of multi-step, app-controlling task the
agent path must support, not a hard-coded feature.

**No action without permission.** The assistant must not do anything without
permission. The user chooses between security levels, which set what the
assistant is allowed to attempt.

## 2. The island and its UI

### Placement

- On notch Macs: around the notch, like a Dynamic Island pill, or using the
  native Apple Dynamic Island.
- On non-notch Macs: a pill beside the camera, or centred on screen when there
  is no camera.

### States and motion

- **Idle:** invisible on notch Macs, as if living behind the notch. On non-notch
  Macs it rests as a pill beside the camera, or centred on screen if there is no
  camera.
- **Activated / listening:** moves out from behind the notch (or up from beside
  the camera) in a metaball-like way and pulsates with the spoken audio.
- **Thinking:** becomes a loading circle.
- **Acting:** the blob transforms into a second cursor while it does something.
- **Speaking:** pulsates in time with the spoken reply.

### Activation and input

- The wake word is primary. The user can change the word, or configure
  activation by double-clicking instead.
- Hotkey: fast triple-Shift.
- Files can be dropped onto the blob.
- Clipboard reads are allowed only at permission levels that permit it.
- **Sound effects** (✅ implemented): 8 events — wake, listening start/end, thinking, speaking, task complete, error, permission needed. Uses macOS NSSound with system defaults or custom files. Configurable per-state.
- **Notifications** (✅ implemented): macOS UserNotifications for permission needed, task complete, errors. Respects `ui.verbosity` filter (silent/minimal/normal/verbose).

## 3. Voice, models and routing

**Speech-to-text.** Multilingual, with standard English as the default. Prefer
a free cloud STT service for the variety of voices; fall back to on-device STT
if that is not possible.

**Router classifier.** A small Keras model with a maximum input of 5000
characters, passed by Unicode. It outputs one boolean: agent or no-agent.
Dataset: 10,012 labeled examples (`data/router/train.jsonl`). Trained model:
`src/assistant/router/models/router_classifier.keras` (~51MB).

**Chat path.** Text-only answers use a provider the user can connect themselves.
On first startup the app scans the system for APIs already installed so they can
be reused.

**Agent path.** Non-trivial requests are dispatched to
[pi-mono](https://github.com/badlogic/pi-mono). Execution respects the
permission/security level chosen by the user (see §1 and §4).

## 4. Capabilities and permissions

**Broad capability, gated per task.** With the permissions required for a task
granted, the agent may do anything (shell commands, files, app control, web,
and so on). Permissions are the gate, not a fixed allow-list of actions.

**Private data boundary.** Data the user flags as private must never be
accessed or sent off-device.

## 5. Technical preferences

| Topic | Decision |
| --- | --- |
| Language split | Performance-critical code in C, importable from Python; the custom blob UI in C if Python cannot do it |
| Release | Primarily a `.bin` release on GitHub; later Homebrew and `.dmg` releases on GitHub |
| Footprint | Stay lightweight, but use the space/dependencies when a feature needs them |
| Python version | Newest possible; downgrade to 3.13 if something breaks on, say, 3.14 |

### Module homes in the scaffold

These paths exist as empty stubs and are the intended homes for the decisions
above: `src/assistant/router/classifier.py` (Keras router),
`src/assistant/stt/` (cloud and on-device STT),
`src/assistant/llm/` (local and remote chat providers),
`src/assistant/agent/` (pi-mono dispatch and presets),
`src/assistant/ui/` (island window, view, animation, menu bar),
`src/assistant/audio/` and `native/ringbuffer/` (C ring buffer and capture),
and `src/assistant/wake/` (wake-word engine).

## 6. Project setup

- **Git and uv:** full git and uv setup.
- **Virtualenv:** start a virtualenv now.
- **Testing:** pytest with a 100% coverage target.
- **Name and license:** the project is "Desktop Assistant" (the folder is
  currently `Desktop assistend`). The license should be usable but not
  copyable; the exact text is still to be chosen.

## 7. Quality bar and risk

- **Avoid "AI slop."** Use full capability and care to get the behaviour and
  look right rather than producing a generic result.
- **Biggest risk to avoid:** not finishing the project. Favour completion.
