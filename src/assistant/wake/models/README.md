# Wake-word models

Drop trained openWakeWord `.onnx` models here. The assistant loads the first
`*.onnx` found when no `wake_model` is configured.

Recommended bundled default: `hey_assistant` (generic, agent-neutral).

Train a model with openWakeWord (100% synthetic speech, speaker-independent):

1. Open the official openWakeWord training notebook (free T4 GPU, ~10 min).
2. Set the target phrase, run all cells, download the `.onnx`.
3. Copy it here as `<name>.onnx`, then `assistant config set wake_model <name>`.

Models are not committed to git (see `.gitignore`); ship them via the packaging
step or the `.app` bundle.
