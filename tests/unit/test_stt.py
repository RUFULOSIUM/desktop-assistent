"""Tests for STT module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from assistant.config import STTConfig
from assistant.errors import STTError


def test_stt_provider_base() -> None:
    from assistant.stt import STTProvider

    config = STTConfig()

    class TestProvider(STTProvider):
        def transcribe(self, _audio, _sample_rate):
            return "test"

        def transcribe_stream(self, _audio_chunks, _sample_rate):
            yield "test"

        def start(self):
            pass

        def stop(self):
            pass

        @property
        def name(self):
            return "test"

        @property
        def is_local(self):
            return True

    provider = TestProvider(config)
    assert provider.config == config
    assert provider.name == "test"
    assert provider.is_local is True


def test_create_stt_provider_cloud() -> None:
    from assistant.stt import create_stt_provider

    config = STTConfig(provider="cloud")
    provider = create_stt_provider(config)
    assert provider.name.startswith("cloud:")


def test_create_stt_provider_local() -> None:
    from assistant.stt import create_stt_provider

    config = STTConfig(provider="local")
    provider = create_stt_provider(config)
    assert provider.name.startswith("local:")


def test_create_stt_provider_auto() -> None:
    from assistant.stt import create_stt_provider

    config = STTConfig(provider="auto")
    provider = create_stt_provider(config)
    assert provider.name == "auto"


def test_create_stt_provider_invalid() -> None:
    from assistant.stt import create_stt_provider

    config = STTConfig(provider="invalid")
    with pytest.raises(STTError, match="Unknown STT provider"):
        create_stt_provider(config)


def test_cloud_stt_provider_start() -> None:
    from assistant.stt import CloudSTTProvider

    config = STTConfig(provider="cloud", cloud_service="whisper-api")

    with patch.dict("sys.modules", {"openai": None}):
        provider = CloudSTTProvider(config)
        with pytest.raises(STTError, match="openai package not installed"):
            provider.start()


def test_cloud_stt_provider_unsupported_service() -> None:
    from assistant.stt import CloudSTTProvider

    config = STTConfig(provider="cloud", cloud_service="unsupported")
    provider = CloudSTTProvider(config)
    with pytest.raises(STTError, match="Unsupported cloud STT service"):
        provider.start()


def test_cloud_stt_provider_transcribe() -> None:
    from assistant.stt import CloudSTTProvider

    config = STTConfig(provider="cloud", cloud_service="whisper-api")

    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello world"
    mock_client.audio.transcriptions.create.return_value = mock_response
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict("sys.modules", {"openai": mock_openai}):
        provider = CloudSTTProvider(config)
        provider.start()

        audio = np.zeros(16000, dtype=np.float32)
        result = provider.transcribe(audio, 16000)

        assert result == "Hello world"
        mock_client.audio.transcriptions.create.assert_called_once()


def test_local_stt_provider_start_faster_whisper() -> None:
    from assistant.stt import LocalSTTProvider

    config = STTConfig(provider="local", local_model="base")

    mock_faster_whisper = MagicMock()
    mock_faster_whisper.FasterWhisperSTT = MagicMock()

    with patch.dict("sys.modules", {"assistant.stt.faster_whisper": mock_faster_whisper}):
        provider = LocalSTTProvider(config)
        provider.start()

        mock_faster_whisper.FasterWhisperSTT.assert_called_once_with(config)
        mock_faster_whisper.FasterWhisperSTT.return_value.start.assert_called_once()


def test_local_stt_provider_fallback_to_mlx() -> None:
    from assistant.stt import LocalSTTProvider

    config = STTConfig(provider="local", local_model="base")

    mock_faster_whisper = MagicMock()
    mock_faster_whisper.FasterWhisperSTT.side_effect = ImportError("not available")

    mock_mlx_whisper = MagicMock()
    mock_mlx_whisper.MLXWhisperSTT = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "assistant.stt.faster_whisper": mock_faster_whisper,
            "assistant.stt.mlx_whisper": mock_mlx_whisper,
        },
    ):
        provider = LocalSTTProvider(config)
        provider.start()

        mock_mlx_whisper.MLXWhisperSTT.assert_called_once_with(config)
        mock_mlx_whisper.MLXWhisperSTT.return_value.start.assert_called_once()


def test_local_stt_provider_both_fail() -> None:
    from assistant.stt import LocalSTTProvider

    config = STTConfig(provider="local", local_model="base")

    mock_faster_whisper = MagicMock()
    mock_faster_whisper.FasterWhisperSTT.side_effect = ImportError("not available")

    mock_mlx_whisper = MagicMock()
    mock_mlx_whisper.MLXWhisperSTT.side_effect = ImportError("not available")

    with patch.dict(
        "sys.modules",
        {
            "assistant.stt.faster_whisper": mock_faster_whisper,
            "assistant.stt.mlx_whisper": mock_mlx_whisper,
        },
    ):
        provider = LocalSTTProvider(config)
        with pytest.raises(STTError, match="No local STT engine available"):
            provider.start()


def test_auto_stt_provider_start() -> None:
    from assistant.stt import AutoSTTProvider

    config = STTConfig(provider="auto", fallback_chain=["cloud", "local"])

    mock_cloud_provider = MagicMock()
    mock_cloud_provider.start.return_value = None

    mock_local_provider = MagicMock()
    mock_local_provider.start.return_value = None

    mock_stt = MagicMock()
    mock_stt.CloudSTTProvider.return_value = mock_cloud_provider
    mock_stt.LocalSTTProvider.return_value = mock_local_provider

    mock_faster_whisper = MagicMock()
    mock_faster_whisper.FasterWhisperSTT.return_value = mock_local_provider

    mock_mlx_whisper = MagicMock()
    mock_mlx_whisper.MLXWhisperSTT.return_value = mock_local_provider

    with patch.dict(
        "sys.modules",
        {
            "assistant.stt": mock_stt,
            "assistant.stt.faster_whisper": mock_faster_whisper,
            "assistant.stt.mlx_whisper": mock_mlx_whisper,
        },
    ):
        provider = AutoSTTProvider(config)
        provider.start()

        assert provider._current_provider is not None


def test_auto_stt_provider_all_fail() -> None:
    from assistant.stt import AutoSTTProvider

    config = STTConfig(provider="auto", fallback_chain=["cloud", "local"])

    mock_stt = MagicMock()
    mock_stt.CloudSTTProvider.side_effect = RuntimeError("cloud failed")
    mock_stt.LocalSTTProvider.side_effect = RuntimeError("local failed")

    with patch.dict("sys.modules", {"assistant.stt": mock_stt}):
        provider = AutoSTTProvider(config)
        with pytest.raises(STTError, match="All STT providers failed"):
            provider.start()


def test_faster_whisper_stt_init() -> None:
    from assistant.stt.faster_whisper import FasterWhisperSTT

    config = STTConfig(local_model="base")
    stt = FasterWhisperSTT(config)
    assert stt._model_size == "base"
    assert stt._model is None


def test_faster_whisper_stt_model_sizes() -> None:
    from assistant.stt.faster_whisper import FasterWhisperSTT

    config = STTConfig(local_model="large-v3")
    stt = FasterWhisperSTT(config)
    assert stt._model_size == "large-v3"


def test_faster_whisper_stt_start_import_error() -> None:
    from assistant.stt.faster_whisper import FasterWhisperSTT

    config = STTConfig(local_model="base")
    stt = FasterWhisperSTT(config)

    with patch.dict("sys.modules", {"faster_whisper": None}), pytest.raises(
        STTError, match="faster-whisper not installed"
    ):
        stt.start()


def test_faster_whisper_stt_transcribe() -> None:
    from assistant.stt.faster_whisper import FasterWhisperSTT

    config = STTConfig(local_model="base", language="en")
    stt = FasterWhisperSTT(config)

    mock_model = MagicMock()
    mock_segment1 = MagicMock()
    mock_segment1.text = "Hello"
    mock_segment2 = MagicMock()
    mock_segment2.text = "world"
    mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], MagicMock())
    stt._model = mock_model

    audio = np.zeros(16000, dtype=np.float32)
    result = stt.transcribe(audio, 16000)

    assert result == "Hello world"


def test_faster_whisper_stt_transcribe_resample() -> None:
    from assistant.stt.faster_whisper import FasterWhisperSTT

    config = STTConfig(local_model="base")
    stt = FasterWhisperSTT(config)

    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Hello"
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())
    stt._model = mock_model

    # 48kHz audio
    audio = np.zeros(48000, dtype=np.float32)
    result = stt.transcribe(audio, 48000)

    assert result == "Hello"
    # Check that resampling happened (audio passed to model should be 16kHz)
    call_args = mock_model.transcribe.call_args[0][0]
    assert len(call_args) == 16000


def test_faster_whisper_stt_stream() -> None:
    from assistant.stt.faster_whisper import FasterWhisperSTT

    config = STTConfig(local_model="base")
    stt = FasterWhisperSTT(config)

    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Hello"
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())
    stt._model = mock_model

    def audio_chunks():
        yield np.zeros(8000, dtype=np.float32)  # 0.5s
        yield np.zeros(8000, dtype=np.float32)  # 0.5s
        yield np.zeros(8000, dtype=np.float32)  # 0.5s
        yield np.zeros(8000, dtype=np.float32)  # 0.5s = 2s total

    results = list(stt.transcribe_stream(audio_chunks(), 16000))
    assert len(results) == 1
    assert results[0] == "Hello"


def test_mlx_whisper_stt_init() -> None:
    from assistant.stt.mlx_whisper import MLXWhisperSTT

    config = STTConfig(local_model="base")
    stt = MLXWhisperSTT(config)
    assert stt._model_size == "base"
    assert stt._mlx_whisper is None


def test_mlx_whisper_stt_start_import_error() -> None:
    from assistant.stt.mlx_whisper import MLXWhisperSTT

    config = STTConfig(local_model="base")
    stt = MLXWhisperSTT(config)

    with patch.dict("sys.modules", {"mlx_whisper": None}), pytest.raises(
        STTError, match="mlx-whisper not installed"
    ):
        stt.start()


def test_mlx_whisper_stt_transcribe() -> None:
    from assistant.stt.mlx_whisper import MLXWhisperSTT

    config = STTConfig(local_model="base", language="en")
    stt = MLXWhisperSTT(config)

    mock_mlx = MagicMock()
    mock_mlx.transcribe.return_value = {"text": "Hello world"}
    stt._mlx_whisper = mock_mlx
    stt._model = True

    audio = np.zeros(16000, dtype=np.float32)
    result = stt.transcribe(audio, 16000)

    assert result == "Hello world"


def test_mlx_whisper_stt_transcribe_resample() -> None:
    from assistant.stt.mlx_whisper import MLXWhisperSTT

    config = STTConfig(local_model="base")
    stt = MLXWhisperSTT(config)

    mock_mlx = MagicMock()
    mock_mlx.transcribe.return_value = {"text": "Hello"}
    stt._mlx_whisper = mock_mlx
    stt._model = True

    # 48kHz audio
    audio = np.zeros(48000, dtype=np.float32)
    result = stt.transcribe(audio, 48000)

    assert result == "Hello"
    # Check that resampling happened
    call_args = mock_mlx.transcribe.call_args[0][0]
    assert len(call_args) == 16000


def test_mlx_whisper_stt_stream() -> None:
    from assistant.stt.mlx_whisper import MLXWhisperSTT

    config = STTConfig(local_model="base")
    stt = MLXWhisperSTT(config)

    mock_mlx = MagicMock()
    mock_mlx.transcribe.return_value = {"text": "Hello"}
    stt._mlx_whisper = mock_mlx
    stt._model = True

    def audio_chunks():
        yield np.zeros(8000, dtype=np.float32)  # 0.5s
        yield np.zeros(8000, dtype=np.float32)  # 0.5s
        yield np.zeros(8000, dtype=np.float32)  # 0.5s
        yield np.zeros(8000, dtype=np.float32)  # 0.5s = 2s total

    results = list(stt.transcribe_stream(audio_chunks(), 16000))
    assert len(results) == 1
    assert results[0] == "Hello"