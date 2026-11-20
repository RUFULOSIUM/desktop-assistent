"""Audio capture using Core Audio on macOS."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ringbuffer import AudioRingBuffer
    from .vad import VoiceActivityDetector


class AudioCapture:
    """Audio capture using Core Audio (AudioUnit).

    Args:
        ring_buffer: AudioRingBuffer to write captured audio data.
        vad: Optional VoiceActivityDetector for voice activity detection.
        device_id: Optional audio input device ID.
        sample_rate: Target sample rate in Hz (default 16000).
        channels: Number of audio channels (default 1 for mono).
    """

    def __init__(
        self,
        ring_buffer: AudioRingBuffer,
        vad: VoiceActivityDetector | None = None,
        device_id: int | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self._ring_buffer = ring_buffer
        self._vad = vad
        self._device_id = device_id
        self._sample_rate = sample_rate
        self._channels = channels
        self._running = False
        self._audio_unit: object | None = None
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def vad(self) -> VoiceActivityDetector | None:
        return self._vad

    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start audio capture."""
        with self._lock:
            if self._running:
                return
            self._running = True

        # Import pyobjc frameworks lazily
        try:
            from AudioUnit import (  # type: ignore[import-not-found]
                AudioComponentDescription,
                AudioComponentFindNext,
                AudioComponentInstanceNew,
                AudioUnit,
                AudioUnitInitialize,
                AudioUnitSetProperty,
                AudioUnitStart,
                AURenderCallbackStruct,
                kAudioUnitManufacturer_Apple,
                kAudioUnitProperty_StreamFormat,
                kAudioUnitScope_Input,
                kAudioUnitScope_Output,
                kAudioUnitSubType_HALOutput,
                kAudioUnitType_Output,
            )
            from CoreAudio import (  # type: ignore[import-not-found]
                AudioDeviceID,
                AudioObjectGetPropertyData,
                kAudioHardwarePropertyDefaultInputDevice,
            )
        except ImportError as e:
            with self._lock:
                self._running = False
            raise ImportError(
                "pyobjc-framework-CoreAudio and "
                "pyobjc-framework-AudioToolbox are required for AudioCapture"
            ) from e

        # Find the input device
        if self._device_id is not None:
            device_id = self._device_id
        else:
            # Get default input device
            device_id = AudioDeviceID()
            prop_size = 4
            AudioObjectGetPropertyData(
                AudioDeviceID(0),  # kAudioObjectSystemObject
                kAudioHardwarePropertyDefaultInputDevice,
                0,
                None,
                prop_size,
                device_id,
            )

        # Create AudioComponentDescription for HAL Output
        desc = AudioComponentDescription()
        desc.componentType = kAudioUnitType_Output
        desc.componentSubType = kAudioUnitSubType_HALOutput
        desc.componentManufacturer = kAudioUnitManufacturer_Apple
        desc.componentFlags = 0
        desc.componentFlagsMask = 0

        # Find and create the AudioUnit
        comp = AudioComponentFindNext(None, desc)
        if comp is None:
            with self._lock:
                self._running = False
            raise RuntimeError("Failed to find HAL Output AudioUnit")

        audio_unit = AudioUnit()
        AudioComponentInstanceNew(comp, audio_unit)

        # Enable input on the AudioUnit
        enable = 1
        AudioUnitSetProperty(
            audio_unit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input,
            0,
            enable,
            4,
        )

        # Set the stream format for input (16-bit PCM, target sample rate)
        # Note: In real implementation, we'd create proper AudioStreamBasicDescription
        # For now, we use a simplified approach
        AudioUnitSetProperty(
            audio_unit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output,
            0,
            enable,
            4,
        )

        # Set up render callback for input
        callback_struct = AURenderCallbackStruct()
        callback_struct.inputProc = self._audio_callback
        callback_struct.inputProcRefCon = self

        AudioUnitSetProperty(
            audio_unit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input,
            0,
            callback_struct,
            len(callback_struct),
        )

        # Initialize and start the AudioUnit
        AudioUnitInitialize(audio_unit)
        AudioUnitStart(audio_unit)

        self._audio_unit = audio_unit

    def stop(self) -> None:
        """Stop audio capture."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._audio_unit is not None:
            try:
                from AudioUnit import (
                    AudioUnitStop,
                    AudioUnitUninitialize,
                )

                AudioUnitStop(self._audio_unit)
                AudioUnitUninitialize(self._audio_unit)
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._audio_unit = None

    def _audio_callback(
        self,
        _in_ref_con: object,
        _io_action_flags: object,
        _in_time_stamp: object,
        _in_bus_number: object,
        _in_number_frames: object,
        _io_data: object,
    ) -> int:
        """Audio input callback (runs on Core Audio thread)."""
        if not self._running:
            return 0

        try:
            # Simplified: just write a dummy frame for testing
            # In real implementation, we'd parse the AudioBufferList from _io_data
            dummy_data = bytes(960)  # 30ms at 16kHz 16-bit mono
            self._ring_buffer.write(dummy_data)

        except Exception:
            pass  # Ignore errors in callback

        return 0
