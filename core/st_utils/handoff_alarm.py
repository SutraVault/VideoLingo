"""Repeating desktop alarm independent of Windows event sound settings."""
import io
import math
import struct
import sys
import threading
import wave

ALARM_VERSION = 5
# Preserve the active alarm across module reloads and browser sessions.
_active_lock = globals().get("_active_lock", threading.Lock())
_active_alarm = globals().get("_active_alarm", None)


def _alarm_wav(kind, pulses=None):
    sample_rate = 22050
    frequencies = (880, 1100, 880) if kind != "error" else (660, 440, 660)
    samples = bytearray()
    if pulses is None:
        pulses = [(frequency, 0.12) for frequency in frequencies]
    for frequency, pause in pulses:
        length = int(sample_rate * 0.25)
        for index in range(length):
            # Fade each pulse to avoid sharp clicks.
            envelope = min(1.0, index / 220, (length - 1 - index) / 220)
            value = int(12000 * envelope * math.sin(2 * math.pi * frequency * index / sample_rate))
            samples.extend(struct.pack("<h", value))
        samples.extend(b"\x00\x00" * int(sample_rate * pause))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples)
    return buffer.getvalue()


def play_completion_alarm():
    """Play two pairs of short tones once, with one second between pairs."""
    if not sys.platform.startswith("win"):
        return
    try:
        import winsound

        sound = _alarm_wav("info", pulses=[
            (880, 0.12), (880, 1.0), (880, 0.12), (880, 0.0),
        ])
        winsound.PlaySound(sound, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
    except Exception as exc:
        # A notification failure must not invalidate a completed video.
        print(f"Completion alarm playback failed: {exc}")


class HandoffAlarm:
    def __init__(self, kind="info", interval=15, on_event=None):
        self._stop = threading.Event()
        self._thread = None
        self.error_msg = ""
        self._on_event = on_event
        self._reported_backend = None
        self.version = ALARM_VERSION
        if sys.platform.startswith("win"):
            global _active_alarm
            with _active_lock:
                if _active_alarm is not None:
                    _active_alarm.stop()
                    if _active_alarm._thread is not None:
                        _active_alarm._thread.join(timeout=2)
                _active_alarm = self
                self._thread = threading.Thread(
                    target=self._ring, args=(kind, interval), daemon=True
                )
                self._thread.start()

    def _report(self, event, detail):
        if self._on_event:
            self._on_event(event, detail)

    def _ring(self, kind, interval):
        try:
            import winsound
            sound = _alarm_wav(kind)
            while not self._stop.is_set():
                try:
                    winsound.PlaySound(sound, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
                    if self._stop.is_set():
                        return
                    if self._reported_backend != "waveform":
                        self._reported_backend = "waveform"
                        self._report("alarm_playback", "waveform_v5")
                except RuntimeError as exc:
                    if self._reported_backend != "beep":
                        self._reported_backend = "beep"
                        self._report("alarm_fallback", str(exc))
                    for frequency in ((880, 1100, 880) if kind != "error" else (660, 440, 660)):
                        if self._stop.is_set():
                            return
                        winsound.Beep(frequency, 250)
                        if self._stop.wait(0.12):
                            return
                self._report("alarm_cycle_finished", f"next_alarm_after_{interval}_seconds_of_silence")
                if self._stop.wait(interval):
                    return
        except Exception as exc:
            self.error_msg = str(exc)
            self._report("alarm_error", self.error_msg)

    def stop(self):
        if not self._stop.is_set():
            self._stop.set()
            self._report("alarm_stopped", "acknowledged_or_processing_resumed")
