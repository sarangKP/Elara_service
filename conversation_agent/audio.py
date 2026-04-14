"""
Audio I/O for ELARA with barge-in support.

Architecture:
- AudioManager runs a background thread that ALWAYS reads from the mic.
- While ELARA is speaking, that thread detects energy above a threshold
  and fires a barge-in event, stopping playback immediately.
- Collected mic frames are assembled into a complete utterance and queued.

STT : faster-whisper (local, offline)
TTS : edge-tts → pygame  (online, better quality)
      falls back to pyttsx3 (offline)
"""

import asyncio
import io
import os
import queue
import tempfile
import threading
import wave

import audioop
import pyaudio


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

RATE            = 16000
CHUNK           = 1024
FORMAT          = pyaudio.paInt16
CHANNELS        = 1

BARGE_IN_ENERGY = 700   # RMS threshold to interrupt ELARA mid-speech
SPEECH_ENERGY   = 400   # RMS threshold to start collecting an utterance
SILENCE_CHUNKS  = int(RATE / CHUNK * 1.5)   # 1.5 s of silence ends utterance
MAX_CHUNKS      = int(RATE / CHUNK * 30)    # 30 s max recording


# ---------------------------------------------------------------------------
# Whisper (lazy-loaded on first use)
# ---------------------------------------------------------------------------

_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(audio_bytes: bytes) -> str:
    model = _get_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        segments, _ = model.transcribe(tmp, language="en", beam_size=5)
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# AudioManager — continuous mic monitor + TTS with barge-in
# ---------------------------------------------------------------------------

class AudioManager:
    """
    Always-on mic monitor.  Start once at session begin; keep running.

    Typical usage
    -------------
        am = AudioManager()
        am.start()

        # listen for user
        audio = am.listen()
        text  = transcribe(audio)

        # speak with barge-in; returns (interrupted, barge_in_audio_or_None)
        interrupted, bi_audio = am.speak("Hello! How are you today?")
        if interrupted and bi_audio:
            barge_in_text = transcribe(bi_audio)
    """

    def __init__(self, tts_voice: str = "en-GB-SoniaNeural"):
        self.tts_voice    = tts_voice
        self._pa          = pyaudio.PyAudio()
        self._stream      = None
        self._running     = False

        # Barge-in signal: set by mic thread, cleared before each TTS play
        self._barge_in    = threading.Event()

        # Completed utterances waiting to be consumed
        self._utt_queue   : queue.Queue[bytes] = queue.Queue()

        # Internal mic-thread state
        self._collecting  = False
        self._frames      : list[bytes] = []
        self._silent      = 0
        self._show_level  = False   # flip to True while listening

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        import pygame
        pygame.mixer.init(frequency=22050)

        self._stream = self._pa.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=CHUNK,
        )
        self._running = True
        t = threading.Thread(target=self._mic_loop, daemon=True, name="elara-mic")
        t.start()

    def stop(self) -> None:
        import pygame
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
        pygame.mixer.quit()

    # ------------------------------------------------------------------
    # Internal mic loop (runs in background thread)
    # ------------------------------------------------------------------

    def _mic_loop(self) -> None:
        chunk_count = 0
        while self._running:
            try:
                data = self._stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                continue

            rms = audioop.rms(data, 2)

            # ── Live mic-level indicator (only while listening) ─────────
            if self._show_level:
                bars = min(20, int(rms / 100))
                print(f"\r  mic [{'█' * bars:<20}] {rms:>5}  ", end="", flush=True)

            # ── Barge-in detection ──────────────────────────────────────
            if rms > BARGE_IN_ENERGY:
                self._barge_in.set()

            # ── Utterance collection ────────────────────────────────────
            if rms > SPEECH_ENERGY:
                if not self._collecting:
                    self._collecting  = True
                    self._frames      = []
                    self._silent      = 0
                    chunk_count       = 0
                self._frames.append(data)
                self._silent = 0
                chunk_count += 1

            elif self._collecting:
                self._frames.append(data)
                self._silent += 1
                chunk_count  += 1

                if self._silent >= SILENCE_CHUNKS or chunk_count >= MAX_CHUNKS:
                    self._utt_queue.put(self._pack_wav(self._frames))
                    self._collecting = False
                    self._frames     = []
                    self._silent     = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pack_wav(self, frames: list[bytes]) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self._pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    def _drain_utt_queue(self) -> None:
        """Discard any utterances queued while ELARA was speaking."""
        while not self._utt_queue.empty():
            try:
                self._utt_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Public: listen
    # ------------------------------------------------------------------

    def listen(self, timeout: float = 30.0) -> bytes:
        """Block until a complete utterance is detected. Returns WAV bytes."""
        self._show_level = True
        try:
            result = self._utt_queue.get(timeout=timeout)
        finally:
            self._show_level = False
            print()  # newline after the level bar
        return result

    # ------------------------------------------------------------------
    # Public: speak with barge-in
    # ------------------------------------------------------------------

    def speak(self, text: str) -> tuple[bool, bytes | None]:
        """
        Speak *text* aloud.  While playing, the mic thread keeps running.
        If the user starts speaking (energy > BARGE_IN_ENERGY), playback
        stops immediately (barge-in).

        Returns
        -------
        (interrupted, barge_in_audio)
            interrupted      : True if the user cut in
            barge_in_audio   : WAV bytes of what the user said, or None
        """
        self._barge_in.clear()
        self._show_level = False
        self._drain_utt_queue()

        interrupted = self._play_with_barge_in(text)

        barge_in_audio = None
        if interrupted:
            # Wait up to 4 s for the user's utterance to complete
            try:
                barge_in_audio = self._utt_queue.get(timeout=4.0)
            except queue.Empty:
                pass

        return interrupted, barge_in_audio

    def play_buf(self, buf: "io.BytesIO") -> tuple[bool, bytes | None]:
        """
        Play a pre-generated audio BytesIO with barge-in support.
        Skips TTS generation — use when audio was pre-generated in background.
        """
        import pygame
        self._barge_in.clear()
        self._show_level = False
        self._drain_utt_queue()

        buf.seek(0)
        pygame.mixer.music.load(buf, "mp3")
        pygame.mixer.music.play()

        pygame.time.wait(300)
        self._barge_in.clear()

        interrupted = False
        while pygame.mixer.music.get_busy():
            if self._barge_in.is_set():
                pygame.mixer.music.stop()
                interrupted = True
                break
            pygame.time.wait(50)

        barge_in_audio = None
        if interrupted:
            try:
                barge_in_audio = self._utt_queue.get(timeout=4.0)
            except Exception:
                pass

        return interrupted, barge_in_audio

    def _play_with_barge_in(self, text: str) -> bool:
        """
        Play TTS for *text*, checking for barge-in every 50 ms.
        Returns True if interrupted.
        """
        try:
            return self._play_edge_tts(text)
        except Exception:
            return self._play_pyttsx3(text)

    def _play_edge_tts(self, text: str) -> bool:
        import edge_tts
        import pygame

        async def _generate() -> io.BytesIO:
            communicate = edge_tts.Communicate(text, self.tts_voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf

        buf = asyncio.run(_generate())

        # Clear any barge-in that fired during TTS generation (before playback)
        self._barge_in.clear()

        pygame.mixer.music.load(buf, "mp3")
        pygame.mixer.music.play()

        # Grace period: don't allow barge-in for first 300 ms of playback
        pygame.time.wait(300)
        self._barge_in.clear()

        interrupted = False
        while pygame.mixer.music.get_busy():
            if self._barge_in.is_set():
                pygame.mixer.music.stop()
                interrupted = True
                break
            pygame.time.wait(50)

        return interrupted

    def _play_pyttsx3(self, text: str) -> bool:
        """
        pyttsx3 is synchronous — barge-in works by running it in a thread
        and killing the engine when barge-in fires.
        """
        import pyttsx3

        done_event = threading.Event()

        def _run():
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            done_event.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # Grace period before allowing barge-in
        threading.Event().wait(timeout=0.3)
        self._barge_in.clear()

        interrupted = False
        while not done_event.is_set():
            if self._barge_in.is_set():
                interrupted = True
                break
            threading.Event().wait(timeout=0.05)

        t.join(timeout=0.5)
        return interrupted


# ---------------------------------------------------------------------------
# Convenience: stream TTS sentence-by-sentence while LLM is still generating
# ---------------------------------------------------------------------------

import re

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def sentence_chunks(text_iter):
    """
    Yield complete sentences from a streaming text iterator.
    Buffers partial text and yields as soon as a sentence boundary is found.
    """
    buf = ""
    for chunk in text_iter:
        buf += chunk
        parts = _SENTENCE_END.split(buf)
        # All parts except the last are complete sentences
        for sentence in parts[:-1]:
            sentence = sentence.strip()
            if sentence:
                yield sentence
        buf = parts[-1]
    # Yield whatever remains
    if buf.strip():
        yield buf.strip()
