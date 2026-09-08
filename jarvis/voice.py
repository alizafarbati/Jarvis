"""Text-to-speech and speech recognition capabilities."""

import threading
import queue
import speech_recognition as sr
from typing import Optional, Callable
from rich.console import Console

from jarvis.config import config

console = Console()


class TTS:
    """Text-to-Speech engine."""

    def __init__(self):
        self.engine = None
        self.speech_queue = queue.Queue()
        self.speaking = False
        self._lock = threading.Lock()
        self._init_engine()

    def _init_engine(self):
        """Initialize TTS engine."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', config.voice_rate)
            self.engine.setProperty('volume', config.voice_volume)

            # Try to use a good voice
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'english' in voice.name.lower() or 'david' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

            self._start_speech_thread()
        except Exception as e:
            console.print(f"[yellow]TTS not available: {e}[/yellow]")

    def _start_speech_thread(self):
        """Start the speech processing thread."""
        def process_queue():
            while True:
                text = self.speech_queue.get()
                if text is None:
                    break
                self.speaking = True
                try:
                    with self._lock:
                        self.engine.say(text)
                        self.engine.runAndWait()
                except Exception as e:
                    console.print(f"[red]TTS Error: {e}[/red]")
                self.speaking = False
                self.speech_queue.task_done()

        thread = threading.Thread(target=process_queue, daemon=True)
        thread.start()

    def speak(self, text: str):
        """Add text to speech queue."""
        if self.engine:
            # Clean up text for speech
            clean_text = self._clean_for_speech(text)
            self.speech_queue.put(clean_text)

    def _clean_for_speech(self, text: str) -> str:
        """Clean text for better speech."""
        # Remove markdown and special characters
        import re
        text = re.sub(r'[*_`#]', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
        text = re.sub(r'https?://\S+', 'URL', text)
        return text.strip()

    def stop(self):
        """Stop speaking."""
        if self.engine:
            with self._lock:
                self.engine.stop()
            # Clear queue
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self.speaking


class VoiceRecognizer:
    """Speech recognition for voice commands.

    Gracefully degrades when PyAudio / a microphone is unavailable
    (headless servers, CI runners): recognition is disabled and
    `listen_once` returns None instead of crashing.
    """

    def __init__(self, wake_word: Optional[str] = None):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.available = False
        self.wake_word = wake_word or config.wake_word
        self.listening = False
        self.callback: Optional[Callable[[str], None]] = None

        try:
            self.microphone = sr.Microphone()
        except Exception as e:
            console.print(f"[yellow]Microphone unavailable ({e}); voice input disabled.[/yellow]")
            return

        # Calibrate for ambient noise
        with self.microphone as source:
            console.print("[blue]Calibrating for ambient noise...[/blue]")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            console.print("[green]Calibration complete.[/green]")

        self.available = True

        # Recognition settings
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    def listen_once(self, timeout: Optional[float] = None) -> Optional[str]:
        """Listen for a single command."""
        if not self.available or self.microphone is None:
            return None
        try:
            with self.microphone as source:
                console.print("[dim]Listening...[/dim]")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

            console.print("[dim]Processing speech...[/dim]")
            text = self.recognizer.recognize_google(audio).lower()
            console.print(f"[green]Heard: '{text}'[/green]")
            return text

        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            console.print(f"[red]Speech recognition error: {e}[/red]")
            return None

    def listen_for_wake_word(self) -> bool:
        """Listen for the wake word. Returns True if detected."""
        text = self.listen_once(timeout=5)
        if text and self.wake_word in text:
            return True
        return False

    def start_continuous_listening(self, callback: Callable[[str], None]):
        """Start continuous listening in background."""
        self.callback = callback
        self.listening = True

        def listen_loop():
            while self.listening:
                try:
                    text = self.listen_once(timeout=1)
                    if text and self.callback:
                        self.callback(text)
                except Exception as e:
                    console.print(f"[red]Listening error: {e}[/red]")

        self.thread = threading.Thread(target=listen_loop, daemon=True)
        self.thread.start()

    def stop_listening(self):
        """Stop continuous listening."""
        self.listening = False

    def listen_command(self) -> Optional[str]:
        """Listen for a command after wake word detection."""
        console.print(f"[cyan]Yes? I'm listening...[/cyan]")
        return self.listen_once(timeout=10)
