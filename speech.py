"""Free speech recognition on the host CPU. No remote transcription API."""

from pathlib import Path


class MediaError(ValueError):
    pass


class LocalTranscriber:
    def __init__(self, model: str, cache: Path, language: str, threads: int, offline: bool = False):
        self.model_name = model
        self.cache = cache
        self.language = language or None
        self.threads = threads
        self.offline = offline
        self.model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self.cache.mkdir(parents=True, exist_ok=True)
        self.model = WhisperModel(
            self.model_name, device="cpu", compute_type="int8",
            cpu_threads=self.threads, num_workers=1,
            download_root=str(self.cache), local_files_only=self.offline,
        )

    @staticmethod
    def decode(path: Path, max_duration: int):
        """Decode only audio, including MP4 audio; enforce the limit on actual samples."""
        import av
        import numpy as np

        samples = []
        count = 0
        try:
            with av.open(str(path)) as container:
                if not container.streams.audio:
                    raise MediaError("В записи нет звуковой дорожки.")
                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                for frame in container.decode(audio=0):
                    for converted in resampler.resample(frame):
                        chunk = converted.to_ndarray().flatten()
                        count += len(chunk)
                        if count > max_duration * 16000:
                            raise MediaError(f"Запись длиннее {max_duration} секунд. Раздели её на части.")
                        samples.append(chunk)
                for converted in resampler.resample(None):
                    chunk = converted.to_ndarray().flatten()
                    count += len(chunk)
                    if count > max_duration * 16000:
                        raise MediaError(f"Запись длиннее {max_duration} секунд. Раздели её на части.")
                    samples.append(chunk)
        except MediaError:
            raise
        except (av.error.FFmpegError, ValueError, EOFError) as exc:
            raise MediaError("Не удалось прочитать звук. Попробуй отправить другую запись.") from exc
        if not samples:
            raise MediaError("В записи не найден звук.")
        return np.concatenate(samples).astype(np.float32) / 32768.0

    def transcribe(self, path: Path, max_duration: int) -> str:
        if self.model is None:
            raise RuntimeError("Local model is not loaded")
        audio = self.decode(path, max_duration)
        # No vocabulary blacklist, rewriting or profanity replacement.
        # VAD avoids inventing text over silence. No example swear words are injected.
        segments, _ = self.model.transcribe(
            audio, language=self.language, task="transcribe", beam_size=5,
            vad_filter=True, condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
