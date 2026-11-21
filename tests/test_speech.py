import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import av
import numpy as np
from speech import LocalTranscriber, MediaError


def make_media(path, seconds=0.5, with_video=False):
    with av.open(str(path), "w") as container:
        audio = container.add_stream("aac" if with_video else "libopus", rate=48000)
        audio.layout = "mono"
        if with_video:
            video = container.add_stream("libx264", rate=25)
            video.width = video.height = 240
            video.pix_fmt = "yuv420p"
            for index in range(int(seconds * 25)):
                frame = av.VideoFrame.from_ndarray(np.zeros((240, 240, 3), dtype=np.uint8), format="rgb24")
                frame.pts = index
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
        count = int(48000 * seconds)
        samples = (np.sin(np.arange(count) * 2 * np.pi * 440 / 48000) * 8000).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 48000
        frame.time_base = Fraction(1, 48000)
        frame.pts = 0
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


class SpeechTests(unittest.TestCase):
    def test_real_ogg_and_mp4_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename, video in (("voice.ogg", False), ("circle.mp4", True)):
                path = Path(directory) / filename
                make_media(path, with_video=video)
                audio = LocalTranscriber.decode(path, 3)
                self.assertEqual(audio.dtype, np.float32)
                self.assertGreater(len(audio), 7000)
                self.assertLess(len(audio), 10000)

    def test_real_duration_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.ogg"
            make_media(path, seconds=2)
            with self.assertRaisesRegex(MediaError, "длиннее"):
                LocalTranscriber.decode(path, 1)

    def test_broken_media(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.mp4"
            path.write_bytes(b"not a video")
            with self.assertRaises(MediaError):
                LocalTranscriber.decode(path, 3)

    def test_model_output_is_returned_without_censorship_or_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.ogg"
            make_media(path)
            recognizer = LocalTranscriber("tiny", Path(directory), "ru", 1)
            recognizer.model = Mock()
            recognizer.model.transcribe.return_value = (iter([
                SimpleNamespace(text=" Блядь, это "), SimpleNamespace(text="дословно."),
            ]), None)
            self.assertEqual(recognizer.transcribe(path, 3), "Блядь, это дословно.")
            kwargs = recognizer.model.transcribe.call_args.kwargs
            self.assertEqual(kwargs["task"], "transcribe")
            self.assertTrue(kwargs["vad_filter"])
            self.assertNotIn("initial_prompt", kwargs)
