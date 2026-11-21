import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from telegram import Update
from bot import MAX_TELEGRAM_BYTES, ConfigurationError, Settings, TranscriptionBot, build_application, split_text
from speech import MediaError

SETTINGS = Settings("123456:local-test-token")


def make_update(kind="voice", uid=42, **attributes):
    media = SimpleNamespace(file_size=100, duration=4, get_file=AsyncMock())
    for key, value in attributes.items():
        setattr(media, key, value)
    message = SimpleNamespace(
        voice=media if kind == "voice" else None,
        video_note=media if kind == "video_note" else None,
        chat=SimpleNamespace(type="private"), chat_id=uid,
        reply_text=AsyncMock(),
        get_bot=lambda: SimpleNamespace(send_chat_action=AsyncMock()),
    )
    return SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=uid))


class ConfigurationTests(unittest.TestCase):
    def test_free_public_defaults_and_no_api_key(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "token"}, clear=True), patch("bot.load_dotenv"):
            settings = Settings.from_env()
        self.assertFalse(settings.allowed_users)
        self.assertEqual(settings.model, "tiny")
        self.assertEqual(settings.run_mode, "polling")

    def test_webhook_configuration_requires_secret_and_https(self):
        env = {"TELEGRAM_BOT_TOKEN": "token", "RUN_MODE": "webhook"}
        with patch.dict(os.environ, env, clear=True), patch("bot.load_dotenv"):
            with self.assertRaises(ConfigurationError):
                Settings.from_env()
            os.environ["RENDER_EXTERNAL_URL"] = "https://example.onrender.com"
            os.environ["WEBHOOK_SECRET"] = "a" * 40
            self.assertEqual(Settings.from_env().public_url, "https://example.onrender.com")

    def test_configuration_validation(self):
        for env in ({}, {"TELEGRAM_BOT_TOKEN": "token", "WHISPER_CPU_THREADS": "0"},
                    {"TELEGRAM_BOT_TOKEN": "token", "MAX_PENDING_JOBS": "bad"}):
            with patch.dict(os.environ, env, clear=True), patch("bot.load_dotenv"):
                with self.assertRaises(ConfigurationError):
                    Settings.from_env()

    def test_text_preserves_profanity_emoji_and_markup(self):
        original = ("Блядь, <текст> & **слова** 😀\n" * 1000) + "конец"
        parts = split_text(original)
        self.assertEqual("".join(parts), original)
        self.assertTrue(all(len(p.encode("utf-16-le")) // 2 <= 4000 for p in parts))
        self.assertEqual(split_text(""), [])
        self.assertEqual("".join(split_text("😀" * 5000)), "😀" * 5000)

    def test_real_telegram_voice_and_video_note_routes(self):
        app = build_application(SETTINGS)
        handler = app.handlers[0][2]
        for kind in ("voice", "video_note"):
            media = {"file_id": "abc", "file_unique_id": "def", "duration": 4}
            if kind == "video_note":
                media["length"] = 240
            update = Update.de_json({"update_id": 1, "message": {
                "message_id": 1, "date": 1700000000,
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42, "is_bot": False, "first_name": "Test"}, kind: media,
            }}, app.bot)
            self.assertTrue(handler.check_update(update))


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_voice_and_video_notes_are_processed(self):
        for kind in ("voice", "video_note"):
            service = TranscriptionBot(SETTINGS)
            service.process = AsyncMock()
            await service.media(make_update(kind), None)
            service.process.assert_awaited_once()
            self.assertEqual(service.pending_jobs, 0)
            self.assertEqual(service.user_jobs, {})

    async def test_oversized_and_long_media_do_not_reach_recognizer(self):
        for attributes in ({"file_size": MAX_TELEGRAM_BYTES + 1}, {"duration": 301}):
            service = TranscriptionBot(SETTINGS)
            service.process = AsyncMock()
            update = make_update(**attributes)
            await service.media(update, None)
            service.process.assert_not_awaited()
            update.effective_message.reply_text.assert_awaited_once()

    async def test_public_access_and_groups(self):
        service = TranscriptionBot(SETTINGS)
        service.process = AsyncMock()
        await service.media(make_update(uid=77), None)
        service.process.assert_awaited_once()
        group = make_update()
        group.effective_message.chat.type = "group"
        await service.media(group, None)
        group.effective_message.reply_text.assert_not_awaited()

    async def test_queue_serializes_two_users_without_losing_recordings(self):
        service = TranscriptionBot(SETTINGS)
        started, release = asyncio.Event(), asyncio.Event()
        processed = []

        async def process(message, media):
            processed.append(message.chat_id)
            if len(processed) == 1:
                started.set()
                await release.wait()

        service.process = process
        first = asyncio.create_task(service.media(make_update(uid=42), None))
        await started.wait()
        update = make_update(uid=77)
        second = asyncio.create_task(service.media(update, None))
        await asyncio.sleep(0)
        self.assertEqual(processed, [42])
        self.assertEqual(service.pending_jobs, 2)
        update.effective_message.reply_text.assert_awaited_once()
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(processed, [42, 77])
        self.assertEqual(service.pending_jobs, 0)

    async def test_queue_limits_and_release_on_failure(self):
        service = TranscriptionBot(SETTINGS)
        service.process = AsyncMock()
        service.pending_jobs = SETTINGS.max_pending
        await service.media(make_update(), None)
        service.pending_jobs = 0
        service.user_jobs[42] = SETTINGS.max_per_user
        await service.media(make_update(), None)
        service.process.assert_not_awaited()
        service.user_jobs.clear()
        service.process.side_effect = RuntimeError("test")
        with self.assertRaises(RuntimeError):
            await service.media(make_update(), None)
        self.assertEqual(service.pending_jobs, 0)
        self.assertFalse(service.slots.locked())

    async def test_cleanup_and_uncensored_output_on_success_and_error(self):
        for fail in (False, True):
            service = TranscriptionBot(SETTINGS)
            message = make_update().effective_message
            paths = []

            async def download(path, **kwargs):
                path.write_bytes(b"recording")
                paths.append(path)

            message.voice.get_file.return_value = SimpleNamespace(download_to_drive=AsyncMock(side_effect=download))
            service.transcribe = AsyncMock(return_value="Блядь, <это> дословно.")
            if fail:
                service.transcribe.side_effect = MediaError("Ошибка звука")
            await service.process(message, message.voice)
            self.assertTrue(paths)
            self.assertTrue(all(not path.exists() for path in paths))
            if not fail:
                self.assertEqual(message.reply_text.call_args.args[0], "Блядь, <это> дословно.")
                self.assertIsNone(message.reply_text.call_args.kwargs["parse_mode"])
                self.assertTrue(message.reply_text.call_args.kwargs["do_quote"])
