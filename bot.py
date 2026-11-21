"""Telegram voice/video-note transcription. Run: python bot.py."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from speech import LocalTranscriber, MediaError
from telegram import BotCommand, Message, Update
from telegram.constants import ChatAction, ChatType
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("transcription_bot")
MAX_TELEGRAM_BYTES = 20_000_000
HELP = (
    "Перешли мне голосовое сообщение или видео-кружок — я отвечу текстом.\n\n"
    "Можно также записать сообщение прямо здесь. Мат сохраняю без цензуры. "
    "Длинную расшифровку отправляю несколькими сообщениями.\n\n"
    "Работаю в личном чате. Распознаю звук на своём сервере. "
    "Временные файлы удаляются после обработки.\n\n"
    "/whoami — узнать свой Telegram ID"
)


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    model: str = "tiny"
    language: str = "ru"
    cpu_threads: int = 1
    model_cache: Path = ROOT / ".models"
    offline: bool = False
    allowed_users: frozenset[int] = frozenset()
    max_duration: int = 300
    max_pending: int = 10
    max_per_user: int = 3
    run_mode: str = "polling"
    public_url: str = ""
    webhook_secret: str = ""
    port: int = 10000

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(ROOT / ".env")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise ConfigurationError(
                "Заполните TELEGRAM_BOT_TOKEN в файле .env. "
                "Образец настроек: .env.example."
            )
        try:
            users = frozenset(
                int(value.strip())
                for value in os.getenv("ALLOWED_USER_IDS", "").split(",")
                if value.strip()
            )
            duration = int(os.getenv("MAX_DURATION_SECONDS", "300"))
            threads = int(os.getenv("WHISPER_CPU_THREADS", "1"))
            pending = int(os.getenv("MAX_PENDING_JOBS", "10"))
            per_user = int(os.getenv("MAX_PENDING_PER_USER", "3"))
            port = int(os.getenv("PORT", "10000"))
        except ValueError as exc:
            raise ConfigurationError("ID, длительность и число задач должны быть целыми числами.") from exc
        if not 1 <= duration <= 7200 or not 1 <= threads <= 32 or not 1 <= pending <= 100 or not 1 <= per_user <= pending or any(uid < 1 for uid in users):
            raise ConfigurationError(
                "Проверь настройки: длительность 1–7200, потоки CPU 1–32, "
                "очередь 1–100, лимит пользователя 1–размер очереди, ID положительные."
            )
        cache = Path(os.getenv("MODEL_CACHE_DIR", ".models"))
        if not cache.is_absolute():
            cache = ROOT / cache
        mode = os.getenv("RUN_MODE", "polling").strip().lower()
        public_url = (os.getenv("PUBLIC_URL") or os.getenv("RENDER_EXTERNAL_URL", "")).strip().rstrip("/")
        secret = os.getenv("WEBHOOK_SECRET", "").strip()
        if mode not in {"polling", "webhook"} or not 1 <= port <= 65535:
            raise ConfigurationError("RUN_MODE должен быть polling или webhook; PORT — от 1 до 65535.")
        if mode == "webhook" and (not public_url.startswith("https://") or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", secret)):
            raise ConfigurationError("Для webhook нужны PUBLIC_URL с https:// и WEBHOOK_SECRET из 32–256 символов A-Z, a-z, 0-9, _ или -.")
        return cls(
            token,
            model=os.getenv("WHISPER_MODEL", "tiny").strip() or "tiny",
            language=os.getenv("WHISPER_LANGUAGE", "ru").strip(),
            cpu_threads=threads, model_cache=cache,
            offline=os.getenv("MODEL_OFFLINE", "0").strip() == "1",
            allowed_users=users, max_duration=duration,
            max_pending=pending, max_per_user=per_user,
            run_mode=mode, public_url=public_url, webhook_secret=secret, port=port,
        )


def split_text(text: str, limit: int = 4000) -> list[str]:
    """Split without losing characters; count UTF-16 units for Telegram emoji safety."""
    if limit < 2:
        raise ValueError("limit must be at least 2")
    chunks = []
    while text:
        units = 0
        end = 0
        for char in text:
            size = 2 if ord(char) > 0xFFFF else 1
            if units + size > limit:
                break
            units += size
            end += 1
        if end < len(text):
            boundary = max(text.rfind("\n", 0, end), text.rfind(" ", 0, end))
            if boundary >= end // 2:
                end = boundary + 1
        chunks.append(text[:end])
        text = text[end:]
    return chunks


class TranscriptionBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.slots = asyncio.Semaphore(1)
        self.pending_jobs = 0
        self.user_jobs: dict[int, int] = {}
        self.recognizer = LocalTranscriber(
            settings.model, settings.model_cache, settings.language,
            settings.cpu_threads, settings.offline,
        )

    async def startup(self, app: Application) -> None:
        LOGGER.info("Загружаю локальную модель %s. При первом запуске нужно скачать её файлы.", self.settings.model)
        await asyncio.to_thread(self.recognizer.load)
        await app.bot.set_my_commands([
            BotCommand("start", "Как пользоваться ботом"),
            BotCommand("help", "Помощь"),
            BotCommand("whoami", "Мой Telegram ID"),
        ])
        LOGGER.info("Бот запущен. Ожидаю голосовые и видео-кружки.")

    async def permitted(self, update: Update) -> bool:
        message = update.effective_message
        user = update.effective_user
        if not message or not user or message.chat.type != ChatType.PRIVATE:
            return False
        if self.settings.allowed_users and user.id not in self.settings.allowed_users:
            await message.reply_text("Доступ к этому боту ограничен владельцем.")
            return False
        return True

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self.permitted(update):
            await update.effective_message.reply_text(HELP)

    async def whoami(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # ID is deliberately available before access is granted, to configure the allowlist.
        if update.effective_user and update.effective_chat.type == ChatType.PRIVATE:
            await update.effective_message.reply_text(f"Твой Telegram ID: {update.effective_user.id}")

    async def transcribe(self, audio: Path) -> str:
        return await asyncio.to_thread(self.recognizer.transcribe, audio, self.settings.max_duration)

    async def media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self.permitted(update):
            return
        message = update.effective_message
        media = message.voice or message.video_note
        if not media:
            return
        if media.file_size and media.file_size > MAX_TELEGRAM_BYTES:
            await message.reply_text("Файл больше 20 МБ. Перешли запись меньшего размера.")
            return
        duration = media.duration
        if hasattr(duration, "total_seconds"):
            duration = duration.total_seconds()
        if duration > self.settings.max_duration:
            await message.reply_text(
                f"Запись длиннее {self.settings.max_duration} секунд. Раздели её на части."
            )
            return
        uid = update.effective_user.id
        if self.user_jobs.get(uid, 0) >= self.settings.max_per_user:
            await message.reply_text("У тебя уже несколько записей в очереди. Дождись ответов и отправь следующую.")
            return
        if self.pending_jobs >= self.settings.max_pending:
            await message.reply_text("Очередь заполнена. Попробуй переслать запись чуть позже.")
            return
        queued = self.pending_jobs > 0
        self.pending_jobs += 1
        self.user_jobs[uid] = self.user_jobs.get(uid, 0) + 1
        try:
            if queued:
                await message.reply_text("Запись принята в очередь. Пришлю текст, когда подойдёт её очередь.")
            async with self.slots:
                await self.process(message, media)
        finally:
            self.pending_jobs -= 1
            self.user_jobs[uid] -= 1
            if self.user_jobs[uid] == 0:
                del self.user_jobs[uid]

    async def process(self, message: Message, media) -> None:
        try:
            try:
                await message.get_bot().send_chat_action(message.chat_id, ChatAction.TYPING)
            except TelegramError:
                pass  # A typing indicator must not prevent transcription.
            with TemporaryDirectory(prefix="tg-transcript-") as directory:
                source = Path(directory) / ("message.ogg" if message.voice else "message.mp4")
                remote = await media.get_file()
                await remote.download_to_drive(source, read_timeout=90, write_timeout=90)
                if source.stat().st_size > MAX_TELEGRAM_BYTES:
                    raise MediaError("Файл больше 20 МБ. Перешли запись меньшего размера.")
                text = await self.transcribe(source)
            # Files are already deleted when sending the result.
            if not text:
                await message.reply_text("Не удалось разобрать речь. Попробуй запись с более чётким звуком.")
                return
            for part in split_text(text):
                await message.reply_text(part, parse_mode=None, do_quote=True)
        except MediaError as exc:
            await message.reply_text(str(exc))
        except TelegramError:
            # Let the central handler deal with delivery errors without repeating recognition.
            raise
        except Exception as exc:
            LOGGER.error("Ошибка обработки: %s", type(exc).__name__)
            await message.reply_text("Не удалось обработать запись. Попробуй переслать её ещё раз.")

    async def unsupported(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self.permitted(update):
            await update.effective_message.reply_text("Пришли голосовое сообщение или видео-кружок.")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Exception strings/tracebacks may contain the Telegram token or recording content.
    LOGGER.error("Ошибка Telegram/обработчика: %s", type(context.error).__name__)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("Не удалось отправить или получить сообщение. Попробуй ещё раз.")
        except TelegramError:
            pass


def build_application(settings: Settings) -> Application:
    service = TranscriptionBot(settings)
    app = (
        Application.builder().token(settings.telegram_token)
        .concurrent_updates(settings.max_pending + 4)
        .connect_timeout(30).read_timeout(90).write_timeout(90)
        .post_init(service.startup).build()
    )
    app.add_handler(CommandHandler(["start", "help"], service.help))
    app.add_handler(CommandHandler("whoami", service.whoami))
    app.add_handler(MessageHandler(filters.VOICE | filters.VIDEO_NOTE, service.media))
    app.add_handler(MessageHandler(filters.ALL, service.unsupported))
    app.add_error_handler(on_error)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Проверить настройки и библиотеки без скачивания модели")
    parser.add_argument("--download-model", action="store_true", help="Скачать/загрузить модель без подключения к Telegram")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "httpx2", "httpcore2", "huggingface_hub", "faster_whisper"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        settings = Settings.from_env()
        if args.check:
            import av
            import ctranslate2
            import faster_whisper
            print("Настройки заполнены, локальный движок доступен. Токен Telegram онлайн не проверялся.")
            return 0
        if args.download_model:
            LocalTranscriber(settings.model, settings.model_cache, settings.language, settings.cpu_threads, settings.offline).load()
            print("Локальная модель готова.")
            return 0
        asyncio.set_event_loop(asyncio.new_event_loop())
        app = build_application(settings)
        if settings.run_mode == "webhook":
            app.run_webhook(
                listen="0.0.0.0", port=settings.port, url_path="telegram",
                webhook_url=settings.public_url + "/telegram",
                secret_token=settings.webhook_secret,
                allowed_updates=["message"], drop_pending_updates=False,
                max_connections=1,
            )
        else:
            app.run_polling(allowed_updates=["message"], drop_pending_updates=False)
    except ConfigurationError as exc:
        print(f"Ошибка настройки: {exc}")
        return 1
    except Exception as exc:
        LOGGER.error("Не удалось запустить бота (%s). Проверь токен, сеть, загрузку модели и отсутствие второй копии бота.", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
