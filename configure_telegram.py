"""Verify the token and configure the bot profile, without printing credentials."""

import asyncio
from telegram import Bot
from bot import Settings


async def main():
    settings = Settings.from_env()
    async with Bot(settings.telegram_token) as bot:
        account = await bot.get_me()
        await bot.set_my_short_description("Голосовые и видео-кружки в текст. Бесплатно, без цензуры мата.")
        await bot.set_my_description(
            "Перешли голосовое сообщение или видео-кружок — получи текст в ответ. "
            "Мат сохраняется без цензуры. Распознавание выполняется на сервере бота. "
            "На бесплатном сервере первый ответ после перерыва может занять больше времени."
        )
        print(f"Telegram bot verified and configured: https://t.me/{account.username}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"Telegram setup failed: {type(exc).__name__}")
        raise SystemExit(1)
