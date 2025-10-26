#!/usr/bin/env python3
"""
AI Girlfriend "Lana" — Telegram bot (MVP) with robust fallbacks:
- Works without python-telegram-bot (local sandbox)
- Works without interactive stdin (script/STDIN/demo)
- Works in read-only filesystems (temp DB → shared in-memory DB)
- Resilient DB calls (auto-retry & safe handlers)
- Safe stdin handling (no crashes when stdin is unavailable)

Why this patch?
You saw `OSError: [Errno 29] I/O error` from reading stdin in a sandbox.
Some environments disallow `sys.stdin.read()`. This version now uses a
**safe stdin reader** that catches OSError and gracefully falls back to a demo
or exits, so local mode never crashes.

Quick start
1) Create a bot with @BotFather → grab the token
2) Env vars (bash):
   export TELEGRAM_BOT_TOKEN="your_bot_token"   # only for Telegram mode
   export OPENAI_API_KEY="your_openai_api_key"  # optional; stubbed if missing
   export FREE_MESSAGES_PER_DAY=15               # optional
   export HISTORY_TURNS=16                       # optional
   export LANA_DB="/path/to/lana.db"            # optional; defaults to temp dir
   export LANA_MODE=telegram|local               # optional; auto-detects
3) Install deps for Telegram mode:
   pip install python-telegram-bot==21.6 openai==1.51.2 python-dotenv==1.0.1
4) Run:
   • Local sandbox (interactive):      python LanaTelegramBot.py --local
   • Local with scripted input:        python LanaTelegramBot.py --local --script inputs.txt
   • Local reading from STDIN:         echo -e "hi\n/quit" | python LanaTelegramBot.py --local
   • Self-tests:                       python LanaTelegramBot.py --test
   • Telegram bot:                     python LanaTelegramBot.py

Notes
- PAYWALL_HOOK is a placeholder; later wire Telegram Stars / CryptoBot / Patreon / Gumroad.
- For production deploy on Railway / Render / Fly.io, etc.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sqlite3
import sys
import tempfile
from datetime import date
from typing import Callable, Iterable, List, Tuple, TypeVar, TextIO

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*_, **__):  # no-op if not installed
        return None

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("LANA_MODEL", "gpt-4o-mini")
SYSTEM_NAME = "Lana"
FREE_MESSAGES_PER_DAY = int(os.getenv("FREE_MESSAGES_PER_DAY", "15"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "16"))

# Choose a writable default DB path in temp dir; allow override via env
DEFAULT_DB_PATH = os.path.join(tempfile.gettempdir(), "lana.db")
DB_PATH = os.getenv("LANA_DB", DEFAULT_DB_PATH)

# DB DSN / URI state (may switch to shared in-memory if file I/O fails)
DB_DSN = DB_PATH
DB_URI = False  # set True if using URI (e.g., memory mode)
SQLITE_TIMEOUT = 10.0

# Try Telegram imports, but don't crash if missing
TELEGRAM_AVAILABLE = True
PARSEMODE_HTML = False
try:
    from telegram import Update  # type: ignore
    from telegram.constants import ParseMode  # type: ignore
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters,
    )  # type: ignore
    PARSEMODE_HTML = True
except ModuleNotFoundError:
    TELEGRAM_AVAILABLE = False

# OpenAI client (optional for tests)
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI  # type: ignore
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    OPENAI_AVAILABLE = False
    _openai_client = None

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
log = logging.getLogger("lana")

# ──────────────────────────────────────────────────────────────────────────────
# DB helpers (resilient)
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path or "")
    if parent and not os.path.exists(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception as e:
            log.warning("Failed to create DB directory '%s': %s", parent, e)


def _switch_to_memory_db():
    global DB_DSN, DB_URI
    DB_DSN = "file:lana_memdb?mode=memory&cache=shared"
    DB_URI = True
    log.warning("Switching to shared in-memory SQLite (file I/O unavailable). Data persists until process exit.")


def db():
    return sqlite3.connect(DB_DSN, uri=DB_URI, timeout=SQLITE_TIMEOUT, check_same_thread=False)


# Resilient executor for DB ops (auto-switch to memory on I/O errors)
T = TypeVar("T")

def _db_try(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except (OSError, sqlite3.OperationalError) as e:
        global DB_URI
        if not DB_URI:
            log.warning("DB op failed (%s). Falling back to memory DB and retrying once.", e)
            _switch_to_memory_db()
            return fn()
        raise


def init_db():
    def _init():
        if not DB_URI:
            _ensure_parent_dir(DB_DSN)
        con = db(); cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                messages_today INTEGER DEFAULT 0,
                last_reset DATE
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS convo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                role TEXT CHECK(role IN ('user','assistant','system')),
                content TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.commit(); con.close()
    _db_try(_init)


def get_user(user_id: int, username: str | None) -> Tuple[int, int, str | None]:
    def _get():
        con = db(); cur = con.cursor()
        cur.execute("SELECT user_id, messages_today, last_reset FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO users (user_id, username, messages_today, last_reset) VALUES (?, ?, 0, ?)",
                (user_id, username, date.today().isoformat()),
            )
            con.commit()
            row = (user_id, 0, date.today().isoformat())
        else:
            last_reset = row[2]
            if last_reset is None or last_reset != date.today().isoformat():
                cur.execute("UPDATE users SET messages_today = 0, last_reset = ? WHERE user_id = ?", (date.today().isoformat(), user_id))
                con.commit()
                row = (row[0], 0, date.today().isoformat())
        con.close()
        return row[0], row[1], row[2]
    return _db_try(_get)


def inc_user_counter(user_id: int):
    def _inc():
        con = db(); cur = con.cursor()
        cur.execute("UPDATE users SET messages_today = COALESCE(messages_today,0) + 1 WHERE user_id = ?", (user_id,))
        con.commit(); con.close()
    _db_try(_inc)


def add_msg(user_id: int, role: str, content: str):
    def _add():
        con = db(); cur = con.cursor()
        cur.execute("INSERT INTO convo (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
        cur.execute(
            """
            DELETE FROM convo WHERE id IN (
                SELECT id FROM convo WHERE user_id = ? ORDER BY id DESC LIMIT -1 OFFSET ?
            )
            """,
            (user_id, HISTORY_TURNS * 2),
        )
        con.commit(); con.close()
    _db_try(_add)


def get_history(user_id: int) -> List[Tuple[str, str]]:
    def _hist():
        con = db(); cur = con.cursor()
        cur.execute("SELECT role, content FROM convo WHERE user_id = ? ORDER BY id ASC", (user_id,))
        rows = cur.fetchall(); con.close(); return rows
    return _db_try(_hist)

# ──────────────────────────────────────────────────────────────────────────────
# Core chat logic (transport-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are {SYSTEM_NAME}, an AI girlfriend and supportive companion.
Core traits:
- Warm, playful, witty; flirty but tasteful (PG-13). No explicit sexual content.
- Emotionally intelligent: validate feelings, ask short follow-up questions.
- Concise by default (2–5 sentences), but expand if user asks.
- Mirror the user's language automatically (reply in the same language and register). If user mixes languages, choose the dominant language.
- Use light emoji occasionally if it fits the tone.
Boundaries & Safety:
- Refuse explicit sexual content, illegal, violent, self-harm, medical/financial/legal advice beyond general support; suggest safer alternatives.
- If asked NSFW content, gently decline and steer to romantic/wholesome topics.
Memory:
- If the user shares preferences (likes/dislikes, hobbies, birthdays), naturally remember them during the conversation.
Style:
- Address the user by name if available from platform context (e.g., Telegram username), otherwise use a friendly term.
"""


def _openai_reply(messages: List[dict]) -> str:
    """Call OpenAI if available; otherwise return a stub useful for tests."""
    if OPENAI_AVAILABLE and _openai_client and OPENAI_API_KEY:
        try:
            resp = _openai_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.9,
                max_tokens=600,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.exception("OpenAI error: %s", e)
            return "У меня небольшой сбой с мозгами 🤯 Попробуешь ещё раз?"
    # Fallback stub: mirror language + light persona
    user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return f"Я тут с тобой, милашка 💫\n\nТы написал(а): {user_msg}"


def generate_reply(user_id: int, user_text: str, username: str | None) -> str:
    history = get_history(user_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if username:
        messages.append({"role": "system", "content": f"User telegram username is @{username}."})
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    return _openai_reply(messages)


def PAYWALL_HOOK(user_id: int) -> str:
    return (
        "Бесплатный лимит на сегодня исчерпан. ✨\n\n"
        "Скоро тут появятся способы подписки: Telegram Stars / CryptoBot / Patreon / Gumroad.\n"
        "Хочешь — напиши, какой способ удобнее, я подсуну разработчику 😉"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Telegram transport (only if installed)
# ──────────────────────────────────────────────────────────────────────────────

if TELEGRAM_AVAILABLE:
    async def _tg_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            get_user(user.id, user.username)
            await update.message.reply_text(
                (
                    "Привет! Я Lana — твоя ИИ-компаньонка. 💫\n\n"
                    f"Пиши на любом языке — я подстроюсь. Первый день даю {FREE_MESSAGES_PER_DAY} сообщений бесплатно.\n"
                    "Команды: /help /reset /stats."
                )
            )
        except Exception as e:
            log.exception("_tg_start error: %s", e)

    async def _tg_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text(
                (
                    "Я — Lana: тёплая, остроумная, иногда флиртую 😉\n\n"
                    "Что я умею:\n"
                    "• Поддержать, поболтать, обсудить планы.\n"
                    "• Практиковать языки — отвечаю на том же языке.\n"
                    "• Помнить твои предпочтения в рамках чата.\n\n"
                    "Безопасность: PG-13, без откровенного контента.\n"
                    "Команды: /reset — забыть текущий контекст, /stats — лимит на сегодня."
                )
            )
        except Exception as e:
            log.exception("_tg_help error: %s", e)

    async def _tg_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            def _do_reset():
                con = db(); cur = con.cursor()
                cur.execute("DELETE FROM convo WHERE user_id = ?", (user.id,))
                con.commit(); con.close()
            _db_try(_do_reset)
            await update.message.reply_text("Я всё забыла про этот разговор. Начнём заново ✨")
        except Exception as e:
            log.exception("_tg_reset error: %s", e)
            try:
                await update.message.reply_text("Хм, не смогла очистить историю из-за сбоя хранилища. Попробуем позже.")
            except Exception:
                pass

    async def _tg_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            _uid, used, _last = get_user(user.id, user.username)
            left = max(0, FREE_MESSAGES_PER_DAY - used)
            await update.message.reply_text(f"Сегодня осталось сообщений: {left}/{FREE_MESSAGES_PER_DAY}")
        except Exception as e:
            log.exception("_tg_stats error: %s", e)

    async def _tg_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            uid, used, _ = get_user(user.id, user.username)
            if used >= FREE_MESSAGES_PER_DAY:
                await update.message.reply_text(PAYWALL_HOOK(uid))
                return
            user_text = (update.message.text or "").strip()
            add_msg(uid, "user", user_text)
            reply = generate_reply(uid, user_text, user.username)
            add_msg(uid, "assistant", reply)
            inc_user_counter(uid)
            if PARSEMODE_HTML:
                await update.message.reply_text(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            else:
                await update.message.reply_text(reply)
        except Exception as e:
            log.exception("_tg_text error: %s", e)
            try:
                await update.message.reply_text("Упс, у меня затык с базой/сетью. Напиши ещё раз чуть позже.")
            except Exception:
                pass

    async def _tg_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.exception("Exception while handling an update: %s", context.error)
        if isinstance(update, Update) and getattr(update, "effective_message", None):
            try:
                await update.effective_message.reply_text("Ой... что-то пошло не так. Попробуем ещё раз чуть позже.")
            except Exception:
                pass

    def run_telegram_bot():
        if not BOT_TOKEN:
            raise SystemExit("TELEGRAM_BOT_TOKEN is missing. Set it or use --local/--test.")
        init_db()
        app = Application.builder().token(BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", _tg_start))
        app.add_handler(CommandHandler("help", _tg_help))
        app.add_handler(CommandHandler("reset", _tg_reset))
        app.add_handler(CommandHandler("stats", _tg_stats))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), _tg_text))
        app.add_error_handler(_tg_error)
        log.info("Lana is alive (Telegram). Free/day=%s, model=%s", FREE_MESSAGES_PER_DAY, MODEL)
        app.run_polling(allowed_updates=Update.ALL_TYPES)

# ──────────────────────────────────────────────────────────────────────────────
# Local CLI sandbox (no Telegram needed) + tests
# ──────────────────────────────────────────────────────────────────────────────

def _read_stdin_safely(stream: TextIO) -> str:
    """Read all text from stdin-like stream, returning '' on OSError/unsupported.
    This prevents `OSError: [Errno 29] I/O error` in restricted sandboxes.
    """
    try:
        # Some sandboxes have stream but .read() raises OSError
        return stream.read()
    except (OSError, AttributeError):
        return ""


def run_local_session(lines: Iterable[str]) -> None:
    """Run a local session using an iterable of input lines.
    Safe in non-interactive sandboxes. Prints replies to stdout.
    """
    init_db()
    fake_user_id = 1
    fake_username = "local_user"
    get_user(fake_user_id, fake_username)
    for raw in lines:
        user_text = (raw or "").strip()
        if not user_text:
            continue
        if user_text in {"/quit", ":q", "exit"}:
            break
        if user_text == "/reset":
            def _do_reset():
                con = db(); cur = con.cursor()
                cur.execute("DELETE FROM convo WHERE user_id = ?", (fake_user_id,))
                con.commit(); con.close()
            _db_try(_do_reset)
            print("lana> Ок, начнём сначала ✨")
            continue
        uid, used, _ = get_user(fake_user_id, fake_username)
        if used >= FREE_MESSAGES_PER_DAY:
            print("lana>", PAYWALL_HOOK(uid))
            continue
        add_msg(uid, "user", user_text)
        reply = generate_reply(uid, user_text, fake_username)
        add_msg(uid, "assistant", reply)
        inc_user_counter(uid)
        print("lana>", reply)


def run_local_cli(script_path: str | None = None):
    """Interactive local chat or non-interactive fallback.

    Behavior:
      • If `script_path` provided → read lines from file.
      • Else if stdin is not a TTY and has data → read lines from stdin.
      • Else if stdin is not a TTY and has no data → run a short demo and exit.
      • Else → interactive loop with input().
    """
    # Script file mode
    if script_path:
        with open(script_path, "r", encoding="utf-8") as f:
            return run_local_session(f.readlines())

    # Non-interactive STDIN mode
    if not sys.stdin.isatty():
        data = _read_stdin_safely(sys.stdin)
        if data:
            return run_local_session(io.StringIO(data))
        # No input available → run demo and exit
        demo = [
            "Привет, Лана!", "Как твои дела?", "/reset", "Давай начнём заново", "/quit",
        ]
        print("Lana non-interactive demo ✨ (no stdin detected)")
        return run_local_session(demo)

    # Interactive TTY mode
    init_db()
    print("Lana local sandbox ✨ (type /quit to exit, /reset to clear)")
    fake_user_id = 1
    fake_username = "local_user"
    get_user(fake_user_id, fake_username)
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            print() ; break
        if user_text in {"/quit", ":q", "exit"}:
            break
        if user_text == "/reset":
            def _do_reset():
                con = db(); cur = con.cursor()
                cur.execute("DELETE FROM convo WHERE user_id = ?", (fake_user_id,))
                con.commit(); con.close()
            _db_try(_do_reset)
            print("lana> Ок, начнём сначала ✨")
            continue
        uid, used, _ = get_user(fake_user_id, fake_username)
        if used >= FREE_MESSAGES_PER_DAY:
            print("lana>", PAYWALL_HOOK(uid))
            continue
        add_msg(uid, "user", user_text)
        reply = generate_reply(uid, user_text, fake_username)
        add_msg(uid, "assistant", reply)
        inc_user_counter(uid)
        print("lana>", reply)


# Minimal tests (no external frameworks required)

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def run_tests():
    print("Running Lana self-tests…")
    # Isolated test DB in temp; if file fails, code will fallback to memory
    global DB_PATH, DB_DSN, DB_URI
    DB_PATH = os.path.join(tempfile.gettempdir(), "lana_test.db")
    DB_DSN = DB_PATH
    DB_URI = False
    init_db()

    # 1) Daily counter + paywall
    uid = 42
    get_user(uid, "tester")
    for _ in range(FREE_MESSAGES_PER_DAY):
        add_msg(uid, "user", "hello")
        _ = generate_reply(uid, "hello", "tester")
        add_msg(uid, "assistant", "ok")
        inc_user_counter(uid)
    _uid, used, _ = get_user(uid, "tester")
    _assert(used == FREE_MESSAGES_PER_DAY, "Counter didn't reach free limit")
    pay = PAYWALL_HOOK(uid)
    _assert("лимит" in pay or "limit" in pay.lower(), "Paywall text missing hint about limit")

    # 2) History trimming
    many = HISTORY_TURNS * 2 + 5
    for i in range(many):
        add_msg(uid, "user", f"msg {i}")
    history = get_history(uid)
    _assert(len(history) <= HISTORY_TURNS * 2, "History trimming exceeded bound")

    # 3) Language mirroring stub (works in stub mode)
    add_msg(uid, "user", "Привет, как дела?")
    reply = generate_reply(uid, "Привет, как дела?", "tester")
    _assert(isinstance(reply, str) and len(reply) > 0, "Reply should be a non-empty string")
    _assert("Привет, как дела?" in reply or "как дела" in reply, "Fallback reply should echo user text")

    # 4) Reset logic
    def _do_reset():
        con = db(); cur = con.cursor()
        cur.execute("DELETE FROM convo WHERE user_id = ?", (uid,))
        con.commit(); con.close()
    _db_try(_do_reset)
    _assert(len(get_history(uid)) == 0, "Reset should clear history")

    # 5) Non-interactive session test (no stdin)
    transcript = [
        "hi", "what's up?", "/reset", "again", "exit"
    ]
    run_local_session(transcript)
    _uid2, used2, _ = get_user(1, "local_user")
    _assert(used2 >= 1, "Non-interactive session didn't increment counter")

    # 6) Memory-DB fallback test (shared cache across connections)
    DB_DSN = "file:lana_memdb_test?mode=memory&cache=shared"; DB_URI = True
    init_db()
    get_user(777, "mem")
    add_msg(777, "user", "hello mem")
    con2 = db(); cur2 = con2.cursor()
    cur2.execute("SELECT COUNT(*) FROM convo WHERE user_id=777")
    cnt = cur2.fetchone()[0]; con2.close()

    _assert(cnt >= 1, "Shared memory DB did not persist across connections")

    # 7) NEW: Safe-stdin unit: simulate stream whose .read() raises OSError
    class _Boom:
        def read(self):
            raise OSError(29, "I/O error")
    data = _read_stdin_safely(_Boom())
    _assert(data == "", "Safe stdin should return empty string on OSError")

    print("All tests passed ✔")

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lana — AI Girlfriend bot")
    parser.add_argument("--local", action="store_true", help="Run local CLI (no Telegram)")
    parser.add_argument("--script", type=str, default=None, help="Path to a file with input lines for local run")
    parser.add_argument("--test", action="store_true", help="Run self-tests and exit")
    args = parser.parse_args()

    mode = os.getenv("LANA_MODE")
    if args.test:
        run_tests()
        sys.exit(0)

    if args.local or (mode == "local") or (not TELEGRAM_AVAILABLE):
        if not TELEGRAM_AVAILABLE:
            log.info("python-telegram-bot not found — starting local sandbox.")
        run_local_cli(script_path=args.script)
    else:
        if not TELEGRAM_AVAILABLE:
            raise SystemExit("python-telegram-bot is not installed. Use `--local` or install deps.")
        if not BOT_TOKEN:
            raise SystemExit("TELEGRAM_BOT_TOKEN is missing. Set it or use --local/--test.")
        run_telegram_bot()
