#!/usr/bin/env python3
"""
Diagnostic script for Telegram integration.
Prints whether TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set,
and whether Telegram integration appears enabled.
"""

from __future__ import annotations

import os


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    token_set = bool(token)
    chat_id_set = bool(chat_id)
    enabled = token_set and chat_id_set

    print("TELEGRAM_BOT_TOKEN: set" if token_set else "TELEGRAM_BOT_TOKEN: not set")
    print("TELEGRAM_CHAT_ID: set" if chat_id_set else "TELEGRAM_CHAT_ID: not set")
    print("Telegram integration: enabled" if enabled else "Telegram integration: disabled")


if __name__ == "__main__":
    main()
