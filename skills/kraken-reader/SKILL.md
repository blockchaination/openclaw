---
name: kraken-reader
description: Read public Kraken market data (instruments, ticker, trades, order book). Use when the user asks about Kraken prices, order books, or trade data.
metadata:
  {
    "openclaw": {
      "emoji": "🦑",
      "requires": { "anyBins": ["python", "python3"] },
      "os": ["darwin", "linux", "win32"]
    }
  }
---

# Kraken Reader

## Overview
Fetch public Kraken market data: list instruments, ticker, recent trades, and order book. Uses the Kraken REST API. No API key is required for public endpoints.

## When to use
Use this skill when the user asks about:
- Kraken spot prices or ticker data
- Order book depth on Kraken
- Recent trades for a pair
- Available trading pairs on Kraken

## Quick start

```bash
python {baseDir}/scripts/kraken_reader.py list-instruments
python {baseDir}/scripts/kraken_reader.py ticker --pair <pair>
python {baseDir}/scripts/kraken_reader.py trades --pair <pair>
python {baseDir}/scripts/kraken_reader.py orderbook --pair <pair>
```
