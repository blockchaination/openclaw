---
name: quant-engine
description: Run a paper-trading quant loop from exchange market data. Use when the user asks about backtesting, paper trading, or quant strategy execution.
metadata:
  {
    "openclaw": {
      "emoji": "📊",
      "requires": { "anyBins": ["python", "python3"] },
      "os": ["darwin", "linux", "win32"]
    }
  }
---

# Quant Engine

## Overview
Run a paper-trading quant loop: ingest market data, compute features, apply strategy, and simulate orders via a paper broker. No live exchange execution.

## When to use
Use this skill when the user asks about:
- Paper trading or backtesting
- Quant strategy execution
- Feature computation from order book / tick data
- Risk limits and position sizing

## Quick start

```bash
python {baseDir}/scripts/quant_engine.py run-paper --pair <pair>
```
