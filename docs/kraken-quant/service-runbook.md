# OpenClaw Kraken Quant Service Runbook

## Install the service on the VPS

```bash
sudo cp /opt/openclaw-kraken-quant/scripts/systemd/openclaw.service /etc/systemd/system/
sudo systemctl daemon-reload
```

If credentials are not in `/root/.openclaw-kraken.env`, edit the service file before copying:

```bash
# Edit EnvironmentFile= to match your env file path, e.g.:
# EnvironmentFile=-/home/ubuntu/.openclaw-kraken.env
```

## Start the service

```bash
sudo systemctl start openclaw
```

## Stop the service

```bash
sudo systemctl stop openclaw
```

## Restart the service

```bash
sudo systemctl restart openclaw
```

## Check status

```bash
sudo systemctl status openclaw
```

## View logs

```bash
sudo journalctl -u openclaw -f
```

## Enable live trading manually (outside the service)

The service runs in **observation mode only** (no `--enable-live-orders`). To place real orders, run the engine manually:

```bash
cd /opt/openclaw-kraken-quant
source ~/.openclaw-kraken.env
PYTHONPATH=skills/quant-engine/scripts python3 skills/quant-engine/scripts/quant_engine.py \
  --pair XBTUSD --live --enable-live-orders --usd-order-size 5 --iterations 10
```

The kill switch (`/tmp/openclaw.kill`) applies to both the service and manual runs.

## Operator status command

Print a compact status summary from the logs (read-only; does not affect trading):

```bash
cd /opt/openclaw-kraken-quant
python3 skills/quant-engine/scripts/operator_status.py
```

Optional arguments:

- `--status-file PATH` — override status JSON path (default: `logs/status.json`)
- `--trade-events-file PATH` — override trade events JSONL path (default: `logs/trade_events.jsonl`)
- `--tail N` — show last N trade events (default: 10)

## Operator briefing (daily / weekly)

Generate a compact briefing from logs (read-only; does not affect trading):

**Daily briefing (last 24 hours, default):**

```bash
cd /opt/openclaw-kraken-quant
python3 skills/quant-engine/scripts/operator_briefing.py
```

**Weekly briefing (last 7 days):**

```bash
python3 skills/quant-engine/scripts/operator_briefing.py --days 7
```

Optional arguments:

- `--hours N` — briefing window: last N hours
- `--days N` — briefing window: last N days
- `--status-file PATH` — override status JSON path
- `--trade-events-file PATH` — override trade events JSONL path
