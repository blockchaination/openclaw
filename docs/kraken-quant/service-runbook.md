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
