# Google Drive backup

Cold data (training `outputs*`) is offloaded to Google Drive at `gdrive:kd-backup`
via rclone, mirroring the original paths. The teacher model and logprobs cache stay
local. Training logs are backed up but kept local.

Credentials live in `.env` (gitignored): `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`.
The rclone remote is named `gdrive`.

## Scripts

| Script | Purpose |
|---|---|
| `gdrive-offload.sh` | Upload `outputs*` to Drive, verify checksums, then delete local copies |
| `gdrive-pull` | Fast multi threaded restore, e.g. `gdrive-pull outputs_gptq` |
| `gdrive-mount` | Mount the backup read only at `~/gdrive-kd` for browsing without downloading |
| `gdrive-synclogs` | Push training logs to Drive (also run every 30 min via cron) |

## Restore examples

```bash
gdrive-pull outputs_gptq                       # restore ./outputs_gptq
gdrive-pull cache/teacher_logprobs_r1_full     # if it was ever archived
gdrive-mount && ls ~/gdrive-kd/outputs         # browse without downloading
```
