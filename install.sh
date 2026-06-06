#!/usr/bin/env bash
#
# Установка Local Zoom Transcriber одной командой (macOS).
# Проверяет Python и ffmpeg, создаёт .venv и ставит зависимости.
#
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Проверяю Python 3.10+..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "ОШИБКА: Python 3 не найден. Установите Python 3.10+ (brew install python)." >&2
  exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ОШИБКА: нужен Python 3.10+, а найден $(python3 --version)." >&2
  exit 1
fi
echo "    $(python3 --version)"

echo "==> Проверяю ffmpeg..."
if command -v ffprobe >/dev/null 2>&1; then
  echo "    ffmpeg уже установлен"
elif command -v brew >/dev/null 2>&1; then
  echo "    ffmpeg не найден — ставлю через Homebrew..."
  brew install ffmpeg
else
  echo "ОШИБКА: ffmpeg не найден, а Homebrew недоступен." >&2
  echo "Установите Homebrew (https://brew.sh), затем: brew install ffmpeg" >&2
  exit 1
fi

echo "==> Создаю окружение .venv..."
[ -d .venv ] || python3 -m venv .venv

echo "==> Ставлю зависимости (качается PyTorch, может занять несколько минут)..."
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

cat <<'DONE'

Готово. Установка завершена.

Дальше:
  1. В Zoom включите запись отдельных дорожек участников:
     Settings -> Recording -> Record a separate audio file of each participant
  2. Запустите обработку:
     ./run.sh
     (по умолчанию следит за ~/Documents/Zoom, транскрипты кладёт в ~/MeetingsKB)

При первом запуске модель GigaAM один раз скачается из интернета.
DONE
