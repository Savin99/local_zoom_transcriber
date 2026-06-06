#!/usr/bin/env bash
#
# Запуск watcher: следит за папкой Zoom и складывает транскрипты в папку для Cowork.
#
# Использование:
#   ./run.sh                          # ~/Documents/Zoom  ->  ~/MeetingsKB
#   ./run.sh /путь/к/Zoom /путь/к/KB  # свои пути
#
set -euo pipefail
cd "$(dirname "$0")"

ZOOM_ROOT="${1:-$HOME/Documents/Zoom}"
COWORK_DIR="${2:-$HOME/MeetingsKB}"

if [ ! -d .venv ]; then
  echo "ОШИБКА: окружение не найдено. Сначала запустите ./install.sh" >&2
  exit 1
fi

mkdir -p "$COWORK_DIR"
echo "Слежу за папкой Zoom:  $ZOOM_ROOT"
echo "Транскрипты кладу в:   $COWORK_DIR"
echo "Остановить: Ctrl+C"
exec .venv/bin/python watch_zoom_transcripts.py "$ZOOM_ROOT" "$COWORK_DIR"
