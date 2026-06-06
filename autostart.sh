#!/usr/bin/env bash
#
# Автозапуск watcher через launchd (macOS LaunchAgent).
# Watcher стартует сам при входе в систему и перезапускается при сбоях.
#
# Использование:
#   ./autostart.sh                          # ~/Documents/Zoom -> ~/MeetingsKB
#   ./autostart.sh /путь/к/Zoom /путь/к/KB  # свои пути
#   ./autostart.sh --dry-run [пути...]      # показать plist, ничего не ставить
#   ./autostart.sh --uninstall              # убрать автозапуск
#
set -euo pipefail

LABEL="com.local-zoom-transcriber.watcher"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/local-zoom-transcriber.log"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
UID_NUM="$(id -u)"

MODE="install"
case "${1:-}" in
  --uninstall) MODE="uninstall"; shift ;;
  --dry-run)   MODE="dry-run";   shift ;;
esac

if [ "$MODE" = "uninstall" ]; then
  launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null \
    || launchctl unload -w "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Автозапуск убран."
  exit 0
fi

ZOOM_ROOT="${1:-$HOME/Documents/Zoom}"
COWORK_DIR="${2:-$HOME/MeetingsKB}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
WATCHER="$PROJECT_DIR/watch_zoom_transcripts.py"

if [ ! -x "$PYTHON" ]; then
  echo "ОШИБКА: $PYTHON не найден. Сначала запустите ./install.sh" >&2
  exit 1
fi

# PATH для launchd: добавляем папку с ffprobe (Homebrew), иначе он не найдётся
if command -v ffprobe >/dev/null 2>&1; then
  FF_DIR="$(dirname "$(command -v ffprobe)")"
else
  FF_DIR="/opt/homebrew/bin"
fi

if [ "$MODE" = "dry-run" ]; then
  OUT="$(mktemp)"
else
  mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"
  OUT="$PLIST"
fi

cat > "$OUT" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON</string>
        <string>$WATCHER</string>
        <string>$ZOOM_ROOT</string>
        <string>$COWORK_DIR</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$FF_DIR:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG</string>
    <key>StandardErrorPath</key>
    <string>$LOG</string>
</dict>
</plist>
EOF

plutil -lint "$OUT" >/dev/null

if [ "$MODE" = "dry-run" ]; then
  echo "# Превью LaunchAgent (ничего не установлено):"
  echo "#   Zoom:        $ZOOM_ROOT"
  echo "#   Транскрипты: $COWORK_DIR"
  echo "#   Лог:         $LOG"
  echo "#   Plist:       $PLIST"
  echo
  cat "$OUT"
  rm -f "$OUT"
  exit 0
fi

# (пере)загружаем агент — современный API с откатом на legacy для старых macOS
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null \
  || launchctl unload "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST" 2>/dev/null \
  || launchctl load -w "$PLIST"
launchctl enable "gui/$UID_NUM/$LABEL" 2>/dev/null || true

echo "Автозапуск включён."
echo "  Zoom:        $ZOOM_ROOT"
echo "  Транскрипты: $COWORK_DIR"
echo "  Лог:         tail -f \"$LOG\""
echo
echo "Проверить статус:   launchctl list | grep local-zoom-transcriber"
echo "Убрать автозапуск:  ./autostart.sh --uninstall"
