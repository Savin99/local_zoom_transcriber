# Zoom -> GigaAM -> Cowork

Локальный пайплайн:

```text
Zoom recordings root -> transcribe_zoom_gigaam.py -> Cowork folder
```

`watch_zoom_transcripts.py` следит за общей папкой Zoom, находит новые папки
встреч, запускает GigaAM-транскрибацию и экспортирует итоговый Markdown в папку
Cowork. Экспорт идёт по `call_*.md`: один файл в Cowork соответствует одному
`Call` из Zoom-папки. Если `call_*.md` нет, используется `all_calls.md` как
fallback.

## Папки

Нужно указать два пути:

- `ZOOM_ROOT` - общая папка локальных записей Zoom.
- `COWORK_DIR` - папка, которую будет читать Cowork.

Обычно на macOS:

```bash
ZOOM_ROOT="$HOME/Documents/Zoom"
COWORK_DIR="$HOME/MeetingsKB"
```

Включить в Zoom (`Settings -> Recording`):

- **Record a separate audio file of each participant** - обязательно: именно из
  этих отдельных дорожек проставляются спикеры. Без неё транскрипт получится, но
  без подписей, кто говорит.
- Путь из **Store my recordings at** - это `ZOOM_ROOT`.

Ожидаемая структура внутри `ZOOM_ROOT`:

```text
2026-01-15 10.00.00 Команда Созвон
  audio1075103314.m4a
  Audio Record/
    audioSpeaker11075103314.m4a
    audioSpeaker21075103314.m4a
```

В Cowork подключать нужно `COWORK_DIR`, не `ZOOM_ROOT`.

## Зависимости

Склонировать репозиторий и перейти в него:

```bash
git clone https://github.com/Savin99/local_zoom_transcriber.git
cd local_zoom_transcriber
```

Самый простой путь — установить всё одной командой:

```bash
./install.sh
```

`install.sh` проверит Python и ffmpeg, создаст `.venv` и поставит зависимости.
Ниже — те же шаги вручную, если нужно настроить окружение самостоятельно.

Проверить `ffprobe`:

```bash
ffprobe -version
```

Если команды нет:

```bash
brew install ffmpeg
```

Создать окружение и поставить зависимости:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` ставит PyTorch (`torch`, `torchaudio`) с PyPI и локальный
GigaAM (`-e ./GigaAM`).

Если нужна особая сборка PyTorch (CUDA, Apple MPS или специфичный CPU), сначала
поставьте её по инструкции с <https://pytorch.org/get-started/locally/>, затем
повторите `pip install -r requirements.txt` - уже установленный torch будет
засчитан.

При первом запуске транскрибации GigaAM один раз скачает веса модели в
`~/.cache/gigaam/` (нужен интернет). Последующие запуски работают офлайн.

## Проверка

Запустить тесты:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Ожидаемый итог:

```text
Ran 22 tests
OK
```

Создать папку Cowork:

```bash
mkdir -p "$COWORK_DIR"
```

Проверить, какие записи будут обработаны, без запуска ASR:

```bash
.venv/bin/python watch_zoom_transcripts.py "$ZOOM_ROOT" "$COWORK_DIR" --once --plan-only
```

Пример ожидаемой строки:

```text
2026-01-15 10.00.00 Команда Созвон: new or changed
```

## Разовый запуск

Обработать все готовые записи и выйти:

```bash
.venv/bin/python watch_zoom_transcripts.py "$ZOOM_ROOT" "$COWORK_DIR" --once
```

После успешного запуска в `COWORK_DIR` появятся:

```text
2026-01-15_10.00.00_Команда_Созвон_call_01_transcript.md
2026-01-15_10.00.00_Команда_Созвон_call_02_transcript.md
.zoom_transcriber_state.json
```

State-файл нужен для идемпотентности: уже обработанные встречи не запускаются
повторно.

## Постоянный запуск

Проще всего — `./run.sh` (следит за `~/Documents/Zoom`, складывает в `~/MeetingsKB`;
можно передать свои пути: `./run.sh /путь/к/Zoom /путь/к/KB`).

Эквивалент вручную:

```bash
.venv/bin/python watch_zoom_transcripts.py "$ZOOM_ROOT" "$COWORK_DIR"
```

Фоновый запуск с логом:

```bash
nohup .venv/bin/python watch_zoom_transcripts.py "$ZOOM_ROOT" "$COWORK_DIR" \
  > "$HOME/zoom-transcriber-watch.log" 2>&1 &
```

Лог:

```bash
tail -f "$HOME/zoom-transcriber-watch.log"
```

Остановить watcher:

```bash
pkill -f watch_zoom_transcripts.py
```

По умолчанию watcher:

- проверяет `ZOOM_ROOT` раз в 60 секунд;
- ждёт 120 секунд после последнего изменения `.m4a`, чтобы не забрать файл во
  время конвертации Zoom.

## Автозапуск (launchd)

Чтобы watcher стартовал сам при входе в систему и перезапускался при сбоях,
поставьте LaunchAgent:

```bash
./autostart.sh                          # ~/Documents/Zoom -> ~/MeetingsKB
./autostart.sh /путь/к/Zoom /путь/к/KB  # свои пути
```

После этого терминал трогать не нужно — watcher работает в фоне. Если включён
автозапуск, отдельно `./run.sh` запускать не надо.

```bash
launchctl list | grep local-zoom-transcriber       # проверить, что запущен
tail -f ~/Library/Logs/local-zoom-transcriber.log   # лог
./autostart.sh --dry-run                            # показать plist, ничего не ставя
./autostart.sh --uninstall                          # убрать автозапуск
```

## Повторная обработка

Обычный повторный запуск безопасен:

```bash
.venv/bin/python watch_zoom_transcripts.py "$ZOOM_ROOT" "$COWORK_DIR" --once
```

Принудительно пересобрать найденные записи:

```bash
.venv/bin/python watch_zoom_transcripts.py "$ZOOM_ROOT" "$COWORK_DIR" --once --force
```

Полностью сбросить историю:

```bash
rm "$COWORK_DIR/.zoom_transcriber_state.json"
```

## Диагностика

`No Zoom recording folders with .m4a files found`

Проверить, что `ZOOM_ROOT` указывает на общую папку Zoom, а внутри папок встреч
есть верхнеуровневые `audio*.m4a`.

`waiting until stable`

Нормальное состояние: watcher ждёт завершения записи/конвертации.

Transcriber упал

Смотреть state:

```bash
cat "$COWORK_DIR/.zoom_transcriber_state.json"
```

Проверить конкретную встречу вручную:

```bash
.venv/bin/python transcribe_zoom_gigaam.py "/path/to/zoom/meeting"
```
