# Local Zoom Transcriber

CLI для локальной транскрибации записей Zoom через GigaAM. Работает полностью
локально (без облака и API-ключей), распознаёт русскую речь с пунктуацией,
проставляет спикеров по отдельным дорожкам Zoom и складывает готовые
транскрипты в папку для Cowork.

## Требования

- Python 3.10+
- `ffmpeg`/`ffprobe` (на macOS: `brew install ffmpeg`)

## Установка

```bash
git clone https://github.com/Savin99/local_zoom_transcriber.git
cd local_zoom_transcriber
./install.sh
```

`install.sh` проверит Python и ffmpeg, создаст окружение и поставит зависимости.
После установки запуск обработки — одной командой `./run.sh`.

Чтобы watcher поднимался сам при входе в систему и перезапускался при сбоях:

```bash
./autostart.sh
```

Убрать автозапуск — `./autostart.sh --uninstall`.

Ручные шаги и все опции — в [docs/SETUP.md](docs/SETUP.md).

## Автоматический workflow для Cowork

Если Zoom складывает записи в общую папку, а Cowork должен читать готовые
транскрипты из другой папки, запускай watcher:

```bash
.venv/bin/python watch_zoom_transcripts.py "/path/to/Zoom" "$HOME/MeetingsKB"
```

Что он делает:

1. проверяет общую Zoom-папку и ищет внутри папки отдельных встреч;
2. ждёт, пока `.m4a`-файлы записи перестанут меняться;
3. запускает `transcribe_zoom_gigaam.py` для новой/изменённой встречи;
4. копирует `transcripts/call_*.md` в папку Cowork отдельными файлами:
   `имя_zoom_папки_call_01_transcript.md`, `..._call_02_transcript.md`.

Если `call_*.md` нет, watcher использует `transcripts/all_calls.md` как fallback.

Проверить, что будет обработано, без запуска ASR:

```bash
.venv/bin/python watch_zoom_transcripts.py "/path/to/Zoom" "$HOME/MeetingsKB" --once --plan-only
```

Обработать всё готовое один раз и выйти:

```bash
.venv/bin/python watch_zoom_transcripts.py "/path/to/Zoom" "$HOME/MeetingsKB" --once
```

Состояние хранится в `$HOME/MeetingsKB/.zoom_transcriber_state.json`, поэтому
уже обработанные встречи не гоняются повторно. Если надо пересобрать всё:

```bash
.venv/bin/python watch_zoom_transcripts.py "/path/to/Zoom" "$HOME/MeetingsKB" --once --force
```

Подробная инструкция для разворачивания на другой машине: [docs/SETUP.md](docs/SETUP.md).

Промпт и протокол проверки отчётов в Cowork:

- [docs/COWORK_REPORT_PROMPT.md](docs/COWORK_REPORT_PROMPT.md)
- [docs/COWORK_REPORT_TEST.md](docs/COWORK_REPORT_TEST.md)

## Быстрый запуск

```bash
.venv/bin/python transcribe_zoom_gigaam.py "path/to/zoom-folder"
```

По умолчанию используется `v3_e2e_rnnt`: это лучшая локально доступная GigaAM-модель для созвонов, потому что она добавляет пунктуацию и нормализацию текста.

> При первом запуске GigaAM один раз скачает веса модели в `~/.cache/gigaam/`
> (нужен интернет). Дальше распознавание работает офлайн.

Результат появится в папке записи:

```text
transcripts/
  all_calls.md
  call_01.md
  call_01.txt
  call_02.md
  call_02.txt
  manifest.json
```

Если запуск прервался, можно запустить ту же команду еще раз: скрипт прочитает `transcripts/manifest.json` и пропустит уже распознанные чанки.

Если в Zoom-папке есть `Audio Record/` с отдельными дорожками участников, скрипт автоматически добавит speaker labels (чтобы дорожки появились, включи в Zoom `Settings -> Recording -> Record a separate audio file of each participant`):

```text
[00:01:01 - 00:01:22] **Ivan Petrov / Sergey:** текст...
```

Пересчитать только диаризацию без повторной транскрибации:

```bash
.venv/bin/python transcribe_zoom_gigaam.py "path/to/zoom-folder" --diarize-only
```

## Как определяются созвоны

Zoom сам сохраняет части записи как верхнеуровневые `audio*.m4a`. Скрипт считает каждый такой файл отдельным call.

Скрипт не пытается угадывать встречи внутри одного большого файла по тишине. Тишина используется только технически: чтобы нарезать каждый Zoom audio-файл на короткие ASR-чанки до 25 секунд для GigaAM, а потом собрать обратно в тот же call.

## Проверка без полного распознавания

Построить только план Zoom-файлов и ASR-чанков:

```bash
.venv/bin/python transcribe_zoom_gigaam.py "path/to/zoom-folder" --dry-run
```

Распознать только первый кусок для smoke-test:

```bash
.venv/bin/python transcribe_zoom_gigaam.py "path/to/zoom-folder" --max-chunks 1 --batch-size 1
```

## Полезные настройки

- `--device cpu|mps|cuda|auto` - устройство для модели; по умолчанию `cpu`, потому что MPS подвисал на части чанков.
- `--batch-size 1` - безопасный режим для Mac/MPS; можно увеличить на CUDA или если MPS стабильно работает.
- `--save-every-chunks 5` - как часто сохранять промежуточный результат.
- `--diarize-only` - проставить спикеров по `Audio Record/` без повторного ASR.
- `--no-diarization` - оставить только transcript без speaker labels.
- `--vad-threshold-db auto` - авто-порог речи; можно поставить число, например `-42`.
- `--chunk-sec 22` - максимальная длина куска для GigaAM, должна оставаться меньше 25 секунд.

## Тесты

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Сторонние компоненты

Распознавание речи использует [GigaAM](https://github.com/salute-developers/GigaAM)
(© GigaChat Team) под лицензией MIT — см. [`GigaAM/LICENSE`](GigaAM/LICENSE).
Папка `GigaAM/` включена в проект как вендоренная зависимость.
