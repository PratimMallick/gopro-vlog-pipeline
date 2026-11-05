# GoPro travel-vlog editor

A local pipeline that turns a folder of GoPro clips into a YouTube-ready film: inventory, speech-to-text, an edit decision list, licensed music, burned-in subtitles, and a validated H.264 file. It was built for one trip and then opened up so a second day, or someone else's footage, can use the same code.

The program does not decide the story. A person writes `editorial/edit_plan.json` (which clip, from when to when, and which cards). The pipeline checks that plan against the real files and renders it. Nothing here calls ChatGPT, Claude, or any other chat model. The only model is speech recognition, and you choose it in `config.toml` under `[transcription]`. The default is Whisper large-v3 running on an Apple GPU through MLX. Swap that setting for another Whisper build if you want. There is no switch that hands the cut to a chatbot.

## What stays private

`.gitignore` keeps footage, transcripts, the edit plan, contact sheets, and the finished video out of git. Those files contain people's faces and voices. Publish the code. Keep the episode folder on your own disk.

Music in `music/` is Kevin MacLeod, CC BY 4.0 (incompetech.com). The files themselves are gitignored; `music/LICENSES.json` has the credit line you must put in the video description. Do not drop in commercial songs.

## Run an episode

```bash
python3 -m venv .venv
.venv/bin/pip install numpy opencv-python-headless pillow mlx-whisper soundfile
cp config.example.toml config.toml   # then edit paths, title, and capture_date
```

One episode folder holds `config.toml`, `editorial/edit_plan.json`, and the outputs. Point the pipeline at it:

```bash
export VLOG_PROJECT=/path/to/your/episode
cd pipeline
../.venv/bin/python stage_a_inventory.py
../.venv/bin/python stage_c_speech.py
../.venv/bin/python stage_f_music.py
# write editorial/edit_plan.json, then:
../.venv/bin/python stage_d_edl.py
../.venv/bin/python stage_g_render.py
../.venv/bin/python stage_i_validate.py
```

`capture_date` limits speech and picture analysis to that day, even when the source folder holds a whole trip. A list named `capture_dates` allows more than one day in the same film. `top_level_only` ignores subfolders. Put place-name corrections and, if you want a route inset, the map stops in the edit plan (`translation_fixes`, `route_maps`). A plan with `"route_maps": {}` gets no map. A `place_maps` entry draws a north-up OpenStreetMap inset for two real coordinates, such as a town and an island. Google's map tiles are not downloaded.

FFmpeg has to be on your PATH. Text cards are drawn with Pillow, because many FFmpeg builds have no `drawtext`. The renderer does not upscale a clip to pretend it is 4K: output size is whatever you set in `[render]`, and proxy files are used only when the master is missing.

The finished films stay on the machine that recorded them. This repository is the editor.
