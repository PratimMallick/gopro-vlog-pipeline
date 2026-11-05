# Local travel-vlog editor

A local pipeline that turns a folder of video clips into a YouTube-ready film: inventory, speech-to-text, an edit decision list, licensed music, burned-in subtitles, and a validated H.264 file. The clips can come from any camera or phone. GoPro chapter files are grouped with their proxy and thumbnail. Every other MP4, MOV, M4V, or MKV is its own clip.

Two open-weight models run on this machine, both through MLX on an Apple GPU. Whisper, set under `[transcription]`, transcribes the clips. A second model, set under `[plan]`, reads those transcripts and writes `editorial/edit_plan.json`. The default plan model is Qwen2.5 7B Instruct at 4-bit, about 4 GB, from `mlx-community` on Hugging Face. The download is free. The project has no API key and does not call a paid chat service.

The render does not call the model again. It checks the plan against the real files and plays that cut. Change the JSON and run the render again when you want a different film. `stage_e_plan.py` leaves an existing plan in place unless you pass `--force`.

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
../.venv/bin/python stage_e_plan.py    # local model writes editorial/edit_plan.json
../.venv/bin/python stage_d_edl.py
../.venv/bin/python stage_g_render.py
../.venv/bin/python stage_i_validate.py
```

`capture_date` limits speech and picture analysis to that day, even when the source folder holds a whole trip. A list named `capture_dates` allows more than one day in the same film. `top_level_only` ignores subfolders. Put place-name corrections and, if you want a route inset, the map stops in the edit plan (`translation_fixes`, `route_maps`). A plan with `"route_maps": {}` gets no map. A `place_maps` entry draws a north-up OpenStreetMap inset for two real coordinates, such as a town and an island. Google's map tiles are not downloaded.

FFmpeg has to be on your PATH. Text cards are drawn with Pillow, because many FFmpeg builds have no `drawtext`. The renderer does not upscale a clip to pretend it is 4K: output size is whatever you set in `[render]`, and proxy files are used only when the master is missing.

The finished films stay on the machine that recorded them. This repository is the editor.
