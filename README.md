# GoPro travel-vlog editor

A local pipeline that turns a folder of GoPro clips into a YouTube-ready film: inventory, speech-to-text, an edit decision list, licensed music, burned-in subtitles, and a validated H.264 file.

The program does not decide the story. A person writes `editorial/edit_plan.json`. The pipeline checks that plan against the real files and renders it. The only model is speech recognition, set in `config.toml`. The default is Whisper large-v3 on an Apple GPU through MLX.

Footage, transcripts, the edit plan, and the finished video stay out of git. See [OPEN_SOURCE.md](OPEN_SOURCE.md) for how to run an episode and what the license requires for the music.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.example.toml config.toml
```
