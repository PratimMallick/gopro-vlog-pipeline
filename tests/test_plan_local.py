import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import plan_local  # noqa: E402


CFG = {"episode": {"title": "MY TRIP", "date_label": "1 JANUARY 2026",
                   "min_duration_s": 10, "max_duration_s": 60}}
CLIPS = [
    {"clip_id": "C0001", "duration_s": 30, "source_kind": "master", "episode_date_ok": True,
     "has_audio": True, "capture_local": "2026-01-01 09:00:00"},
    {"clip_id": "C0002", "duration_s": 20, "source_kind": "master", "episode_date_ok": True,
     "has_audio": False, "capture_local": "2026-01-01 09:10:00"},
]
TRANSCRIPTS = {"C0001": {"language": "en", "segments": [
    {"start": 1.2, "end": 5.0, "text": "We reached the old fort this morning.", "keep": True, "avg_logprob": -0.2},
    {"start": 6.0, "end": 7.0, "text": "subscribe to my channel", "keep": False, "avg_logprob": -0.1},
]}}
MUSIC = {"Easy_Lemon.mp3": {"duration_s": 126}}


def plan():
    return {
        "chapters": [
            {"id": "arrival", "music": {"track": "Easy_Lemon.mp3"}, "entries": [
                {"clip": "C0001", "in": 1.0, "out": 12.0, "mode": "dialogue",
                 "title": {"text": "MY TRIP"},
                 "location": {"text": "The old fort"}},
                {"clip": "C0002", "in": 0.5, "out": 8.0, "mode": "music"},
            ]},
        ]
    }


class PlanLocalTest(unittest.TestCase):
    def test_brief_keeps_kept_speech_and_drops_rejected_lines(self):
        brief = plan_local.build_brief(CFG, CLIPS, TRANSCRIPTS, MUSIC)
        self.assertIn("C0001@1.20", brief)
        self.assertNotIn("subscribe", brief)
        self.assertIn("Easy_Lemon.mp3", brief)

    def test_parse_accepts_fenced_json_with_a_trailing_comma(self):
        got = plan_local.parse_json_object("```json\n{\"chapters\": [1,],}\n```")
        self.assertEqual(got, {"chapters": [1]})

    def test_grounded_plan_passes_and_gains_a_credit(self):
        ready, errors, notes = plan_local.prepare_plan(plan(), CLIPS, TRANSCRIPTS, MUSIC, CFG)
        self.assertEqual(errors, [])
        self.assertIn("Kevin MacLeod", ready["end_credits"][0]["text"])
        self.assertEqual(ready["route_maps"], {})
        self.assertTrue(notes)

    def test_unknown_clip_and_invented_card_are_rejected(self):
        bad = plan()
        bad["chapters"][0]["entries"][0]["clip"] = "C9999"
        bad["chapters"][0]["entries"][0]["info"] = {"text": "The statue is 123 feet tall."}
        _, errors, _ = plan_local.prepare_plan(bad, CLIPS, TRANSCRIPTS, MUSIC, CFG)
        self.assertTrue(any("C9999" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
