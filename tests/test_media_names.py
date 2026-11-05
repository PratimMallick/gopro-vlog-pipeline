import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import media_names  # noqa: E402


class MediaNamesTest(unittest.TestCase):
    def test_gopro_chapter_files_share_one_clip(self):
        master = media_names.classify_name("GH010206.MP4")
        proxy = media_names.classify_name("GL010206.LRV")
        self.assertEqual(master["clip_id"], "C0206_01")
        self.assertEqual(proxy["clip_id"], master["clip_id"])
        self.assertEqual(master["role"], "master")
        self.assertEqual(proxy["role"], "proxy")

    def test_any_video_file_is_its_own_clip(self):
        phone = media_names.classify_name("morning at the fort.mov")
        self.assertEqual(phone["kind"], "clip")
        self.assertEqual(phone["clip_id"], "V_morning_at_the_fort")
        self.assertEqual(phone["role"], "master")

    def test_notes_are_not_media(self):
        self.assertIsNone(media_names.classify_name("notes.txt"))


if __name__ == "__main__":
    unittest.main()
