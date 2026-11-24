"""Small semantic checks; no model downloads. Run: python -m unittest test_pilot.py."""
import tempfile
import unittest
from pathlib import Path
import torch
from pilot import build_records, recall_at_k


class RetrievalTests(unittest.TestCase):
    def test_multiple_captions_map_to_one_image(self):
        scores = torch.tensor([[.1, .9, .2], [.8, .1, .2], [.2, .1, .9], [.1, .8, .2]])
        self.assertEqual(recall_at_k(scores, torch.tensor([1, 0, 2, 1]))['R@1'], 1.)
        self.assertEqual(recall_at_k(scores, torch.tensor([0, 1, 0, 0]))['R@1'], 0.)
        self.assertEqual(recall_at_k(scores, torch.tensor([0, 1, 0, 0]))['R@5'], 1.)

    def test_split_excludes_official_test_and_keeps_captions_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cub = root / 'CUB_200_2011'; cub.mkdir()
            captions = root / 'birds/text/001.Bird'; captions.mkdir(parents=True)
            (cub / 'images.txt').write_text(''.join(f'{i} 001.Bird/bird_{i}.jpg\n' for i in range(1, 12)))
            (cub / 'train_test_split.txt').write_text(''.join(f'{i} {int(i < 11)}\n' for i in range(1, 12)))
            for i in range(1, 12): (captions / f'bird_{i}.txt').write_text('a red bird\na bird with red wings\n')
            rows = build_records(root, 42)
            self.assertEqual(rows, build_records(root, 42))
            train = {r['image_id'] for r in rows['train']}
            val = {r['image_id'] for r in rows['val']}
            self.assertFalse(train & val)
            self.assertEqual(len(train), 9)
            self.assertEqual(len(val), 1)
            self.assertNotIn('11', train | val)
            self.assertTrue(all(len(r['captions']) == 2 for r in rows['train'] + rows['val']))


if __name__ == '__main__':
    unittest.main()
