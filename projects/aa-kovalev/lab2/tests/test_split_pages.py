"""Run: uv run --locked python -m unittest discover -s tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from build_index import split_pages
from common import get_tokenizer, settings


class SplitPagesTests(unittest.TestCase):
    def test_token_budget_and_unicode(self):
        samples = [
            '# Раздел\n\n' + 'Пример русского текста с Python.\n\n' * 60,
            '## Код\n\n```python\n' + 'print("Привет")\n' * 60 + '```',
            'ОченьДлинноеСловоБезПробелов' * 60,
            '🙂漢字 кириллица ' * 60,
            '',
        ]
        pages = [dict(text=text, section='Работа с Python', page=i, source_id='test')
                 for i, text in enumerate(samples)]
        for mode in ('markdown', 'window'):
            for headings in (True, False):
                for size in (100, 256, 512):
                    with self.subTest(mode=mode, headings=headings, size=size):
                        config = settings()
                        config.update(splitter=mode, include_headings=headings,
                                      chunk_size=size, overlap=20)
                        tokenizer = get_tokenizer(config)
                        chunks = split_pages(pages, config)
                        self.assertTrue(chunks)
                        self.assertEqual({c['page'] for c in chunks}, {0, 1, 2, 3})
                        self.assertEqual([c['id'] for c in chunks], list(range(len(chunks))))
                        for chunk in chunks:
                            self.assertEqual(chunk['tokens'], len(tokenizer.encode(chunk['text'])))
                            self.assertLessEqual(chunk['tokens'], size)
                            self.assertNotIn('\ufffd', chunk['text'])

    def test_incompatible_tokenizer(self):
        config = settings()
        config['tokenizer'] = 'cl100k_base'
        with self.assertRaises(ValueError):
            get_tokenizer(config)

    def test_heading_exhausts_budget(self):
        config = settings()
        with self.assertRaises(ValueError):
            split_pages([dict(text='Текст', section='Заголовок ' * 1000)], config)


if __name__ == '__main__':
    unittest.main()
