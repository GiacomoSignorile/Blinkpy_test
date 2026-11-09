"""Contrast regression for shared theme tokens (no third-party dependencies)."""
import re
import unittest
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / 'blinkwatch/liquid-glass.css'


def contrast(a, b):
    def lum(color):
        values = [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in values]
        return sum(v * w for v, w in zip(linear, (.2126, .7152, .0722)))
    x, y = sorted((lum(a), lum(b)))
    return (y + .05) / (x + .05)


class ContrastTests(unittest.TestCase):
    def test_theme_text_and_selected_controls(self):
        text = CSS.read_text()
        for theme in ('light', 'dark'):
            block = re.search(r':root\[data-theme="' + theme + r'"\]\s*\{([^}]+)', text).group(1)
            tokens = dict(re.findall(r'--([\w-]+):\s*(#[0-9a-fA-F]{6})', block))
            with self.subTest(theme=theme):
                self.assertIn('on-accent', tokens, 'Selected controls need explicit contrast-safe text')
                for fg, bg in [('ink', 'surface'), ('muted', 'surface'), ('accent', 'accent-soft'), ('on-accent', 'accent')]:
                    self.assertGreaterEqual(contrast(tokens[fg], tokens[bg]), 4.5, (theme, fg, bg))


if __name__ == '__main__':
    unittest.main()
