"""Shared UI contract; dependency-free: python -m unittest discover -s tests -p test_liquid_glass.py."""
import ast
from html.parser import HTMLParser
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAGES = {"index": "/", "cameras": "/cameras", "label": "/label", "power": "/power"}


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.elements = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class LiquidGlassTests(unittest.TestCase):
    def test_all_pages_load_versioned_shared_styles_after_local_styles(self):
        for page in PAGES:
            with self.subTest(page=page):
                text = (ROOT / "blinkwatch" / (page + ".html")).read_text()
                links = [a for t, a in Elements(text).elements if t == "link" and a.get("rel") == "stylesheet"]
                self.assertTrue(any(a.get("href", "").startswith("/assets/liquid-glass.css?v=") for a in links))
                self.assertGreater(text.index('/assets/liquid-glass.css'), text.rindex('</style>'))

    def test_global_navigation_theme_and_italian_language_preserved(self):
        for page, route in PAGES.items():
            with self.subTest(page=page):
                text = (ROOT / "blinkwatch" / (page + ".html")).read_text()
                elements = Elements(text).elements
                self.assertIn(('html', {'lang': 'it'}), elements)
                self.assertTrue(any(a.get('name') == 'theme-color' for t, a in elements if t == 'meta'))
                self.assertTrue(any(a.get('id') == 'theme' for t, a in elements))
                nav_links = [a for t, a in elements if t == 'a' and 'nav' in a.get('class', '').split()]
                self.assertEqual({a['href'] for a in nav_links}, set(PAGES.values()))
                self.assertEqual([a['href'] for a in nav_links if a.get('aria-current') == 'page'], [route])
                self.assertIn("'auto'", text.replace('"', "'"))
                self.assertIn("store.get('theme'", text.replace('"', "'"))

    def test_stylesheet_route_serves_real_css(self):
        tree = ast.parse((ROOT / 'blinkwatch/web.py').read_text())
        matches = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == 'get'
            and d.args and isinstance(d.args[0], ast.Constant) and d.args[0].value == '/assets/liquid-glass.css'
            for d in n.decorator_list)]
        self.assertEqual(len(matches), 1, 'Explicit GET route must serve the shared CSS')
        # Exercise the actual route body without importing optional application dependencies.
        def file_response(path, media_type=None, **kwargs):
            return Path(path).read_text(), media_type
        node = matches[0]
        node.decorator_list = []
        namespace = {'Path': Path, '__file__': str(ROOT / 'blinkwatch/web.py'), 'FileResponse': file_response}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<css-route>', 'exec'), namespace)
        content, media_type = namespace[node.name]()
        self.assertEqual(media_type, 'text/css')
        self.assertIn('.bottom-nav', content)


if __name__ == '__main__':
    unittest.main()
