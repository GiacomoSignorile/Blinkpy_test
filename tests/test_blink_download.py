import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from blinkwatch.blink_client import download


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_fetch_prepares_a_fresh_upload_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.mp4'
            calls = []

            async def prepare(_blink):
                calls.append('prepare')
                return {'id': 1}

            async def fetch(_blink, filename, **_kwargs):
                calls.append('fetch')
                if calls.count('fetch') == 1:
                    return False
                Path(filename).write_bytes(b'video fixture')
                return True

            item = SimpleNamespace(id=42, prepare_download=prepare, download_video=fetch)
            with patch('blinkwatch.blink_client.asyncio.sleep', new_callable=AsyncMock):
                await download(item, object(), path)
            self.assertEqual(calls, ['prepare', 'fetch', 'prepare', 'fetch'])
            self.assertEqual(path.read_bytes(), b'video fixture')


    async def test_exhausted_fetch_is_bounded_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.mp4'
            path.write_bytes(b'original')

            async def fetch(_blink, filename, **_kwargs):
                Path(filename).write_bytes(b'partial')
                return False

            item = SimpleNamespace(id=42, prepare_download=AsyncMock(),
                                   download_video=AsyncMock(side_effect=fetch))
            with patch('blinkwatch.blink_client.asyncio.sleep', new_callable=AsyncMock):
                with self.assertRaisesRegex(OSError, 'stage=cloud-fetch after 2'):
                    await download(item, object(), path)
            self.assertEqual(item.prepare_download.await_count, 2)
            self.assertEqual(item.download_video.await_count, 2)
            self.assertEqual(path.read_bytes(), b'original')
            self.assertFalse(path.with_name(path.name + '.part').exists())

    async def test_empty_cloud_response_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.mp4'

            async def fetch(_blink, filename, **_kwargs):
                Path(filename).write_bytes(b'')
                return True

            item = SimpleNamespace(id=42, prepare_download=AsyncMock(),
                                   download_video=AsyncMock(side_effect=fetch))
            with patch('blinkwatch.blink_client.asyncio.sleep', new_callable=AsyncMock):
                with self.assertRaises(OSError):
                    await download(item, object(), path)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name(path.name + '.part').exists())

    async def test_prepare_timeout_retries_before_fetching(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.mp4'

            async def fetch(_blink, filename, **_kwargs):
                Path(filename).write_bytes(b'video fixture')
                return True

            item = SimpleNamespace(id=42,
                                   prepare_download=AsyncMock(side_effect=[TimeoutError(), {'id': 1}]),
                                   download_video=AsyncMock(side_effect=fetch))
            with patch('blinkwatch.blink_client.asyncio.sleep', new_callable=AsyncMock):
                await download(item, object(), path)
            self.assertEqual(item.prepare_download.await_count, 2)
            self.assertEqual(item.download_video.await_count, 1)
            self.assertEqual(path.read_bytes(), b'video fixture')


if __name__ == '__main__':
    unittest.main()
