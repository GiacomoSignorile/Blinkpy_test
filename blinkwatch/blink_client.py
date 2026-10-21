import asyncio
import logging
from pathlib import Path

from aiohttp import ClientError, ClientSession
from blinkpy.auth import Auth, BlinkTwoFARequiredError
from blinkpy.blinkpy import Blink
from blinkpy.helpers.util import json_load

from . import config

log = logging.getLogger(__name__)


async def _wait_for_2fa_code(timeout=600):
    config.TWOFA_FILE.unlink(missing_ok=True)
    log.warning("2FA richiesto: scrivi il codice ricevuto in %s (attendo %ds)", config.TWOFA_FILE, timeout)
    for _ in range(timeout // 2):
        if config.TWOFA_FILE.exists() and config.TWOFA_FILE.read_text().strip():
            code = config.TWOFA_FILE.read_text().strip()
            config.TWOFA_FILE.unlink()
            return code
        await asyncio.sleep(2)
    raise TimeoutError("2FA code not provided")


async def connect(session: ClientSession) -> Blink:
    """Log in with saved credentials, falling back to .env user/password (+ 2FA code file)."""
    blink = Blink(session=session)
    if config.CREDENTIALS_FILE.exists():
        creds = await json_load(str(config.CREDENTIALS_FILE))
    elif config.BLINK_USERNAME and config.BLINK_PASSWORD:
        creds = {"username": config.BLINK_USERNAME, "password": config.BLINK_PASSWORD}
    else:
        raise SystemExit("Mancano BLINK_USERNAME/BLINK_PASSWORD in .env")
    blink.auth = Auth(creds, no_prompt=True, session=session)
    try:
        if await blink.start() is False:
            raise SystemExit("Login Blink rifiutato (controlla email/password in .env)")
    except BlinkTwoFARequiredError:
        if not await blink.send_2fa_code(await _wait_for_2fa_code()):
            raise SystemExit("Login 2FA fallito")
    await blink.save(str(config.CREDENTIALS_FILE))
    config.CREDENTIALS_FILE.chmod(0o600)
    return blink


async def download(item, blink, path):
    """Retry the complete sync-module upload + cloud fetch, publishing atomically."""
    path = Path(path)
    partial = path.with_name(path.name + ".part")
    for attempt in range(1, 3):
        stage = "prepare-upload"
        try:
            await item.prepare_download(blink)
            stage = "cloud-fetch"
            # Split blinkpy's four fetch attempts across two fresh upload requests.
            if not await item.download_video(blink, str(partial), max_retries=2):
                raise IOError("Blink cloud fetch did not return HTTP 200")
            if not partial.is_file() or partial.stat().st_size == 0:
                raise IOError("Blink returned an empty video")
            partial.replace(path)
            log.info("download complete clip=%s attempt=%d", item.id, attempt)
            return
        except (OSError, ClientError, asyncio.TimeoutError) as exc:
            log.warning("download attempt failed clip=%s stage=%s attempt=%d/2 error=%s",
                        item.id, stage, attempt, type(exc).__name__)
            if attempt == 2:
                raise IOError(f"download failed for clip {item.id} at stage={stage} after 2 upload attempts") from exc
        finally:
            partial.unlink(missing_ok=True)
        await asyncio.sleep(3)
