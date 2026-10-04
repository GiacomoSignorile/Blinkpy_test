import asyncio
import logging

from aiohttp import ClientSession
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
    """Ask the sync module to upload the clip to the cloud, then fetch it."""
    await item.prepare_download(blink)
    if not await item.download_video(blink, str(path)):
        raise IOError(f"download failed for clip {item.id}")
