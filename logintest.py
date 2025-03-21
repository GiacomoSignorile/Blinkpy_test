from blinkpy.blinkpy import Blink
from aiohttp import ClientSession
import asyncio

async def test_login():
    async with ClientSession() as session:
        blink = Blink(session=session)
        await blink.start()
        

asyncio.run(test_login())
