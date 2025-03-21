#!/usr/bin/env python3.9
import asyncio
from datetime import datetime, timedelta
from blinkpy.blinkpy import Blink, BlinkSyncModule
from aiohttp import ClientSession
import os
import logging


DOWNLOAD_PATH = "/home/giacomo/Documenti/Recordings"  # Change this to your desired save location
SYNC_MODULE_NAME = "Casa online" 

# Configure logging to write to a file
logging.basicConfig(level=logging.INFO, filename='blinkpy.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def start(session):
    # Use a context manager for the aiohttp session    
    blink = Blink(session=session)
    await blink.start()
    return blink

async def main():
    # Ensure the download directory exists
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)

    session = ClientSession()
    try:
        blink = await start(session)
        await blink.refresh()
        logger.info("Blink system refreshed")

        # Get the Sync Module
        my_sync: BlinkSyncModule = blink.sync.get(SYNC_MODULE_NAME)
        if not my_sync:
            logger.error(f"Sync Module '{SYNC_MODULE_NAME}' not found! Exiting.")
            return

        await my_sync.refresh()
        logger.info("Sync Module refreshed")

        if my_sync.local_storage and my_sync.local_storage_manifest_ready:
            manifest = my_sync._local_storage["manifest"]
            logger.info("Manifest ready")

            for item in reversed(manifest):
                current_date = datetime.now(item.created_at.tzinfo)
                time_difference = current_date - item.created_at

                if time_difference < timedelta(hours=1):  # Only download recent clips
                    try:
                        file_name = f"{DOWNLOAD_PATH}/video_{item.id}_{item.created_at.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
                        # Use download_video instead of download_clip
                        await item.download_video(blink, file_name)
                        logger.info(f"✅ Downloaded: {file_name}")

                        # Verify file creation
                        if os.path.exists(file_name):
                            logger.info(f"File successfully created: {file_name}")
                        else:
                            logger.error(f"File not found after download: {file_name}")

                        await asyncio.sleep(2)  # Avoid overwhelming the API
                    except Exception as e:
                        logger.error(f"❌ Error downloading video {item.id}: {e}")
        else:
            logger.warning("⚠️ Manifest not ready!")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        await session.close()
        logger.info("Session closed")

asyncio.run(main())