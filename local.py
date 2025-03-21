import asyncio
import time
import requests
from datetime import datetime, timedelta
from sortedcontainers import SortedSet
from blinkpy.helpers.util import json_load
from blinkpy.blinkpy import Blink, BlinkSyncModule
from blinkpy.auth import Auth
from aiohttp import ClientSession
import os
import logging

# Use the same endpoint that appears during login.
BASE_URL = "https://rest-e006.immedia-semi.com"

def request_manifest(account_id, network_id, sync_id, headers):
    url = f"{BASE_URL}/api/v1/accounts/{account_id}/networks/{network_id}/sync_modules/{sync_id}/local_storage/manifest/request"
    print("Request URL:", url)
    response = requests.post(url, headers=headers)
    print("Response Status Code:", response.status_code)
    print("Response Content:", response.content)
    if response.status_code == 200:
        data = response.json()
        return data.get("id")
    return None

def get_manifest(account_id, network_id, sync_id, manifest_request_id, headers):
    url = f"{BASE_URL}/api/v1/accounts/{account_id}/networks/{network_id}/sync_modules/{sync_id}/local_storage/manifest/request/{manifest_request_id}"
    # Allow time for the manifest to be generated
    time.sleep(5)
    response = requests.get(url, headers=headers)
    print("Manifest retrieval status:", response.status_code)
    if response.status_code == 200:
        data = response.json()
        return data.get("manifest_id"), data.get("clips", [])
    return None, []

def request_clip_upload(account_id, network_id, sync_id, manifest_id, clip_id, headers):
    url = f"{BASE_URL}/api/v1/accounts/{account_id}/networks/{network_id}/sync_modules/{sync_id}/local_storage/manifest/{manifest_id}/clip/request/{clip_id}"
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def download_clip(clip_url, output_filename):
    response = requests.get(clip_url, stream=True)
    if response.status_code == 200:
        with open(output_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Downloaded clip to", output_filename)
    else:
        print("Error downloading clip:", response.status_code)

async def start(session):
    # Use a context manager for the aiohttp session    
    blink = Blink(session=session)
    await blink.start()
    return blink

DOWNLOAD_PATH = "/home/giacomosig/blinkpy/recordings"  # Change this to your desired save location
SYNC_MODULE_NAME = "Casa online" 

# Configure logging to write to a file
logging.basicConfig(level=logging.INFO, filename='blinkpy.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                        await item.download_clip(blink, file_name)
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