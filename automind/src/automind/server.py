import asyncio

import aiohttp


async def wait_until_server_ready(url, timeout=5):
    start = asyncio.get_event_loop().time()
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200 or resp.status == 404:
                        return True
        except Exception:
            pass

        await asyncio.sleep(0.2)

        if asyncio.get_event_loop().time() - start > timeout:
            raise RuntimeError("Server not responding")
