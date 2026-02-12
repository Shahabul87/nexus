"""
Capture screenshots of NEXUS demo from LOCAL Streamlit with actual file uploads.
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

SPACE_URL = "http://localhost:8502"
FRAMES_DIR = Path(__file__).parent / "frames"
ASSETS_DIR = Path(__file__).parent / "assets"

ANEMIA_IMAGE = str(ASSETS_DIR / "anemia" / "20200223_185602_palpebral.png")
JAUNDICE_IMAGE = str(ASSETS_DIR / "jaundice" / "sample_jaundice_1.jpg")
CRY_AUDIO = str(ASSETS_DIR / "cry" / "Real_Infantcry.wav")


async def wait_for_app(page, timeout=30000):
    await page.wait_for_selector("text=NEXUS", timeout=timeout)
    await asyncio.sleep(3)


async def select_tab(page, tab_name):
    await page.locator("label").filter(has_text=tab_name).click()
    await asyncio.sleep(2)


async def wait_for_no_running(page, max_wait=300):
    """Wait until the 'Running' indicator disappears."""
    for _ in range(max_wait):
        running = await page.locator("img[alt='Running...']").count()
        if running == 0:
            return True
        await asyncio.sleep(1)
    return False


async def capture_anemia(page):
    print("Capturing Anemia Screening with results...")
    await select_tab(page, "Maternal Anemia Screening")
    await asyncio.sleep(1)

    # Upload
    async with page.expect_file_chooser() as fc_info:
        await page.get_by_test_id("stBaseButton-secondary").click()
    file_chooser = await fc_info.value
    await file_chooser.set_files(ANEMIA_IMAGE)

    print("  Waiting for analysis...")
    await asyncio.sleep(5)
    await wait_for_no_running(page, max_wait=120)
    await asyncio.sleep(2)

    await page.screenshot(path=str(FRAMES_DIR / "03_anemia_with_results.png"), full_page=True)
    print("  Saved 03_anemia_with_results.png")


async def capture_jaundice(page):
    print("Capturing Jaundice Detection with results...")
    await select_tab(page, "Neonatal Jaundice Detection")
    await asyncio.sleep(1)

    async with page.expect_file_chooser() as fc_info:
        await page.get_by_test_id("stBaseButton-secondary").click()
    file_chooser = await fc_info.value
    await file_chooser.set_files(JAUNDICE_IMAGE)

    print("  Waiting for analysis...")
    await asyncio.sleep(5)
    await wait_for_no_running(page, max_wait=120)
    await asyncio.sleep(2)

    await page.screenshot(path=str(FRAMES_DIR / "04_jaundice_with_results.png"), full_page=True)
    print("  Saved 04_jaundice_with_results.png")


async def capture_cry(page):
    print("Capturing Cry Analysis with results...")
    await select_tab(page, "Cry Analysis")
    await asyncio.sleep(1)

    async with page.expect_file_chooser() as fc_info:
        await page.get_by_test_id("stBaseButton-secondary").click()
    file_chooser = await fc_info.value
    await file_chooser.set_files(CRY_AUDIO)

    print("  Waiting for analysis...")
    await asyncio.sleep(5)
    await wait_for_no_running(page, max_wait=120)
    await asyncio.sleep(2)

    await page.screenshot(path=str(FRAMES_DIR / "06_cry_with_results.png"), full_page=True)
    print("  Saved 06_cry_with_results.png")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 810})

        print(f"Navigating to {SPACE_URL}...")
        await page.goto(SPACE_URL, timeout=30000)
        await wait_for_app(page)
        print("App loaded!")

        await capture_anemia(page)
        await capture_jaundice(page)
        await capture_cry(page)

        await browser.close()
        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
