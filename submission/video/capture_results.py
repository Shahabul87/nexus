"""
Capture screenshots of NEXUS demo with actual file uploads and results.
Uses Playwright to navigate the HF Space and upload sample files.
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

SPACE_URL = "https://shahabul-nexus.hf.space"
FRAMES_DIR = Path(__file__).parent / "frames"
ASSETS_DIR = Path(__file__).parent / "assets"

ANEMIA_IMAGE = str(ASSETS_DIR / "anemia" / "20200223_185602_palpebral.png")
JAUNDICE_IMAGE = str(ASSETS_DIR / "jaundice" / "sample_jaundice_1.jpg")
CRY_AUDIO = str(ASSETS_DIR / "cry" / "Real_Infantcry.wav")


async def wait_for_app(page, timeout=30000):
    """Wait for Streamlit app to fully load."""
    await page.wait_for_selector("text=NEXUS", timeout=timeout)
    await asyncio.sleep(3)


async def select_tab(page, tab_name):
    """Select a radio button tab in the sidebar."""
    await page.locator("label").filter(has_text=tab_name).click()
    await asyncio.sleep(2)


async def capture_anemia(page):
    """Capture anemia tab with uploaded image and results."""
    print("Capturing Anemia Screening with results...")
    await select_tab(page, "Maternal Anemia Screening")

    # Upload file
    async with page.expect_file_chooser() as fc_info:
        await page.get_by_test_id("stBaseButton-secondary").click()
    file_chooser = await fc_info.value
    await file_chooser.set_files(ANEMIA_IMAGE)

    # Wait for analysis to complete (model loading + inference)
    print("  Waiting for anemia analysis...")
    await asyncio.sleep(60)  # MedSigLIP loading takes time

    # Check if still running
    try:
        running = await page.locator("text=Running").count()
        if running > 0:
            print("  Still running, waiting more...")
            await asyncio.sleep(60)
    except Exception:
        pass

    await page.screenshot(path=str(FRAMES_DIR / "03_anemia_with_results.png"), full_page=True)
    print("  Saved 03_anemia_with_results.png")


async def capture_jaundice(page):
    """Capture jaundice tab with uploaded image and results."""
    print("Capturing Jaundice Detection with results...")
    await select_tab(page, "Neonatal Jaundice Detection")
    await asyncio.sleep(2)

    # Upload file
    async with page.expect_file_chooser() as fc_info:
        await page.get_by_test_id("stBaseButton-secondary").click()
    file_chooser = await fc_info.value
    await file_chooser.set_files(JAUNDICE_IMAGE)

    # Wait for analysis
    print("  Waiting for jaundice analysis...")
    await asyncio.sleep(60)

    try:
        running = await page.locator("text=Running").count()
        if running > 0:
            print("  Still running, waiting more...")
            await asyncio.sleep(60)
    except Exception:
        pass

    await page.screenshot(path=str(FRAMES_DIR / "04_jaundice_with_results.png"), full_page=True)
    print("  Saved 04_jaundice_with_results.png")


async def capture_cry(page):
    """Capture cry analysis tab with uploaded audio and results."""
    print("Capturing Cry Analysis with results...")
    await select_tab(page, "Cry Analysis")
    await asyncio.sleep(2)

    # Upload file
    async with page.expect_file_chooser() as fc_info:
        await page.get_by_test_id("stBaseButton-secondary").click()
    file_chooser = await fc_info.value
    await file_chooser.set_files(CRY_AUDIO)

    # Wait for analysis
    print("  Waiting for cry analysis...")
    await asyncio.sleep(60)

    try:
        running = await page.locator("text=Running").count()
        if running > 0:
            print("  Still running, waiting more...")
            await asyncio.sleep(60)
    except Exception:
        pass

    await page.screenshot(path=str(FRAMES_DIR / "06_cry_with_results.png"), full_page=True)
    print("  Saved 06_cry_with_results.png")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 810})

        print(f"Navigating to {SPACE_URL}...")
        await page.goto(SPACE_URL, timeout=60000)
        await wait_for_app(page)
        print("App loaded!")

        # Capture anemia with results
        await capture_anemia(page)

        # Capture jaundice with results
        await capture_jaundice(page)

        # Capture cry with results (we already have this but recapture for consistency)
        await capture_cry(page)

        await browser.close()
        print("\nDone! All screenshots captured with results.")


if __name__ == "__main__":
    asyncio.run(main())
