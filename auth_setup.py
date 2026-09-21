import asyncio
import os
from playwright.async_api import async_playwright

async def setup_session():
    async with async_playwright() as p:
        # Launch Chromium directly
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )
        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()
        await page.goto("https://www.linkedin.com/login")
        
        print("\n" + "="*60)
        print("1. Log in on the browser window.")
        print("2. Navigate freely or complete 2FA.")
        print("3. Once your feed is visible, return here and press Enter.")
        print("="*60 + "\n")

        await asyncio.to_thread(input, "Press ENTER after logging in successfully: ")

        # Save cookies and local storage tokens to state.json
        await context.storage_state(path="state.json")
        print("\nSession saved to state.json successfully.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(setup_session())