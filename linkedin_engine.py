import asyncio
import os
import re
from urllib.parse import quote
from playwright.async_api import async_playwright
from career_tools import load_user_profile

async def handle_security_challenges(page, state_file: str, return_url: str = None) -> bool:
    """Detects and resolves inline password confirmation prompts."""
    pwd_input = page.locator("input[type='password'], input#password, input#session_password").first
    challenge_header = page.locator("h1:has-text('Enter your LinkedIn password'), header:has-text('Enter your LinkedIn password')").first

    if (await pwd_input.count() and await pwd_input.is_visible()) or (await challenge_header.count() and await challenge_header.is_visible()):
        print("[LinkedIn Engine] Password checkpoint detected. Supplying password...")
        linkedin_pwd = os.getenv("LINKEDIN_PASSWORD", "")
        if not linkedin_pwd:
            print("[LinkedIn Engine] Error: LINKEDIN_PASSWORD is not set in environment.")
            return False

        # Ensure field is interactive and fully cleared
        await pwd_input.click()
        await asyncio.sleep(0.5)
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
        await asyncio.sleep(0.3)

        # Type using simulated hardware events
        for char in linkedin_pwd:
            await page.keyboard.type(char, delay=65)
        await asyncio.sleep(1)

        # Dispatch DOM input events for reactive forms
        await pwd_input.dispatch_event("input")
        await pwd_input.dispatch_event("change")
        await asyncio.sleep(0.5)

        submit_btn = page.locator("button[type='submit'], button#password-submit, button:has-text('Submit'), button:has-text('Sign in')").first
        if await submit_btn.count() and await submit_btn.is_visible():
            await submit_btn.click()
        else:
            await page.keyboard.press("Enter")

        print("[LinkedIn Engine] Password submitted. Waiting for checkpoint resolution...")

        # Wait for the password modal to disappear
        try:
            await pwd_input.wait_for(state="hidden", timeout=15000)
            print("[LinkedIn Engine] Checkpoint modal cleared.")
        except Exception:
            print("[LinkedIn Engine] Password modal did not dismiss automatically.")
            await page.screenshot(path="checkpoint_submit_failed.png")

        await asyncio.sleep(6)

        # Persist updated session tokens immediately
        await page.context.storage_state(path=state_file)
        print("[LinkedIn Engine] Authenticated state updated after challenge.")

        # Re-navigate to the desired target URL if provided
        if return_url:
            print(f"[LinkedIn Engine] Navigating back to target URL: {return_url}")
            await page.goto(return_url, wait_until="domcontentloaded")
            await asyncio.sleep(5)

        return True
    return False

# ==============================================================================
# COMMON FORM FILLING UTILITIES
# ==============================================================================
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Fast LLM instance for immediate form evaluations
llm_evaluator = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def evaluate_form_choice(question: str, options: list[str], profile_text: str) -> str:
    """Uses LLM to evaluate the most accurate answer given the user's profile."""
    system_prompt = (
        "You are an assistant answering job application questions truthfully on behalf of a candidate.\n"
        "Analyze the question, the available multiple-choice options, and the candidate's profile.\n"
        "- If the question asks whether the candidate lives in a list of states/locations (e.g. TX, TN, FL, LA, OH) and the candidate does not reside in those locations, choose 'No'.\n"
        "- If the question asks about professional/production experience with a specific tool (e.g., Kubernetes in production) and it is absent from the profile, choose 'No'.\n"
        "- Output ONLY the exact text of the best matching option from the provided options list."
    )
    user_prompt = (
        f"Candidate Profile:\n{profile_text}\n\n"
        f"Question: {question}\n"
        f"Available Options: {json.dumps(options)}\n\n"
        "Chosen Option:"
    )
    try:
        res = llm_evaluator.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        selected = res.content.strip().strip('"').strip("'")
        return selected
    except Exception:
        return options[0] if options else ""

async def answer_common_questions(page):
    """Answers numeric, text, radio, native select, and LinkedIn combobox questions using profile context."""
    profile_text = load_user_profile()

    # Ensure we scroll the modal to expose form fields
    modal_body = page.locator(".jobs-easy-apply-modal__content, div[role='dialog']").first

    # --------------------------------------------------------------------------
    # 1. LinkedIn Custom Dropdowns / Comboboxes (button or div with role='combobox')
    # --------------------------------------------------------------------------
    comboboxes = await page.locator("button[role='combobox'], div[role='combobox'], select").all()
    for cb in comboboxes:
        try:
            await cb.scroll_into_view_if_needed()
            await asyncio.sleep(0.3)

            tag_name = await cb.evaluate("el => el.tagName.toLowerCase()")

            # Case A: Standard native <select>
            if tag_name == "select":
                sel_id = await cb.get_attribute("id") or ""
                label_elem = page.locator(f"label[for='{sel_id}']").first
                q_text = await label_elem.inner_text() if await label_elem.count() else ""
                
                if not q_text:
                    wrapper = cb.locator("xpath=ancestor::div[contains(@class, 'jobs-easy-apply-form-element') or contains(@class, 'fb-dash-form-element')]").first
                    if await wrapper.count():
                        q_text = await wrapper.locator("label, span").first.inner_text()

                options = await cb.locator("option").all()
                opt_map = {}
                for opt in options:
                    t = (await opt.inner_text()).strip()
                    v = await opt.get_attribute("value")
                    if v and "select" not in t.lower():
                        opt_map[t] = v

                if opt_map:
                    chosen = evaluate_form_choice(q_text or "Select option", list(opt_map.keys()), profile_text)
                    val = next((v for k, v in opt_map.items() if k.lower() == chosen.lower()), list(opt_map.values())[0])
                    await cb.select_option(value=val)
                    await cb.dispatch_event("change")
                continue

            # Case B: LinkedIn Art Deco Custom Combobox (Button)
            cb_text = (await cb.inner_text()).strip()
            # If already filled with an answer, skip
            if cb_text and "select an option" not in cb_text.lower():
                continue

            # Find the enclosing question text
            parent_container = cb.locator("xpath=ancestor::div[contains(@class, 'jobs-easy-apply-form-element') or contains(@class, 'fb-dash-form-element') or contains(@class, 'artdeco-dropdown')]").first
            q_elem = parent_container.locator("label, span.fb-dash-form-element__label, .artdeco-dropdown__label").first
            question_text = (await q_elem.inner_text()).strip() if await q_elem.count() else "Question"

            # Click to expand the options menu
            await cb.click(force=True)
            await asyncio.sleep(0.6)

            # Locate the opened dropdown options
            options_loc = page.locator("div[role='listbox'] div[role='option'], div.artdeco-dropdown__item, ul.dropdown-list li")
            opt_count = await options_loc.count()

            if opt_count > 0:
                opt_names = []
                opt_elements = []
                for idx in range(opt_count):
                    el = options_loc.nth(idx)
                    txt = (await el.inner_text()).strip()
                    if txt and "select an option" not in txt.lower():
                        opt_names.append(txt)
                        opt_elements.append(el)

                if opt_names:
                    chosen_text = evaluate_form_choice(question_text, opt_names, profile_text)
                    
                    # Click the matching option
                    selected_elem = None
                    for i, name in enumerate(opt_names):
                        if name.lower() == chosen_text.lower() or chosen_text.lower() in name.lower():
                            selected_elem = opt_elements[i]
                            break

                    if not selected_elem:
                        selected_elem = opt_elements[0]

                    await selected_elem.scroll_into_view_if_needed()
                    await selected_elem.click(force=True)
                    await asyncio.sleep(0.4)

        except Exception as e:
            print(f"[LinkedIn Engine] Combobox error: {e}")
            continue

    # --------------------------------------------------------------------------
    # 2. Radio Groups (<fieldset>)
    # --------------------------------------------------------------------------
    fieldsets = await page.locator("fieldset:visible").all()
    for fs in fieldsets:
        try:
            await fs.scroll_into_view_if_needed()
            legend = fs.locator("legend").first
            question_text = (await legend.inner_text()).strip() if await legend.count() else ""
            radios = await fs.locator("label").all()

            radio_map = {}
            for r in radios:
                txt = (await r.inner_text()).strip()
                if txt:
                    radio_map[txt] = r

            if radio_map and question_text:
                chosen_text = evaluate_form_choice(question_text, list(radio_map.keys()), profile_text)
                target = next((el for t, el in radio_map.items() if t.lower() == chosen_text.lower()), None)
                if target:
                    await target.click(force=True)
                    await asyncio.sleep(0.2)
        except Exception:
            continue

    # --------------------------------------------------------------------------
    # 3. Numeric & Text Inputs
    # --------------------------------------------------------------------------
    text_inputs = await page.locator("input[type='text']:visible, input[type='number']:visible").all()
    for inp in text_inputs:
        try:
            await inp.scroll_into_view_if_needed()
            val = await inp.input_value()
            if not val.strip():
                inp_id = await inp.get_attribute("id") or ""
                label_elem = page.locator(f"label[for='{inp_id}']").first
                label_text = await label_elem.inner_text() if await label_elem.count() else ""

                if any(term in label_text.lower() for term in ["year", "experience", "how many"]):
                    await inp.fill("2")
                    await inp.dispatch_event("input")
                    await inp.dispatch_event("change")
                elif "gpa" in label_text.lower():
                    await inp.fill("3.5")
                    await inp.dispatch_event("input")
                    await inp.dispatch_event("change")
        except Exception:
            continue

# ==============================================================================
# SUBMIT CONFIRMED APPLICATION WORKFLOW
# ==============================================================================
async def submit_confirmed_application(job_url: str, company: str) -> dict:
    state_file = os.path.abspath("state.json")
    resume_file = os.path.abspath("resume.pdf")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = await browser.new_context(
            storage_state=state_file if os.path.exists(state_file) else None,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        clean_url = job_url.split("?")[0] if "currentJobId=" not in job_url else job_url
        print(f"[LinkedIn Engine] Resuming submission at: {clean_url}")
        await page.goto(clean_url, wait_until="domcontentloaded")
        await asyncio.sleep(5)

        # Handle checkpoint if triggered
        await handle_security_challenges(page, state_file, return_url=clean_url)

        # 1. Check if already submitted
        applied_badge = page.locator("span:has-text('Applied'), button:has-text('Applied')").first
        if await applied_badge.count() and await applied_badge.is_visible():
            await context.storage_state(path=state_file)
            await browser.close()
            return {
                "success": True,
                "status": "Already submitted",
                "confirmation_screenshot": ""
            }

        # 2. Open the Easy Apply modal
        modal = page.locator("div[role='dialog'], .jobs-easy-apply-modal").first
        if not (await modal.count() and await modal.is_visible()):
            apply_candidates = [
                ".jobs-apply-button",
                "button.jobs-apply-button",
                ".jobs-apply-button--top-card button",
                "button:has-text('Easy Apply')",
                ".jobs-unified-top-card button:has-text('Easy Apply')",
                ".job-card-container--clickable"
            ]

            apply_clicked = False
            for selector in apply_candidates:
                btn = page.locator(selector).first
                if await btn.count():
                    try:
                        await btn.scroll_into_view_if_needed()
                        await asyncio.sleep(0.5)
                        await btn.click(force=True)
                        apply_clicked = True
                        print(f"[LinkedIn Engine] Clicked Easy Apply using selector '{selector}'.")
                        await asyncio.sleep(3)
                        break
                    except Exception:
                        continue

            if not apply_clicked:
                os.makedirs("outputs/applications", exist_ok=True)
                err_shot = os.path.abspath("outputs/applications/apply_button_missing.png")
                await page.screenshot(path=err_shot)
                await context.storage_state(path=state_file)
                await browser.close()
                return {
                    "success": False,
                    "status": "Easy Apply button not accessible on resume.",
                    "confirmation_screenshot": err_shot
                }

        # 3. Step forward through modal to final submission
        for step in range(12):
            await asyncio.sleep(1.5)

            # Auto-fill common inputs
            await answer_common_questions(page)

            # Upload resume if prompted
            file_input = page.locator("input[type='file']").first
            if await file_input.count() and os.path.exists(resume_file):
                try:
                    await file_input.set_input_files(resume_file)
                except Exception:
                    pass

            # Scroll down the modal content to reveal action buttons
            modal_body = page.locator(".jobs-easy-apply-modal__content, div[role='dialog']").first
            if await modal_body.count():
                try:
                    await modal_body.evaluate("el => el.scrollTop = el.scrollHeight")
                except Exception:
                    pass
                await asyncio.sleep(0.5)

            # Check for Submit application
            submit_btn = page.locator("button[aria-label='Submit application'], button:has-text('Submit application')").first
            if await submit_btn.count():
                print("[LinkedIn Engine] Scrolling to and clicking 'Submit application'...")
                await submit_btn.scroll_into_view_if_needed()
                await asyncio.sleep(0.5)
                await submit_btn.click(force=True)
                await asyncio.sleep(5)
                break

            # Advance via Next or Review
            next_or_review = page.locator(
                "button[aria-label='Review your application'], button[aria-label='Continue to next step'], button:has-text('Next'), button:has-text('Review')"
            ).first
            if await next_or_review.count():
                await next_or_review.scroll_into_view_if_needed()
                await asyncio.sleep(0.3)
                await next_or_review.click(force=True)
            else:
                break

        # 4. Save proof screenshot of the completed submission modal
        os.makedirs("outputs/applications", exist_ok=True)
        clean_company = re.sub(r'[^a-zA-Z0-9]', '_', company)
        confirm_screenshot = os.path.abspath(f"outputs/applications/{clean_company}_CONFIRMED.png")
        await page.screenshot(path=confirm_screenshot)

        # 5. Check confirmation indicators
        success_indicator = page.locator(
            "h3:has-text('Application submitted'), h2:has-text('Application submitted'), div:has-text('Your application was sent to'), span:has-text('Applied')"
        ).first
        is_confirmed = (await success_indicator.count() > 0) or (await page.locator("span:has-text('Applied')").count() > 0)

        await context.storage_state(path=state_file)
        await browser.close()

        return {
            "success": is_confirmed,
            "status": "Application submitted successfully" if is_confirmed else "Submission executed",
            "confirmation_screenshot": confirm_screenshot
        }

# ==============================================================================
# SEARCH & EASY APPLY WORKFLOW
# ==============================================================================
async def search_and_prep_easy_apply(role: str, location: str = "United States", excluded_ids: list = None) -> dict:
    if excluded_ids is None:
        excluded_ids = []
        
    resume_file = os.path.abspath("resume.pdf")

    if not os.path.exists(resume_file):
        return {"error": f"resume.pdf not found at: {resume_file}. Add your PDF to the project root."}

    print(f"\n[LinkedIn Engine] Starting search for role: '{role}' in '{location}'...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--start-maximized"]
        )
        
        state_file = os.path.abspath("state.json")
        context = await browser.new_context(
            storage_state=state_file if os.path.exists(state_file) else None,
            no_viewport=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        encoded_role = quote(role)
        encoded_loc = quote(location)
        url = f"https://www.linkedin.com/jobs/search/?keywords={encoded_role}&location={encoded_loc}&f_AL=true"
        
        print(f"[LinkedIn Engine] Navigating to: {url}")
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(4)

        # Handle checkpoint if triggered
        handled = await handle_security_challenges(page, state_file, return_url=url)
        if not handled:
            await asyncio.sleep(2)
            await handle_security_challenges(page, state_file, return_url=url)

        # Dismiss app modal if present
        try:
            dismiss_btn = page.locator("button[aria-label='Dismiss'], button.modal__dismiss, button[data-tracking-control-name='public_jobs_contextual-sign-in-modal_modal_dismiss']").first
            if await dismiss_btn.count() and await dismiss_btn.is_visible():
                await dismiss_btn.click()
                await asyncio.sleep(1)
        except Exception:
            pass

        card_selectors = [
            ".job-card-container",
            "ul.scaffold-layout__list-container li",
            ".jobs-search-results__list-item",
            "li.jobs-search-results__list-item",
            "div[data-job-id]",
            "li[data-occludable-job-id]",
            "ul.jobs-search__results-list li",
            "div.base-card"
        ]

        cards = []
        for selector in card_selectors:
            found = await page.locator(selector).all()
            if found:
                cards = found
                print(f"[LinkedIn Engine] Found {len(cards)} listings using selector '{selector}'.")
                break

        if not cards:
            await context.storage_state(path=state_file)
            await browser.close()
            return {"error": "No jobs found or page layout blocked. Verify search query and filters."}

        target_job = None
        for i, card in enumerate(cards[:10]):
            try:
                # 1. Extract job ID first before opening or clicking
                job_id_attr = await card.get_attribute("data-job-id")
                if not job_id_attr:
                    card_link = card.locator("a[data-control-name='job_card_click'], a.job-card-list__title--link, a.job-card-container__link").first
                    href = await card_link.get_attribute("href") if await card_link.count() else ""
                    match = re.search(r"view/(\d+)", href) or re.search(r"currentJobId=(\d+)", href)
                    job_id = match.group(1) if match else None
                else:
                    job_id = job_id_attr

                # 2. Check exclusion list
                if job_id and job_id in excluded_ids:
                    print(f"[LinkedIn Engine] Skipping already applied Job ID: {job_id}")
                    continue

                await card.scroll_into_view_if_needed()
                await card.click()
                await asyncio.sleep(2.5)

                apply_btn = page.locator("button.jobs-apply-button, button[data-job-id]").first
                if await apply_btn.count() and await apply_btn.is_visible():
                    btn_text = await apply_btn.inner_text()
                    if "easy apply" not in btn_text.lower():
                        continue

                    title_elem = page.locator(".job-details-jobs-unified-top-card__job-title, h1.t-24").first
                    comp_elem = page.locator(".job-details-jobs-unified-top-card__company-name, .job-details-jobs-unified-top-card__primary-description a").first

                    title = await title_elem.inner_text() if await title_elem.count() else role
                    company = await comp_elem.inner_text() if await comp_elem.count() else "Target Company"

                    direct_job_url = f"https://www.linkedin.com/jobs/search/?currentJobId={job_id}&f_AL=true" if job_id else page.url

                    target_job = {
                        "title": title.strip(),
                        "company": company.strip(),
                        "url": direct_job_url,
                        "job_id": job_id
                    }
                    print(f"[LinkedIn Engine] Target found: {title.strip()} at {company.strip()} ({direct_job_url})")
                    await apply_btn.click()
                    await asyncio.sleep(2)
                    break
            except Exception as e:
                print(f"[LinkedIn Engine] Error checking listing {i}: {e}")
                continue

        if not target_job:
            await context.storage_state(path=state_file)
            await browser.close()
            return {"error": "Could not find an accessible Easy Apply button on top listings."}

        max_steps = 7
        step = 0
        os.makedirs("outputs/applications", exist_ok=True)
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', target_job['company'])
        screenshot_path = os.path.abspath(f"outputs/applications/{clean_name}_review.png")

        while step < max_steps:
            step += 1
            await asyncio.sleep(1.5)
            
            await answer_common_questions(page)

            file_input = page.locator("input[type='file']").first
            if await file_input.count():
                try:
                    await file_input.set_input_files(resume_file)
                    print("[LinkedIn Engine] Attached resume.pdf")
                except Exception:
                    pass

            submit_btn = page.locator("button[aria-label='Submit application'], button:has-text('Submit application')").first
            review_btn = page.locator("button[aria-label='Review your application'], button:has-text('Review')").first
            next_btn = page.locator("button[aria-label='Continue to next step'], button:has-text('Next')").first
            review_header = page.locator("h3:has-text('Review your application'), h2:has-text('Review your application')").first

            if (await submit_btn.count()) or (await review_header.count() and await review_header.is_visible()):
                print("[LinkedIn Engine] Review stage reached. Taking proof screenshot...")
                if await submit_btn.count():
                    await submit_btn.scroll_into_view_if_needed()
                await page.screenshot(path=screenshot_path)
                break

            if await review_btn.count() and await review_btn.is_visible():
                await review_btn.click()
                continue

            if await next_btn.count() and await next_btn.is_visible():
                await next_btn.click()
                continue

            print("[LinkedIn Engine] Custom questions or intermediate modal reached. Capturing current screen state.")
            await page.screenshot(path=screenshot_path)
            break

        # Fallback guarantee: verify screenshot exists before continuing
        if not os.path.exists(screenshot_path):
            await page.screenshot(path=screenshot_path)

        await context.storage_state(path=state_file)
        await browser.close()

        return {
            "success": True,
            "title": target_job["title"],
            "company": target_job["company"],
            "screenshot": screenshot_path,
            "job_url": target_job["url"]
        }