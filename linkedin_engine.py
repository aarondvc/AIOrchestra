import asyncio
import os
import re
import json
import random
from urllib.parse import quote
from playwright.async_api import async_playwright
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# ==============================================================================
# PROFILE LOADER & MAPPER
# ==============================================================================
def load_user_profile_data() -> dict:
    """Loads the user profile JSON data."""
    profile_path = os.path.abspath("user_profile.json")
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[LinkedIn Engine] Error reading user_profile.json: {e}")
    return {}

def get_profile_context_summary() -> str:
    """Formats user_profile.json into a concise text blob for LLM context."""
    p = load_user_profile_data()
    if not p:
        return ""
    return json.dumps(p, indent=2)

# Dynamic mapping derived directly from user_profile.json
def get_field_maps():
    p = load_user_profile_data()
    legal = p.get("legal_and_work_authorization", {})
    eeo = p.get("eeo_demographics", {})
    edu = p.get("education", {})
    
    return {
        # Legal & Work Auth
        "sponsorship": legal.get("requires_sponsorship", "No"),
        "require sponsorship": legal.get("requires_sponsorship", "No"),
        "future sponsorship": legal.get("future_sponsorship", "No"),
        "authorized to work": legal.get("authorized_in_us", "Yes"),
        "legally authorized": legal.get("authorized_in_us", "Yes"),
        "legal right to work": legal.get("authorized_in_us", "Yes"),
        "18 years of age": legal.get("at_least_18", "Yes"),
        "at least 18": legal.get("at_least_18", "Yes"),
        
        # EEO Demographics
        "gender": eeo.get("gender", "Male"),
        "race": eeo.get("race_ethnicity", "Hispanic or Latino"),
        "ethnicity": eeo.get("race_ethnicity", "Hispanic or Latino"),
        "veteran": eeo.get("veteran_status", "No"),
        "disability": eeo.get("disability_status", "No"),
        
        # Education
        "bachelor": "Yes" if "bachelor" in edu.get("highest_degree", "").lower() else "No",
        "master": "Yes" if "master" in edu.get("highest_degree", "").lower() else "No",
        "degree": edu.get("highest_degree", "Bachelor's Degree"),
        "gpa": edu.get("gpa", "3.72")
    }

# ==============================================================================
# SHARED HELPER: JOB ID EXTRACTION
# ==============================================================================
JOB_ID_PATTERNS = [
    r"currentJobId=(\d+)",
    r"/jobs/view/(\d+)",
    r"view/(\d+)",
    r"[?&]jobId=(\d+)",
]

def extract_job_id(url_or_attr: str) -> str | None:
    """Tries multiple known LinkedIn URL/attribute shapes to pull a job id."""
    if not url_or_attr:
        return None
    for pattern in JOB_ID_PATTERNS:
        match = re.search(pattern, url_or_attr)
        if match:
            return match.group(1)
    return None

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

        await pwd_input.click()
        await asyncio.sleep(0.5)
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
        await asyncio.sleep(0.3)

        for char in linkedin_pwd:
            await page.keyboard.type(char, delay=65)
        await asyncio.sleep(1)

        await pwd_input.dispatch_event("input")
        await pwd_input.dispatch_event("change")
        await asyncio.sleep(0.5)

        submit_btn = page.locator("button[type='submit'], button#password-submit, button:has-text('Submit'), button:has-text('Sign in')").first
        if await submit_btn.count() and await submit_btn.is_visible():
            await submit_btn.click()
        else:
            await page.keyboard.press("Enter")

        print("[LinkedIn Engine] Password submitted. Waiting for checkpoint resolution...")

        try:
            await pwd_input.wait_for(state="hidden", timeout=15000)
            print("[LinkedIn Engine] Checkpoint modal cleared.")
        except Exception:
            print("[LinkedIn Engine] Password modal did not dismiss automatically.")
            await page.screenshot(path="checkpoint_submit_failed.png")

        await asyncio.sleep(6)
        await page.context.storage_state(path=state_file)
        print("[LinkedIn Engine] Authenticated state updated after challenge.")

        if return_url:
            print(f"[LinkedIn Engine] Navigating back to target URL: {return_url}")
            await page.goto(return_url, wait_until="domcontentloaded")
            await asyncio.sleep(5)

        return True
    return False

# ==============================================================================
# TOKEN EFFICIENT FORM EVALUATION & HEURISTICS
# ==============================================================================
llm_evaluator = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def is_local_to_location(question_text: str, user_location: dict) -> bool:
    """Checks if a location in the question matches user's actual profile location."""
    q_lower = question_text.lower()
    user_city = user_location.get("city", "").lower()
    user_state = user_location.get("state", "").lower()

    # Common location/relocation/onsite keywords
    location_keywords = ["local to", "located in", "commute to", "live in", "near", "relocate to", "in-person", "in house", "onsite"]
    
    if any(kw in q_lower for kw in location_keywords):
        # If user's city/state isn't in the question text, they aren't local
        if user_city and user_city not in q_lower:
            return False
        if user_state and user_state not in q_lower:
            return False
    return True

def evaluate_choice_heuristically(question: str, options: list[str]) -> str | None:
    """Intercepts common questions locally using user_profile.json mapped values with strict matching."""
    q_lower = question.lower()
    p = load_user_profile_data()
    user_loc = p.get("location", {})
    legal = p.get("legal_and_work_authorization", {})
    eeo = p.get("eeo_demographics", {})
    edu = p.get("education", {})

    # 1. Check Location / On-site / Commute requirement
    if any(kw in q_lower for kw in ["local to", "located in", "commute", "live in", "onsite", "in house", "in-house"]):
        user_city = user_loc.get("city", "").lower()
        user_state = user_loc.get("state", "").lower()
        
        # If specific city/state asked in question doesn't match candidate's location
        if (user_city and user_city not in q_lower) or (user_state and user_state not in q_lower):
            for opt in options:
                if opt.lower() in ["no", "false"]:
                    return opt
            return "No"

    # 2. Strict Exact Keyword Mapping (Word boundaries prevent partial string overlaps)
    strict_mappings = [
        (r"\brequire.*sponsorship\b|\bsponsor\b", legal.get("requires_sponsorship", "No")),
        (r"\bfuture.*sponsorship\b", legal.get("future_sponsorship", "No")),
        (r"\bauthorized\b|\blegal right\b", legal.get("authorized_in_us", "Yes")),
        (r"\b18 years\b|\bat least 18\b", legal.get("at_least_18", "Yes")),
        (r"\bgender\b", eeo.get("gender", "Male")),
        (r"\brace\b|\bethnicity\b", eeo.get("race_ethnicity", "Hispanic or Latino")),
        (r"\bveteran\b", eeo.get("veteran_status", "No")),
        (r"\bdisability\b", eeo.get("disability_status", "No")),
    ]

    for regex_pattern, preferred_val in strict_mappings:
        if re.search(regex_pattern, q_lower):
            if options:
                for opt in options:
                    if preferred_val.lower() == opt.lower() or preferred_val.lower() in opt.lower():
                        return opt
            return preferred_val

    return None

def extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts).strip()
    return str(content).strip()

def evaluate_form_choice(question: str, options: list[str], profile_text: str) -> str:
    """Evaluates option choice using local heuristics first, then Gemini with safe fallback."""
    if not options:
        return ""

def evaluate_form_choice(question: str, options: list[str], profile_text: str) -> str:
    if not options:
        return ""

    # Primary Gemini evaluator
    primary = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # Fallback Groq evaluator
    fallback = ChatGroq(model_name="llama-3.1-8b-instant")
    
    llm_evaluator = primary.with_fallbacks([fallback])

    try:
        res = llm_evaluator.invoke([
            SystemMessage(content=system_prompt), 
            HumanMessage(content=user_prompt)
        ])
        selected = res.content.strip().strip('"').strip("'")
        if selected in options:
            return selected
    except Exception as e:
        print(f"[LinkedIn Engine] All LLMs failed: {e}")

    # Fallback to local heuristic if both primary and fallback fail
    return options[0]

    # Safer Default: For binary Yes/No questions, default to "No" unless specifically matching profile
    yn_no = next((o for o in options if o.lower() in ["no", "false"]), None)
    if yn_no:
        return yn_no

    return options[0]

def lookup_experience_years(question_label: str) -> str:
    """Matches form question labels against skill keys in user_profile.json."""
    profile = load_user_profile_data()
    yoe_map = profile.get("years_of_experience", {})
    q_lower = question_label.lower()

    for skill, years in yoe_map.items():
        if skill != "default" and skill in q_lower:
            return str(years)
    
    return str(yoe_map.get("default", 0))

async def answer_common_questions(page):
    """Answers numeric, text, radio, native select, and LinkedIn combobox questions using user_profile.json context."""
    profile_text = get_profile_context_summary()
    profile = load_user_profile_data()

    # 1. Comboboxes & Native Selects
    comboboxes = await page.locator("button[role='combobox'], div[role='combobox'], select").all()
    for cb in comboboxes:
        try:
            await cb.scroll_into_view_if_needed()
            await asyncio.sleep(0.3)

            tag_name = await cb.evaluate("el => el.tagName.toLowerCase()")

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

            cb_text = (await cb.inner_text()).strip()
            if cb_text and "select an option" not in cb_text.lower():
                continue

            parent_container = cb.locator("xpath=ancestor::div[contains(@class, 'jobs-easy-apply-form-element') or contains(@class, 'fb-dash-form-element') or contains(@class, 'artdeco-dropdown')]").first
            q_elem = parent_container.locator("label, span.fb-dash-form-element__label, .artdeco-dropdown__label").first
            question_text = (await q_elem.inner_text()).strip() if await q_elem.count() else "Question"

            await cb.click(force=True)
            await asyncio.sleep(0.6)

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

    # 2. Radio Groups
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

    # 3. Numeric & Text Inputs populated dynamically from JSON
    text_inputs = await page.locator("input[type='text']:visible, input[type='number']:visible").all()
    for inp in text_inputs:
        try:
            await inp.scroll_into_view_if_needed()
            val = await inp.input_value()
            if not val.strip():
                inp_id = await inp.get_attribute("id") or ""
                label_elem = page.locator(f"label[for='{inp_id}']").first
                label_text = await label_elem.inner_text() if await label_elem.count() else ""
                lbl_lower = label_text.lower()

                fill_value = None
                if any(term in lbl_lower for term in ["year", "experience", "how many"]):
                    fill_value = lookup_experience_years(label_text)
                elif "gpa" in lbl_lower:
                    fill_value = profile.get("education", {}).get("gpa", "3.72")
                elif "city" in lbl_lower:
                    fill_value = profile.get("location", {}).get("city", "Atlanta")
                elif "zip" in lbl_lower or "postal" in lbl_lower:
                    fill_value = profile.get("location", {}).get("zip_code", "30043")

                if fill_value:
                    await inp.fill(fill_value)
                    await inp.dispatch_event("input")
                    await inp.dispatch_event("change")
        except Exception:
            continue

async def attach_resume(page, resume_file: str = None) -> bool:
    """Verifies that LinkedIn has pre-selected default resume or attaches file."""
    resume_card = page.locator(
        ".jobs-document-upload__file-name, "
        "div[class*='jobs-document-upload'], "
        "span:has-text('.pdf'), "
        "span:has-text('.doc')"
    ).first

    if await resume_card.count() and await resume_card.is_visible():
        card_text = await resume_card.inner_text()
        print(f"[LinkedIn Engine] Default profile resume detected on application form: '{card_text.strip()}'")
        return True

    file_input = page.locator("input[type='file']").first
    if await file_input.count():
        if resume_file and os.path.exists(resume_file) and os.path.getsize(resume_file) > 0:
            print(f"[LinkedIn Engine] File upload required. Attaching local file: {resume_file}")
            await file_input.set_input_files(resume_file)
            await file_input.dispatch_event("input")
            await file_input.dispatch_event("change")
            await asyncio.sleep(2)
            return True

    print("[LinkedIn Engine] No file upload required; proceeding with default account resume.")
    return True

TRACE_DIR = os.path.abspath("outputs/traces")

def _trace_path(label: str) -> str:
    os.makedirs(TRACE_DIR, exist_ok=True)
    clean_label = re.sub(r'[^a-zA-Z0-9]', '_', label)[:60]
    timestamp = asyncio.get_event_loop().time()
    return os.path.join(TRACE_DIR, f"{clean_label}_{int(timestamp)}.zip")

async def _finalize_session(context, browser, state_file: str, trace_path: str = None):
    if trace_path:
        try:
            await context.tracing.stop(path=trace_path)
            print(f"[LinkedIn Engine] Trace saved: {trace_path} (view with `playwright show-trace {trace_path}`)")
        except Exception as e:
            print(f"[LinkedIn Engine] Failed to save trace: {e}")

    try:
        await context.storage_state(path=state_file)
    except Exception as e:
        print(f"[LinkedIn Engine] Failed to persist session state: {e}")

    await browser.close()

# ==============================================================================
# SUBMIT CONFIRMED APPLICATION WORKFLOW
# ==============================================================================
async def submit_confirmed_application(job_url: str, company: str) -> dict:
    state_file = os.path.abspath("state.json")
    resume_file = os.path.abspath("resume.pdf")
    trace_path = _trace_path(f"submit_{company}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = await browser.new_context(
            storage_state=state_file if os.path.exists(state_file) else None,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = await context.new_page()

        clean_url = job_url.split("?")[0] if "currentJobId=" not in job_url else job_url
        print(f"[LinkedIn Engine] Resuming submission at: {clean_url}")
        await page.goto(clean_url, wait_until="domcontentloaded")
        await asyncio.sleep(5)

        await handle_security_challenges(page, state_file, return_url=clean_url)

        applied_badge = page.locator("span:has-text('Applied'), button:has-text('Applied')").first
        if await applied_badge.count() and await applied_badge.is_visible():
            await _finalize_session(context, browser, state_file, trace_path)
            return {
                "success": True,
                "status": "Already submitted",
                "confirmation_screenshot": ""
            }

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
                await _finalize_session(context, browser, state_file, trace_path)
                return {
                    "success": False,
                    "status": "Easy Apply button not accessible on resume.",
                    "confirmation_screenshot": err_shot
                }

        for step in range(12):
            await asyncio.sleep(1.5)

            await answer_common_questions(page)

            await attach_resume(page, resume_file)

            modal_body = page.locator(".jobs-easy-apply-modal__content, div[role='dialog']").first
            if await modal_body.count():
                try:
                    await modal_body.evaluate("el => el.scrollTop = el.scrollHeight")
                except Exception:
                    pass
                await asyncio.sleep(0.5)

            submit_btn = page.locator("button[aria-label='Submit application'], button:has-text('Submit application')").first
            if await submit_btn.count():
                print("[LinkedIn Engine] Scrolling to and clicking 'Submit application'...")
                await submit_btn.scroll_into_view_if_needed()
                await asyncio.sleep(0.5)
                await submit_btn.click(force=True)
                await asyncio.sleep(5)
                break

            next_or_review = page.locator(
                "button[aria-label='Review your application'], button[aria-label='Continue to next step'], button:has-text('Next'), button:has-text('Review')"
            ).first
            if await next_or_review.count():
                await next_or_review.scroll_into_view_if_needed()
                await asyncio.sleep(0.3)
                await next_or_review.click(force=True)
            else:
                break

        os.makedirs("outputs/applications", exist_ok=True)
        clean_company = re.sub(r'[^a-zA-Z0-9]', '_', company)
        confirm_screenshot = os.path.abspath(f"outputs/applications/{clean_company}_CONFIRMED.png")
        await page.screenshot(path=confirm_screenshot)

        success_indicator = page.locator(
            "h3:has-text('Application submitted'), h2:has-text('Application submitted'), div:has-text('Your application was sent to'), span:has-text('Applied')"
        ).first
        is_confirmed = (await success_indicator.count() > 0) or (await page.locator("span:has-text('Applied')").count() > 0)

        await _finalize_session(context, browser, state_file, trace_path)

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
    trace_path = _trace_path(f"search_{role}")

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
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = await context.new_page()

        encoded_role = quote(role)
        encoded_loc = quote(location)
        url = f"https://www.linkedin.com/jobs/search/?keywords={encoded_role}&location={encoded_loc}&f_AL=true"

        print(f"[LinkedIn Engine] Navigating to: {url}")
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(4)

        handled = await handle_security_challenges(page, state_file, return_url=url)
        if not handled:
            await asyncio.sleep(2)
            await handle_security_challenges(page, state_file, return_url=url)

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
            await _finalize_session(context, browser, state_file, trace_path)
            return {"error": "No jobs found or page layout blocked. Verify search query and filters."}

        target_job = None
        for i, card in enumerate(cards[:10]):
            try:
                job_id_attr = await card.get_attribute("data-job-id")
                if not job_id_attr:
                    card_link = card.locator("a[data-control-name='job_card_click'], a.job-card-list__title--link, a.job-card-container__link").first
                    href = await card_link.get_attribute("href") if await card_link.count() else ""
                    job_id = extract_job_id(href)
                else:
                    job_id = job_id_attr

                if job_id and job_id in excluded_ids:
                    print(f"[LinkedIn Engine] Skipping already applied Job ID: {job_id}")
                    continue

                await card.scroll_into_view_if_needed()
                await card.click()
                await asyncio.sleep(2.5 + random.uniform(0, 1.5))

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
            await _finalize_session(context, browser, state_file, trace_path)
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

            await attach_resume(page, resume_file)

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

        if not os.path.exists(screenshot_path):
            await page.screenshot(path=screenshot_path)

        await _finalize_session(context, browser, state_file, trace_path)

        return {
            "success": True,
            "title": target_job["title"],
            "company": target_job["company"],
            "screenshot": screenshot_path,
            "job_url": target_job["url"]
        }