"""
Automated Screenshot Capture for Streamlit Pages using Playwright
Architecture Reference: docs/EXPLANATIONS.md

Captures:
1. Page 1: Research Workspace
2. Page 2: Evaluation Dashboard
3. Page 3: Phase 1 vs Phase 2 Comparison
"""

import os
import shutil
import time
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = os.path.join("outputs", "screenshots")
ARTIFACTS_DIR = r"C:\Users\aksha\.gemini\antigravity\brain\4e5a1a56-55fd-4183-91ab-a618eb916796"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use full HD viewport
        context = browser.new_context(viewport={"width": 1400, "height": 950})
        page = context.new_page()

        print("[CAPTURE] Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(3)

        # -------------------------------------------------------------
        # 1. Page 1: Research Workspace
        # -------------------------------------------------------------
        print("[CAPTURE] Capturing Page 1: Research Workspace...")
        # Verify default page is active
        page.wait_for_selector("text=Research Workspace", timeout=20000)
        time.sleep(2)
        p1_path = os.path.join(SCREENSHOTS_DIR, "page1_research_workspace.png")
        page.screenshot(path=p1_path, full_page=True)
        shutil.copy2(p1_path, os.path.join(ARTIFACTS_DIR, "page1_research_workspace.png"))
        print(f"[CAPTURE] Saved: {p1_path}")

        # -------------------------------------------------------------
        # 2. Page 2: Evaluation Dashboard
        # -------------------------------------------------------------
        print("[CAPTURE] Navigating to Page 2: Evaluation Dashboard...")
        # Click radio button for "Evaluation Dashboard" in sidebar
        page.click("text=Evaluation Dashboard")
        time.sleep(3)
        page.wait_for_selector("text=Experimental Evaluation Dashboard", timeout=20000)
        time.sleep(2)
        p2_path = os.path.join(SCREENSHOTS_DIR, "page2_evaluation_dashboard.png")
        page.screenshot(path=p2_path, full_page=True)
        shutil.copy2(p2_path, os.path.join(ARTIFACTS_DIR, "page2_evaluation_dashboard.png"))
        print(f"[CAPTURE] Saved: {p2_path}")

        # -------------------------------------------------------------
        # 3. Page 3: Phase 1 vs Phase 2 Comparison
        # -------------------------------------------------------------
        print("[CAPTURE] Navigating to Page 3: Phase 1 vs Phase 2 Comparison...")
        # Click radio button for "Phase 1 vs Phase 2 Comparison" in sidebar
        page.click("text=Phase 1 vs Phase 2 Comparison")
        time.sleep(3)
        page.wait_for_selector("text=Phase 1 (Baseline) vs Phase 2 (Critic Loop) Comparison", timeout=20000)
        time.sleep(2)
        p3_path = os.path.join(SCREENSHOTS_DIR, "page3_comparison_matrix.png")
        page.screenshot(path=p3_path, full_page=True)
        shutil.copy2(p3_path, os.path.join(ARTIFACTS_DIR, "page3_comparison_matrix.png"))
        print(f"[CAPTURE] Saved: {p3_path}")

        browser.close()
        print("[CAPTURE] All 3 screenshots captured and copied to artifact directory successfully.")

if __name__ == "__main__":
    capture_all()
