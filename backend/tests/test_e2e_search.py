import pytest
from playwright.sync_api import Page, expect


@pytest.mark.e2e
def test_neural_search_flow(page: Page):
    """
    E2E Test for Neural Search Feature (Phase 4).
    Tests the guest mode neural search flow from start to result display.
    """
    # 1. Navigate to Home
    page.goto("http://localhost:3000/")
    page.wait_for_load_state("networkidle")

    # 2. Open Discovery Modal (Guest Mode)
    start_btn = (
        page.get_by_role("button", name="Try Guest Mode")
        .or_(page.get_by_role("button", name="Guest Settings"))
        .first
    )
    expect(start_btn).to_be_visible(timeout=10000)
    start_btn.click()

    # 3. Wait for modal to appear and find the search input
    search_input = page.get_by_label("Describe what you want to watch (Neural Search)")
    expect(search_input).to_be_visible(timeout=5000)

    # 4. Enter Neural Search Query
    search_input.fill("A mind-bending sci-fi movie about dreams")

    # 5. Submit - button should now be enabled
    submit_btn = page.get_by_role("button", name="Get Recommendations")
    expect(submit_btn).to_be_enabled(timeout=2000)
    submit_btn.click()

    # 6. Wait for API response and modal to close
    # The modal closes and recommendations appear on the main page
    # Wait for modal to close (longer timeout for API call which includes SBERT encoding)
    page.wait_for_selector("[data-mantine-modal]", state="hidden", timeout=60000)

    # 7. Verify recommendations appeared on the page
    # Check for "Top Picks for You" with "(Guest Profile)" text
    guest_profile_text = page.get_by_text("Guest Profile")
    expect(guest_profile_text).to_be_visible(timeout=10000)

    # 8. Verify movie cards are displayed (at least one card)
    movie_cards = page.locator("[class*='mantine-Card']")
    expect(movie_cards.first).to_be_visible(timeout=5000)
