"""E2E: the global Keyboard shortcuts dialog opens and reflects shell context.

The dialog is mounted once from ``AppShell`` and opens from the global
``Ctrl/Cmd+/`` handler in ``KeyboardShortcutsDialog``. Its pinned-session row is
runtime-specific: plain browser tabs show ``Ctrl/Cmd+Alt+1...0`` because
``Ctrl/Cmd+digit`` belongs to browser tab switching, while the Electron shell
shows ``Ctrl/Cmd+1...0`` because native app menus own those browser tab
shortcuts instead.

These tests drive the real browser overlay instead of the component unit tests:
load an actual session route, press the global shortcut, assert representative
rows are visible, and then repeat with a minimal Electron preload stub so the
native-shell shortcut copy is covered by the same Playwright suite that guards
the rest of the web UI.
"""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect

_NATIVE_SHELL_INIT_SCRIPT = """
window.omnigentDesktop = {
  kind: "electron",
  setBadgeCount: function () {},
  notify: function () { return Promise.resolve(false); },
  onNotificationActivated: function () { return function () {}; },
  getServerPicker: function () { return Promise.resolve(null); },
  switchServer: function () { return Promise.resolve(); },
  openServerSetup: function () {},
};
"""


def _open_shortcuts_dialog(page: Page) -> None:
    """Press the global Keyboard shortcuts hotkey and wait for the dialog."""
    page.locator("body").press("ControlOrMeta+/")
    dialog = page.get_by_role("dialog", name="Keyboard shortcuts")
    expect(dialog).to_be_visible(timeout=10_000)


def test_keyboard_shortcuts_dialog_opens_from_global_hotkey(
    page: Page, seeded_session: tuple[str, str]
) -> None:
    """The browser SPA opens the shortcut reference with Ctrl/Cmd+/."""
    base_url, session_id = seeded_session

    page.goto(f"{base_url}/c/{session_id}")
    _open_shortcuts_dialog(page)

    dialog = page.get_by_role("dialog", name="Keyboard shortcuts")
    expect(dialog).to_contain_text("Open command palette")
    expect(dialog).to_contain_text("Show keyboard shortcuts")
    expect(dialog).to_contain_text("Send message")
    expect(dialog).to_contain_text("Toggle workspace sidebar")

    row = dialog.locator("li").filter(has_text="Jump to pinned session")
    expect(row.locator("kbd")).to_have_count(3)
    expect(row.locator("kbd").last).to_have_text(re.compile(r"^1.*0$"))


def test_keyboard_shortcuts_dialog_uses_electron_pinned_session_chord(
    page: Page, seeded_session: tuple[str, str]
) -> None:
    """Under the Electron shell bridge, the pinned-session row drops Alt."""
    base_url, session_id = seeded_session
    page.add_init_script(_NATIVE_SHELL_INIT_SCRIPT)

    page.goto(f"{base_url}/c/{session_id}")
    _open_shortcuts_dialog(page)

    dialog = page.get_by_role("dialog", name="Keyboard shortcuts")
    row = dialog.locator("li").filter(has_text="Jump to pinned session")
    expect(row).to_be_visible()
    expect(row.locator("kbd")).to_have_count(2)
    expect(row.locator("kbd").last).to_have_text(re.compile(r"^1.*0$"))
