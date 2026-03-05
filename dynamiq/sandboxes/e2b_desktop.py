"""E2B Desktop sandbox implementation."""

from e2b_desktop import Sandbox as E2BDesktopSDK

from dynamiq.sandboxes.e2b import E2BSandbox


class E2BDesktopSandbox(E2BSandbox):
    """E2B Desktop sandbox variant.

    Identical to E2BSandbox but uses the e2b_desktop SDK which provides
    a full desktop environment (screen, mouse, keyboard) instead of the
    headless Code Interpreter sandbox.
    """

    @property
    def _sdk_class(self) -> type:
        return E2BDesktopSDK
