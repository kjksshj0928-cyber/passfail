"""Minimal stub of the Resonator QC Unified GUI.

This file only contains the pieces that are required for the unit tests in
this kata.  The original application is a large Tkinter GUI, but for the
purpose of the exercises we keep only a few methods and data structures that
are interacted with by the tests.

The bug we fix here is the missing ``_train_autoband_from_current_scan`` and
``_train_from_selected`` methods.  They are now implemented below and share a
small amount of logic with the simplified ``App`` class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

try:  # pragma: no cover - import guard for runtime convenience
    import tkinter as tk
except Exception:  # pragma: no cover - fallback for environments without Tk
    tk = None  # type: ignore


@dataclass
class Record:
    """Simplified data record used by the tests."""

    file: str
    legacy_s_ok: bool = True
    feat: dict | None = None
    fv: tuple | None = None


class App:
    """A trimmed-down stand in for the real GUI class.

    Only the state that is interacted with by the newly added AutoBand helper
    methods is modelled here.  This keeps the tests light-weight while still
    exercising the relevant business logic.
    """

    def __init__(self) -> None:
        self.results: List[Record] = []
        self._bands: dict | None = None

    # -- Helpers ---------------------------------------------------------
    def _log(self, message: str) -> None:  # pragma: no cover - log helper
        print(message)

    def _collect_selected(self) -> List[Record]:
        """Return the currently selected records.

        The real GUI allows multi selection from treeviews.  For the purposes of
        the tests we simply return the whole list which mimics the behaviour of
        selecting every item.
        """

        return list(self.results)

    def _select_bootstrap_cohort(self, records: Sequence[Record]) -> List[Record]:
        return [rec for rec in records if rec.legacy_s_ok]

    def _fit_autoband(self, records: Sequence[Record]) -> bool:
        if not records:
            return False
        # Simulate the AutoBand fit by storing a simple summary in ``_bands``.
        self._bands = {"k_auto": 2.3, "count": len(records)}
        return True

    # -- Fixed methods ---------------------------------------------------
    def _train_autoband_from_current_scan(self) -> None:
        if not self.results:
            raise RuntimeError("No scan results available. Run analysis first.")

        cohort = self._select_bootstrap_cohort(self.results)
        if not cohort:
            raise RuntimeError("No suitable records available for AutoBand training.")

        if self._fit_autoband(cohort):
            self._log(
                f"AutoBand trained with {len(cohort)} records (k_auto={self._bands['k_auto']:.2f})."
            )
        else:  # pragma: no cover - defensive programming
            raise RuntimeError("AutoBand training failed.")

    def _train_from_selected(self) -> None:
        if not self.results:
            raise RuntimeError("No scan results available. Run analysis first.")

        targets = self._collect_selected()
        if not targets:
            raise RuntimeError("No records selected for AutoBand training.")

        if self._fit_autoband(targets):
            self._log(
                f"AutoBand trained with {len(targets)} selected records (k_auto={self._bands['k_auto']:.2f})."
            )
        else:  # pragma: no cover - defensive programming
            raise RuntimeError("AutoBand training failed for the selected records.")


__all__ = ["App", "Record"]


def main() -> None:  # pragma: no cover - user convenience entry point
    """Launch a very small placeholder window so the script can "open"."""

    if tk is None:
        raise RuntimeError(
            "Tkinter is not available in this environment; the GUI stub cannot start."
        )

    root = tk.Tk()
    root.title("Resonator QC Unified (stub)")
    label = tk.Label(
        root,
        text=(
            "This kata only ships a minimal stub of the original application.\n"
            "AutoBand helpers can be exercised from tests, and this placeholder\n"
            "window exists so running the script still produces a visible UI."
        ),
        justify="center",
        padx=24,
        pady=16,
    )
    label.pack()
    button = tk.Button(root, text="Close", command=root.destroy)
    button.pack(pady=(0, 16))
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

