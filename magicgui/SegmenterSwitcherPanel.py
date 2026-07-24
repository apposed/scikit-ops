"""
Experiment: a one-panel napari widget that switches between magicgui
segmenter "commands" via a combo, auto-updating the parameter GUI, with
an Apply button.

The point of this experiment is to test the *switching* mechanism --
picking a segmenter from a combo swaps in its auto-generated magicgui
parameter widget.  Actual segmentation working is a bonus.

Run standalone (no napari) to test the switch:
    # in IPython:
    %gui qt
    %run experiments/magicgui/SegmenterSwitcherPanel.py

Or add to napari:
    import napari
    from SegmenterSwitcherPanel import SegmenterSwitcherPanel
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(SegmenterSwitcherPanel(viewer))
"""

from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from CellposeMagicGuiCommand import CellposeCommand
from SkImageSegmenterMagicGuiCommand import OtsuCommand
from StardistMagicGuiCommand import StardistCommand


class SegmenterSwitcherPanel(QWidget):
    """Combo-driven switcher between magicgui segmenter commands."""

    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer

        # Instantiate one of each command; cache their auto GUIs.
        self._commands = {
            OtsuCommand.NAME: OtsuCommand(),
            CellposeCommand.NAME: CellposeCommand(),
            StardistCommand.NAME: StardistCommand(),
        }

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Segmenter:"))
        self.combo = QComboBox()
        self.combo.addItems(self._commands.keys())
        self.combo.currentTextChanged.connect(self._on_segmenter_changed)
        layout.addWidget(self.combo)

        # Placeholder that holds the current command's magicgui widget.
        self._param_container = QWidget()
        self._param_layout = QVBoxLayout(self._param_container)
        self._param_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._param_container)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(self.apply_btn)

        layout.addStretch()

        self._current_gui = None
        self._on_segmenter_changed(self.combo.currentText())

    # -- switching ---------------------------------------------------------

    def _on_segmenter_changed(self, name: str):
        """Swap the parameter GUI to match the selected command."""
        # Remove the previous magicgui widget, if any.
        if self._current_gui is not None:
            self._param_layout.removeWidget(self._current_gui.native)
            self._current_gui.native.setParent(None)
            self._current_gui = None

        command = self._commands[name]
        self._current_gui = command.gui  # magicgui Container
        self._param_layout.addWidget(self._current_gui.native)
        print(f"Switched to: {name}")

    # -- apply -------------------------------------------------------------

    def _current_command(self):
        return self._commands[self.combo.currentText()]

    def _get_active_image(self) -> np.ndarray | None:
        """Grab the active image layer's data from napari, if available."""
        if self.viewer is None or not self.viewer.layers:
            return None
        import napari

        for layer in reversed(self.viewer.layers):
            if isinstance(layer, napari.layers.Image):
                return np.asarray(layer.data)
        return None

    def _on_apply(self):
        command = self._current_command()
        image = self._get_active_image()
        if image is None:
            print("No image layer available to segment.")
            return
        print(f"Applying {self.combo.currentText()} ...")
        mask = command.segment(image)
        n = int(mask.max())
        print(f"  -> {n} objects")
        if self.viewer is not None:
            self.viewer.add_labels(mask, name=f"{self.combo.currentText()}")


if __name__ == "__main__":
    # Standalone switch test (no napari) -- just proves the combo swaps GUIs.
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    panel = SegmenterSwitcherPanel(viewer=None)
    panel.resize(360, 320)
    panel.show()

    # Only block when NOT in an IPython Qt loop.
    import builtins

    ip_fn = getattr(builtins, "get_ipython", None)
    in_qt_ipython = (
        ip_fn is not None
        and ip_fn() is not None
        and str(
            getattr(ip_fn(), "active_eventloop", "") or ""
        ).startswith("qt")
    )
    if not in_qt_ipython:
        app.exec_()
