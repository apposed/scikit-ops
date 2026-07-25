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

import sys
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

def _load_commands():
    """Import and instantiate each segmenter command defensively.

    Returns three things:
    * ``order``     -- list of display names in a stable order.
    * ``available`` -- maps NAME -> command instance for commands whose
                       dependencies are installed.
    * ``missing``   -- maps NAME -> error message for commands that failed.
    """
    order: list[str] = []
    available: dict[str, object] = {}

    def _try(name, factory):
        try:
            command = factory()
            # Touch the auto-GUI now so missing deps surface here, not later.
            _ = command.gui
            order.append(command.NAME)
            available[command.NAME] = command
        except Exception as exc:  # noqa: BLE001 - report any failure
            order.append(name)

    def _otsu():
        from imgops.magicgui.SkImageSegmenterMagicGuiCommand import OtsuCommand
        return OtsuCommand()

    def _cellpose():
        from imgops.magicgui.CellposeMagicGuiCommand import CellposeCommand
        return CellposeCommand()

    def _stardist():
        from imgops.magicgui.StardistMagicGuiCommand import StardistCommand
        return StardistCommand()

    _try("Otsu", _otsu)
    _try("CellposeSAM", _cellpose)
    _try("StarDist", _stardist)

    return order, available 


class SegmenterSwitcherPanel(QWidget):
    """Combo-driven switcher between magicgui segmenter commands."""

    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer

        self.env = None

        # Load all commands; keep those that failed so we can still list them.
        self._order, self._commands = _load_commands()

        layout = QVBoxLayout(self)

        # Per-selection dependency status: green if OK, red if missing.
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(QLabel("Segmenter:"))
        self.combo = QComboBox()
        # Show every segmenter, even ones with missing dependencies.
        self.combo.addItems(self._order)
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

        self.env_btn = QPushButton("Choose Environment…")
        self.env_btn.clicked.connect(self._on_choose_env)
        self.env_btn.setVisible(False)
        layout.addWidget(self.env_btn)

        layout.addStretch()

        self._current_gui = None
        if self._order:
            self._on_segmenter_changed(self.combo.currentText())

    # -- switching ---------------------------------------------------------

    def _on_segmenter_changed(self, name: str):
        """Swap the parameter GUI to match the selected command."""
        # Remove the previous magicgui widget, if any.
        if self._current_gui is not None:
            self._param_layout.removeWidget(self._current_gui.native)
            self._current_gui.native.setParent(None)
            self._current_gui = None

        if name in self._commands:
            if self._commands[name].are_dependencies_available():
                # Dependencies present -> show the parameter GUI, green status.
                self.status_label.setText("Dependencies OK")
                self.status_label.setStyleSheet("color: green;")
                self.env_btn.setVisible(False)
            else:
                # Dependencies missing -> yellow status + env picker, Apply disabled.
                self.status_label.setText(
                    f"Dependencies missing: {name} will be run in a custom environment."
                )
                self.status_label.setStyleSheet("color: yellow;")
                self.env_btn.setVisible(True)
            command = self._commands[name]
            self._current_gui = command.gui  # magicgui Container
            self._param_layout.addWidget(self._current_gui.native)
            self.apply_btn.setEnabled(True)
            self.status_label.setToolTip("")
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

        if command.are_dependencies_available():
            mask = command.segment(image)
        else:
            from imgops.appose.execute_appose import execute_appose
            
            # user chose custom evvironment for this command
            if self.env is not None:
                env_path = self.env.path
            else:
                env_name = command.environment_name()
                repo_root = Path(__file__).resolve().parents[3]
                env_path = str(repo_root / "pixi" / env_name)
            
            if env_path is None:
                print("No remote environment configured. Use 'Choose Environment…'.")
                return
            
            mask = execute_appose(image, command, env_path)

        n = int(mask.max())
        print(f"  -> {n} objects")
        if self.viewer is not None:
            self.viewer.add_labels(mask, name=f"{self.combo.currentText()}")

    def _on_choose_env(self):
        """Open the RemoteEnvDialog for the currently selected command."""
        name = self.combo.currentText()
        from imgops.appose.remote_env_dialog import RemoteEnvDialog
        dlg = RemoteEnvDialog(segmenter_class_name=name, parent=self)
        if dlg.exec_():
            self.env = dlg.selected_environment
            if self.env is not None:
                self.status_label.setText(
                    f"Remote env set: {self.env.name}  ({self.env.path})"
                )
                self.status_label.setStyleSheet("color: orange;")
                self.apply_btn.setEnabled(True)
                print(f"Pinned env '{self.env.name}' for {name}")


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
