"""
Dialog for picking / registering a remote Python environment for a segmenter.

Small Qt wrapper around
:class:`napari_ai_lab.Segmenters.execute_appose.RemoteEnvironmentRegistry`.

Usage::

    from napari_ai_lab.apps.remote_env_dialog import RemoteEnvDialog
    dlg = RemoteEnvDialog(segmenter_class_name="CellposeSegmenter", parent=self)
    if dlg.exec_():
        env = dlg.selected_environment  # RemoteEnvironment (also pinned in registry)
"""

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

try:
    # When used inside the napari-ai-lab package
    from ..Segmenters.execute_appose import RemoteEnvironment, get_registry
except ImportError:
    # Standalone / script use from the appose/ folder
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from execute_appose import RemoteEnvironment, get_registry


class RemoteEnvDialog(QDialog):
    """Pick an existing env or register a new one, then pin it for a segmenter.

    On accept, the selected environment is stored on
    ``self.selected_environment`` and (if ``segmenter_class_name`` was
    provided) pinned to that segmenter in the registry.
    """

    def __init__(self, segmenter_class_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Remote Environment")
        self.segmenter_class_name = segmenter_class_name
        self.selected_environment: RemoteEnvironment | None = None

        self._registry = get_registry()

        outer = QVBoxLayout(self)

        if segmenter_class_name:
            outer.addWidget(
                QLabel(
                    f"Select an environment to run "
                    f"<b>{segmenter_class_name}</b> in:"
                )
            )
        else:
            outer.addWidget(QLabel("Select or register a remote environment:"))

        # -- Existing envs ---------------------------------------------------
        self.combo = QComboBox()
        self._refresh_combo()
        outer.addWidget(self.combo)

        # -- New env form ----------------------------------------------------
        form_wrap = QVBoxLayout()
        form_wrap.addWidget(QLabel("<b>Or register a new environment:</b>"))
        form = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText('e.g. "cellpose-env"')
        form.addRow("Name:", self.name_edit)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path to pixi/conda env root")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_btn)
        form.addRow("Path:", path_row)

        self.notes_edit = QLineEdit()
        form.addRow("Notes:", self.notes_edit)

        add_btn = QPushButton("Add / Update")
        add_btn.clicked.connect(self._on_add)
        form.addRow("", add_btn)

        form_wrap.addLayout(form)
        outer.addLayout(form_wrap)

        # -- OK / Cancel -----------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # Pre-select the currently pinned env for this segmenter, if any.
        if segmenter_class_name:
            pinned = self._registry.resolve(segmenter_class_name)
            if pinned is not None:
                idx = self.combo.findText(pinned.name)
                if idx >= 0:
                    self.combo.setCurrentIndex(idx)

    # -- helpers -------------------------------------------------------------

    def _refresh_combo(self):
        self.combo.clear()
        envs = self._registry.list()
        if not envs:
            self.combo.addItem("(no environments registered yet)")
            self.combo.setEnabled(False)
        else:
            self.combo.setEnabled(True)
            for env in envs:
                self.combo.addItem(env.name)

    def _on_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Environment Root Directory"
        )
        if directory:
            self.path_edit.setText(directory)

    def _on_add(self):
        name = self.name_edit.text().strip()
        path = self.path_edit.text().strip()
        if not name or not path:
            QMessageBox.warning(
                self, "Missing fields", "Name and path are both required."
            )
            return
        env = RemoteEnvironment(
            name=name, path=path, notes=self.notes_edit.text().strip()
        )
        self._registry.add(env)
        self._refresh_combo()
        idx = self.combo.findText(name)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def _on_accept(self):
        if not self.combo.isEnabled():
            QMessageBox.warning(
                self,
                "No environment selected",
                "Register an environment first, or cancel.",
            )
            return
        env_name = self.combo.currentText()
        env = self._registry.get(env_name)
        if env is None:
            QMessageBox.warning(
                self, "Unknown environment", f"No env named {env_name!r}."
            )
            return
        self.selected_environment = env
        if self.segmenter_class_name:
            self._registry.pin(self.segmenter_class_name, env.name)
        self.accept()
