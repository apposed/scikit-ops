"""
Interactive test: launch napari with the SegmenterSwitcherPanel.

Starts a napari viewer, loads a sample image, and docks the
SegmenterSwitcherPanel so you can switch between the Otsu and CellposeSAM
magicgui commands and hit Apply.

Run from this folder so the sibling imports resolve, e.g.:

    cd experiments/magicgui
    python interactive_test.py

Use a pixi env that has napari + magicgui (and cellpose if you want the
CellposeSAM Apply to actually run), e.g.:

    ../../pixi/microsam_cellposesam_czi/.pixi/envs/default/bin/python interactive_test.py
"""

import os
import sys

import napari
from skimage import data

# Ensure sibling modules (SegmenterSwitcherPanel, *Command) are importable
# regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imgops.magicgui.SegmenterSwitcherPanel import SegmenterSwitcherPanel  # noqa: E402


def main():
    viewer = napari.Viewer()

    # skimage.data.cells3d() is ZCYX; grab a single 2D nuclei slice so the
    # 2D-only commands (Otsu, CellposeSAM) work directly.
    cells = data.cells3d()  # (Z, C, Y, X)
    nuclei = cells[cells.shape[0] // 2, 1]  # mid Z, nuclei channel
    viewer.add_image(nuclei, name="cells (nuclei)")

    viewer.window.add_dock_widget(
        SegmenterSwitcherPanel(viewer),
        name="Segmenter Switcher",
        area="right",
    )

    print("✨ napari launched with SegmenterSwitcherPanel")
    print("   - pick a segmenter in the combo (GUI updates automatically)")
    print("   - click Apply to segment the active image layer")

    napari.run()


if __name__ == "__main__":
    main()
