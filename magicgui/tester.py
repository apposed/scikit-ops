"""Run in IPython with `%gui qt` first, then `%run tester.py`."""

from magicgui.experimental import guiclass


@guiclass
class MyDataclass:
    a: int = 0
    b: str = 'hello'
    c: bool = True


obj = MyDataclass()
obj.gui.show()