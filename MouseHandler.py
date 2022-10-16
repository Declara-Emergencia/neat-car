class MouseStateHandler(dict):
    """Simple handler that tracks the state of buttons from the mouse. If a
    button is pressed then this handler holds a True value for it.
    For example::
        >>> win = window.Window
        >>> mousebuttons = mouse.MouseStateHandler()
        >>> win.push_handlers(mousebuttons)
        # Hold down the "left" button...
        >>> mousebuttons[mouse.LEFT]
        True
        >>> mousebuttons[mouse.RIGHT]
        False
    """

    def __init__(self):
        self["x"] = 0
        self["y"] = 0

    def on_mouse_press(self, x, y, button, modifiers):
        self[button] = True

    def on_mouse_release(self, x, y, button, modifiers):
        self[button] = False

    def on_mouse_motion(self, x, y, dx, dy):
        self["x"] = x
        self["y"] = y

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self["x"] = x
        self["y"] = y

    def __getitem__(self, key):
        return self.get(key, False)