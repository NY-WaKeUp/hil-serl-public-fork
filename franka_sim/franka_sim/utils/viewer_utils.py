import os
import time

import glfw
import mujoco
import mujoco.viewer
from Xlib import X, display
from Xlib.protocol import event


def _set_fixed_camera(handle, model, name):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    handle.cam.fixedcamid = cam_id


def _client_ids(dpy):
    root = dpy.screen().root
    atom = dpy.intern_atom("_NET_CLIENT_LIST")
    prop = root.get_full_property(atom, X.AnyPropertyType)
    return list(prop.value)


def _window_title(dpy, win):
    utf8 = dpy.intern_atom("UTF8_STRING")
    net_name = dpy.intern_atom("_NET_WM_NAME")
    prop = win.get_full_property(net_name, utf8)
    if prop is not None:
        return prop.value.decode("utf-8")
    name = win.get_wm_name()
    if name is None:
        return ""
    return name


def _mujoco_windows(dpy):
    pid_atom = dpy.intern_atom("_NET_WM_PID")
    pid = os.getpid()
    wins = []
    for wid in _client_ids(dpy):
        win = dpy.create_resource_object("window", wid)
        pid_prop = win.get_full_property(pid_atom, X.AnyPropertyType)
        if pid_prop is None or int(pid_prop.value[0]) != pid:
            continue
        if "MuJoCo" not in _window_title(dpy, win):
            continue
        wins.append(win)
    wins.sort(key=lambda item: item.id)
    return wins


def _wait_mujoco_windows(dpy, n):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        wins = _mujoco_windows(dpy)
        if len(wins) >= n:
            return wins[:n]
        time.sleep(0.02)
    raise RuntimeError(f"expected {n} MuJoCo X11 windows")


def _unmaximize(dpy, win):
    root = dpy.screen().root
    wm_state = dpy.intern_atom("_NET_WM_STATE")
    max_v = dpy.intern_atom("_NET_WM_STATE_MAXIMIZED_VERT")
    max_h = dpy.intern_atom("_NET_WM_STATE_MAXIMIZED_HORZ")
    ev = event.ClientMessage(
        window=win,
        client_type=wm_state,
        data=(32, (0, max_v, max_h, 1, 0)),
    )
    mask = X.SubstructureRedirectMask | X.SubstructureNotifyMask
    root.send_event(ev, event_mask=mask)


def _ewmh_place(dpy, win, x, y, w, h):
    # WMs ignore XConfigureWindow on unmapped GLFW windows; EWMH after map.
    _unmaximize(dpy, win)
    root = dpy.screen().root
    atom = dpy.intern_atom("_NET_MOVERESIZE_WINDOW")
    gravity = 1
    source = 1
    flags = gravity | (source << 12) | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11)
    ev = event.ClientMessage(
        window=win,
        client_type=atom,
        data=(32, (flags, x, y, w, h)),
    )
    mask = X.SubstructureRedirectMask | X.SubstructureNotifyMask
    root.send_event(ev, event_mask=mask)
    win.configure(x=x, y=y, width=w, height=h)
    dpy.flush()


def _tile_left_half(dpy, top_win, bot_win):
    screen = dpy.screen()
    half_w = screen.width_in_pixels // 2
    half_h = screen.height_in_pixels // 2
    for _ in range(25):
        _ewmh_place(dpy, top_win, 0, 0, half_w, half_h)
        _ewmh_place(dpy, bot_win, 0, half_h, half_w, half_h)
        dpy.sync()
        top_g = top_win.get_geometry()
        bot_g = bot_win.get_geometry()
        if top_g.width == half_w and bot_g.width == half_w and bot_g.height == half_h:
            return
        time.sleep(0.04)


class DualMujocoViewer:
    def __init__(self, model, data, key_callback=None):
        self.model = model
        self.data = data
        self.viewer_1 = None
        self.viewer_2 = None
        self.key_callback = key_callback
        self._dpy = None
        self._tile_left = 0
        self._top_win = None
        self._bot_win = None

    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def launch(self):
        self._dpy = display.Display()
        self.viewer_1 = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback,
        )
        self.viewer_2 = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback,
        )
        top_win, bot_win = _wait_mujoco_windows(self._dpy, 2)
        _tile_left_half(self._dpy, top_win, bot_win)
        self._top_win = top_win
        self._bot_win = bot_win
        self._tile_left = 15
        _set_fixed_camera(self.viewer_1, self.model, "front")
        _set_fixed_camera(self.viewer_2, self.model, "handcam_rgb")

    def is_running(self):
        return self.viewer_1.is_running() and self.viewer_2.is_running()

    def sync(self):
        _set_fixed_camera(self.viewer_1, self.model, "front")
        _set_fixed_camera(self.viewer_2, self.model, "handcam_rgb")
        self.viewer_1.sync()
        self.viewer_2.sync()
        if self._tile_left > 0:
            _tile_left_half(self._dpy, self._top_win, self._bot_win)
            self._tile_left -= 1

    def close(self):
        if self.viewer_1:
            self.viewer_1.close()
        if self.viewer_2:
            self.viewer_2.close()
        if self._dpy is not None:
            self._dpy.close()
            self._dpy = None
        glfw.terminate()
