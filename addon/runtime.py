# SPDX-License-Identifier: GPL-3.0-or-later
"""Cooperative cancellation shared by CPU and GPU paths."""
class RenderCancelled(Exception):
    pass


def checkpoint(cancel):
    if cancel is not None and cancel():
        raise RenderCancelled()


def read_framebuffer(framebuffer, width, height, *, cancel=None, progress=None):
    """Read bounded strips; GPU read_color returns a shaped Buffer, not a flat list."""
    pixels = []
    for y in range(0, height, 32):
        checkpoint(cancel)
        rows = min(32, height - y)
        with framebuffer.bind():
            buf = framebuffer.read_color(0, y, width, rows, 4, 0, "FLOAT")
        buf.dimensions = (width * rows, 4)
        pixels.extend(buf.to_list())
        if progress:
            progress((y + rows) / height)
    checkpoint(cancel)
    return pixels
