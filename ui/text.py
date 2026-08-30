"""Anti-aliased TrueType text for the operator panel.

OpenCV can only draw its built-in Hershey stroke fonts, which were designed for
a pen plotter and look like it. Pillow renders real TrueType, but calling it
every frame costs about 8 ms - roughly twenty times what the entire panel costs
today, and more than half the cost of hand detection.

So each string is rendered once into an alpha mask and kept. The panel's text
comes from a small fixed vocabulary - state names, action labels, colour names,
a handful of hints - so the cache fills within the first second and after that
essentially never misses. Warm, a full panel of text blits in ~0.4 ms; a string
appearing for the first time costs ~1.1 ms, once.

Coordinates follow cv2.putText: `org` is the left end of the text baseline.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))

# Bundled first so the panel looks the same on every machine, then whatever the
# system has, then Pillow's bitmap font so a missing file degrades instead of
# crashing the operator UI.
_FONT_FILES = {
    False: (
        os.path.join(_HERE, "assets", "fonts", "NotoSans-Regular.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ),
    True: (
        os.path.join(_HERE, "assets", "fonts", "NotoSans-Bold.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
}

# A status line carrying a changing number would otherwise grow the cache without
# limit. The vocabulary is far smaller than this, so the bound is never reached
# in normal use.
_MAX_ENTRIES = 512

_fonts = {}
_sprites = {}
_widths = {}


def _font(size, bold):
    cached = _fonts.get((size, bold))
    if cached is not None:
        return cached
    for path in _FONT_FILES[bold]:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size)
            break
    else:
        font = ImageFont.load_default()
    _fonts[(size, bold)] = font
    return font


def _sprite(string, size, bold, color):
    """Alpha mask plus premultiplied colour for one string, rendered once."""
    key = (string, size, bold, color)
    cached = _sprites.get(key)
    if cached is not None:
        return cached

    font = _font(size, bold)
    left, top, right, bottom = font.getbbox(string, anchor="ls")
    width, height = max(1, right - left), max(1, bottom - top)

    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).text((-left, -top), string, font=font, fill=255, anchor="ls")

    alpha = (np.asarray(mask, dtype=np.float32) / 255.0)[:, :, None]
    premultiplied = alpha * np.asarray(color, dtype=np.float32)

    if len(_sprites) >= _MAX_ENTRIES:
        _sprites.clear()
    _sprites[key] = (left, top, alpha, premultiplied)
    return _sprites[key]


def text(panel, string, org, size=12, color=(230, 230, 230), bold=False):
    """Blend one string onto the panel, its baseline starting at `org`."""
    if not string:
        return
    left, top, alpha, premultiplied = _sprite(string, size, bold, tuple(color))

    x, y = int(org[0]) + left, int(org[1]) + top
    height, width = alpha.shape[:2]
    panel_height, panel_width = panel.shape[:2]

    # Clip against the panel edges rather than dropping the string, so a long
    # status message is truncated instead of vanishing.
    src_x, src_y = max(0, -x), max(0, -y)
    dst_x, dst_y = max(0, x), max(0, y)
    span_w = min(width - src_x, panel_width - dst_x)
    span_h = min(height - src_y, panel_height - dst_y)
    if span_w <= 0 or span_h <= 0:
        return

    region = panel[dst_y:dst_y + span_h, dst_x:dst_x + span_w]
    a = alpha[src_y:src_y + span_h, src_x:src_x + span_w]
    c = premultiplied[src_y:src_y + span_h, src_x:src_x + span_w]
    region[:] = (region * (1.0 - a) + c).astype(np.uint8)


def text_width(string, size=12, bold=False):
    """Advance width in pixels, for centring and right-alignment.

    Cached for the same reason the sprites are: Pillow shapes the string to
    measure it, which costs about as much as drawing it. Measured uncached this
    was 0.22 ms a call and the single most expensive thing on the panel.
    """
    key = (string, size, bold)
    width = _widths.get(key)
    if width is None:
        if len(_widths) >= _MAX_ENTRIES:
            _widths.clear()
        width = _widths[key] = int(round(_font(size, bold).getlength(string)))
    return width
