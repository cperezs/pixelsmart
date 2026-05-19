"""Sample autolabeling plugin — divides the image into four horizontal stripes."""

import numpy as np
from application.plugin_base import AutolabelPlugin


class SamplePlugin(AutolabelPlugin):
    """Assigns four equal horizontal stripes to background/staff/notes/lyrics.

    If the image height is not divisible by 4 the remaining rows are
    assigned to the bottom stripe (lyrics).
    """

    @property
    def id(self):
        return "sample"

    @property
    def display_name(self):
        return "Sample (4 Stripes)"

    @property
    def supported_layers(self):
        return ["background", "staff", "notes", "lyrics"]

    def run(self, image):
        h, w = image.shape[:2]
        label_map = np.zeros((h, w), dtype=np.uint8)

        stripe_h = h // 4

        # 0 = background  (top)
        # 1 = staff
        # 2 = notes
        # 3 = lyrics       (bottom, receives remainder rows)
        label_map[0:stripe_h, :] = 0
        label_map[stripe_h : 2 * stripe_h, :] = 1
        label_map[2 * stripe_h : 3 * stripe_h, :] = 2
        label_map[3 * stripe_h :, :] = 3

        return label_map
