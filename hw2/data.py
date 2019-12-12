from typing import Tuple


class ImageSize:
    def __init__(self, width: int, height: int):
        self.height = height
        self.channels_order = "channels_last"
        self.width = width

    def dimensions(self) -> Tuple[int, int]:
        return self.height, self.width

    def rgb_dimensions(self) -> Tuple[int, int, int]:
        return self.height, self.width, 3
