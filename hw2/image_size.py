class ImageSize:
    def __init__(self, width, height):
        self.height = height
        self.channels_order = "channels_last"
        self.width = width

    def dimensions(self):
        return self.height, self.width

    def rgb_dimensions(self):
        return self.height, self.width, 3
