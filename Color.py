class Color:
    def __init__(self, rgb, freq):
        self.rgb = rgb
        self.freq = freq

    def __str__(self):
        return str(self.__dict__)

    def rgb_to_hex(self):
        return "#{:02x}{:02x}{:02x}".format(self.rgb[0], self.rgb[1], self.rgb[2]).upper()
