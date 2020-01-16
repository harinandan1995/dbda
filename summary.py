from models.generator_v2 import Generator
from models.corner_detector import CornerDetector

generator = Generator(1, [3, 10], 64, 12, width=256, height=256)
corner_detector = CornerDetector(1, 13, width=256, height=256)


print(corner_detector.summary())
print(generator.summary())
