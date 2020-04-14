from models.generator import Generator
from models.corner_detector import CornerDetector

generator = Generator(1, [3, 10], 64, 14, width=256, height=256)
corner_detector = CornerDetector(3, 17, width=256, height=256)

print(corner_detector.summary())
print(generator.summary())
