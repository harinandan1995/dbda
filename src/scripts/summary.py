from src.models.generator import Generator
from src.models.corner_detector import CornerDetector


def model_summary():
    generator = Generator(1, [3, 10], 64, 14, width=256, height=256)
    corner_detector = CornerDetector(3, 17, width=256, height=256)

    print(corner_detector.summary())
    print(generator.summary())

if __name__ == '__main__':
    model_summary()