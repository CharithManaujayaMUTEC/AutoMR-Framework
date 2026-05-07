
from automr.core.tester import MRTester

from automr.transforms.geometric import flip
from automr.transforms.brightness import increase_brightness
from automr.transforms.translation import shift_right
from automr.transforms.noise import add_noise
from automr.transforms.crop import crop_top

from automr.relations.flip_relation import FlipRelation
from automr.relations.brightness_relation import BrightnessRelation
from automr.relations.translation_relation import TranslationRelation
from automr.relations.noise_relation import NoiseRelation
from automr.relations.crop_relation import CropRelation


class AutoMR:

    def __init__(self, model):
        self.model = model
        self.tester = MRTester()

        self.transforms = [
            flip,
            increase_brightness,
            shift_right,
            add_noise,
            crop_top
        ]

        self.relations = [
            FlipRelation(),
            BrightnessRelation(),
            TranslationRelation(),
            NoiseRelation(),
            CropRelation()
        ]

    def test(self, input_data):
        return self.tester.run_all(
            self.model,
            input_data,
            self.transforms,
            self.relations
        )
