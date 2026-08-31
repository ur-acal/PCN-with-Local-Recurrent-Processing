import unittest

from baseline.run_tinyimagenet_wrn_aug_search import (
    ALLOWED_TUNED_FIELDS,
    ANCHOR,
    CANDIDATES,
    FIXED_RECIPE,
    candidate_config,
    encode_override,
)
from baseline.train_baseline_cifar import parse_kv_overrides


class TinyImageNetWrnAugSearchTests(unittest.TestCase):
    def test_search_is_bounded_and_augmentation_only(self):
        self.assertEqual(len(CANDIDATES), 12)
        for name, changes in CANDIDATES.items():
            self.assertLessEqual(set(changes), ALLOWED_TUNED_FIELDS, name)
            resolved = candidate_config(name)
            self.assertEqual(set(resolved), set(ANCHOR))

    def test_fixed_training_recipe_is_identical_for_every_trial(self):
        for name in CANDIDATES:
            parsed = parse_kv_overrides(encode_override(name))
            for key, value in FIXED_RECIPE.items():
                self.assertEqual(parsed[key], value, (name, key))

    def test_colon_separated_timm_geometry_is_parsed_as_tuples(self):
        parsed = parse_kv_overrides(
            "timm_train_scale=0.5:1.0,timm_train_ratio=0.75:1.3333333333333333"
        )
        self.assertEqual(parsed["timm_train_scale"], (0.5, 1.0))
        self.assertEqual(parsed["timm_train_ratio"], (0.75, 4.0 / 3.0))

    def test_none_disables_randaugment_without_changing_other_augmentations(self):
        parsed = parse_kv_overrides(encode_override("no_randaugment"))
        self.assertIsNone(parsed["auto_augment"])
        for key, value in ANCHOR.items():
            if key != "auto_augment":
                self.assertEqual(parsed[key], value)


if __name__ == "__main__":
    unittest.main()
