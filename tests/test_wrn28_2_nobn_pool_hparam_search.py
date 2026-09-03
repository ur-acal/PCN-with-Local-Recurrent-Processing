import unittest

from baseline.run_wrn28_2_nobn_pool_hparam_search import (
    AUGMENTATIONS,
    CURRENT_RECIPES,
    FULL_SEEDS,
    STUDY_PLANS,
    candidate_rows,
    encode_override,
    make_trial,
    promoted_candidates,
    resolved_recipe,
    stage_trials,
    winning_candidate,
)


def record(trial, status="completed", accuracy=0.5):
    return {
        **{key: str(value) for key, value in trial.items()},
        "status": status,
        "best_accuracy": str(accuracy),
    }


class PoolHyperparameterSearchTests(unittest.TestCase):
    def test_study_budgets(self):
        self.assertEqual(STUDY_PLANS[("cifar100", "avgpool_main")]["candidate_count"], 16)
        self.assertEqual(STUDY_PLANS[("cifar10", "stride2_main")]["candidate_count"], 16)
        self.assertEqual(STUDY_PLANS[("cifar10", "avgpool_main")]["candidate_count"], 8)
        self.assertEqual(STUDY_PLANS[("cifar100", "stride2_main")]["candidate_count"], 8)

    def test_joint_design_covers_requested_values(self):
        rows = candidate_rows("cifar100", "avgpool_main")
        self.assertEqual(len(rows), 16)
        self.assertEqual({row["lr"] for row in rows.values()}, {0.05, 0.1, 0.15})
        self.assertEqual({row["max_norm"] for row in rows.values()}, {None, 1.0, 2.0})
        self.assertEqual({row["dropout_rate"] for row in rows.values()}, {0.0, 0.1})
        self.assertEqual({row["bias_lr_multiplier"] for row in rows.values()}, {0.5, 1.0})
        self.assertEqual({row["bias_weight_decay"] for row in rows.values()}, {0.0, None})
        self.assertEqual({row["augmentation"] for row in rows.values()}, set(AUGMENTATIONS))
        self.assertEqual({row["weight_decay"] for row in rows.values()}, {1e-4, 5e-4, 1e-3, 2e-3})
        self.assertEqual({row["final_dropout_rate"] for row in rows.values()}, {0.0, 0.1, 0.25, 0.4})

    def test_candidate_one_is_exact_original_recipe(self):
        for dataset, study in STUDY_PLANS:
            row = candidate_rows(dataset, study)["c01"]
            expected = CURRENT_RECIPES[dataset]
            actual = (
                row["lr"], row["max_norm"], row["dropout_rate"],
                row["bias_lr_multiplier"], row["bias_weight_decay"],
                row["augmentation"], row["weight_decay"],
                row["final_dropout_rate"],
            )
            self.assertEqual(actual, expected)
            self.assertEqual(row["label"], "original_recipe")

    def test_recipe_enables_monitor_without_changing_fixed_training_controls(self):
        recipe = resolved_recipe("cifar100", "avgpool_main", "c02", 150)
        self.assertTrue(recipe["collapse_monitor_enabled"])
        self.assertEqual(recipe["batch_size"], 128)
        self.assertEqual(recipe["timm_sched"], "cosine")
        self.assertEqual(recipe["num_epochs"], 150)
        encoded = encode_override("cifar100", "avgpool_main", "c02", 150)
        self.assertIn("collapse_monitor_enabled=true", encoded)

    def test_screen_budget_uses_all_required_seeds(self):
        high = stage_trials({}, "cifar100", "avgpool_main", "screen")
        lower = stage_trials({}, "cifar100", "stride2_main", "screen")
        self.assertEqual(len(high), 16 * 3)
        self.assertEqual(len(lower), 8 * 2)

    def test_promotion_requires_every_seed_and_ranks_mean_then_worst(self):
        dataset, study = "cifar100", "avgpool_main"
        seeds = STUDY_PLANS[(dataset, study)]["screen_seeds"]
        records = {}
        scores = {
            "c01": (0.70, 0.70, 0.70),
            "c02": (0.80, 0.80, 0.50),
            "c03": (0.72, 0.72, 0.72),
            "c04": (0.99, 0.99, 0.99),
        }
        for candidate, accuracies in scores.items():
            for seed, accuracy in zip(seeds, accuracies):
                trial = make_trial(dataset, study, "screen", candidate, seed, 150)
                status = "collapsed" if candidate == "c04" and seed == seeds[-1] else "completed"
                records[trial["trial_id"]] = record(trial, status, accuracy)
        self.assertEqual(promoted_candidates(records, dataset, study), ["c03", "c01", "c02"])

    def test_full_winner_requires_five_successful_seeds(self):
        dataset, study = "cifar10", "avgpool_main"
        records = {}
        for candidate, accuracy in (("c01", 0.90), ("c02", 0.92)):
            for seed in FULL_SEEDS:
                trial = make_trial(dataset, study, "full", candidate, seed, 300)
                status = "not_learned" if candidate == "c02" and seed == FULL_SEEDS[-1] else "completed"
                records[trial["trial_id"]] = record(trial, status, accuracy)
        self.assertEqual(winning_candidate(records, dataset, study), "c01")


if __name__ == "__main__":
    unittest.main()
