"""Synthetic checks for plateau decisions; no training or process termination."""
import unittest

from .early_stop_policy import assess


def history(end, values=None):
    values = {1:1., 10:.8, 20:.6, 30:.4, 40:.2, **(values or {})}
    best = reference = float('inf')
    stale = 0
    rows = []
    for epoch in range(1, end+1):
        run = epoch == 1 or epoch % 10 == 0
        if run:
            value = values.get(epoch, .3)
            validation = {'uniform': {'loss_per_pixel': value}}
            best = min(best, value)
            if value < reference*.998:
                reference, stale = value, 0
            else:
                stale += 1
        rows.append(dict(epoch=epoch-1, epochs_completed=epoch,
                         validation_was_run=run, validation=validation,
                         stale_checks=stale, best_validation_per_pixel=best))
    return rows


class PlateauDecisions(unittest.TestCase):
    def test_current_three_checks_do_not_stop(self):
        result = assess(history(71))
        self.assertEqual(result['stale_checks'], 3)
        self.assertFalse(result['eligible'])

    def test_stops_at_first_eligible_complete_epoch(self):
        self.assertFalse(assess(history(99))['eligible'])
        result = assess(history(100))
        self.assertTrue(result['eligible'])
        self.assertEqual(result['stale_checks'], 6)
        self.assertEqual(result['best_validation_epoch'], 40)
        self.assertFalse(result['stop_applied'])
        self.assertFalse(result['training_complete'])

    def test_improvement_restarts_wait(self):
        result = assess(history(100, {80:.1}))
        self.assertFalse(result['eligible'])
        self.assertEqual(result['stale_checks'], 2)
        self.assertEqual(result['best_validation_epoch'], 80)

    def test_small_best_improvement_does_not_reset_plateau(self):
        result = assess(history(100, {80:.1998}))
        self.assertTrue(result['eligible'])
        self.assertEqual(result['best_validation_epoch'], 80)
        self.assertEqual(result['plateau_reference'], .2)

    def test_carried_validation_not_counted(self):
        result = assess(history(79))
        self.assertEqual(result['stale_checks'], 3)
        self.assertEqual(len(result['validation_checks']), 8)

    def test_minimum_epoch_applies_even_if_plateau_longer(self):
        self.assertFalse(assess(history(100), minimum_epochs=110)['eligible'])

    def test_missing_history_rejected(self):
        rows = history(100)
        del rows[50]
        with self.assertRaises(ValueError):
            assess(rows)

    def test_mislabeled_validation_rejected(self):
        rows = history(100)
        rows[98]['validation_was_run'] = True
        with self.assertRaises(ValueError):
            assess(rows)

    def test_tampered_counter_rejected(self):
        rows = history(100)
        rows[-1]['stale_checks'] = 8
        with self.assertRaises(ValueError):
            assess(rows)


if __name__ == '__main__':
    unittest.main()
