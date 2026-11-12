"""Проверки реализации; статистическое соответствие распределению проверяется отдельно."""
import unittest
import numpy as np
from rng_protein import ParkMiller, XorShift64Star, ProteinRNG


class GeneratorTests(unittest.TestCase):
    def test_park_miller_reference(self):
        # Первые состояния считаются напрямую по рекуррентной формуле при seed=1.
        actual = ParkMiller(1).uniform(5)
        expected = np.array([16807, 282475249, 1622650073,
                             984943658, 1144108930]) / (2**31 - 1)
        np.testing.assert_array_equal(actual, expected)

    def test_uniform_open_interval(self):
        for engine in (ParkMiller, XorShift64Star):
            with self.subTest(engine=engine.__name__):
                u = engine(123).uniform(10000)
                self.assertEqual(u.shape, (10000,))
                self.assertTrue(np.all(np.isfinite(u) & (u > 0) & (u < 1)))

    def test_same_seed_reproduces_mixture(self):
        for engine in ('lcg', 'xorshift'):
            with self.subTest(engine=engine):
                a = ProteinRNG(123, engine).mixture(101, [.8, .2], [1., 2.], [.3, .2])
                b = ProteinRNG(123, engine).mixture(101, [.8, .2], [1., 2.], [.3, .2])
                np.testing.assert_array_equal(a, b)

    def test_normal_chunking_preserves_spare(self):
        # Нечётные запросы оставляют второе число пары Бокса—Мюллера в кэше.
        # Нулевой запрос между ними не должен менять поток.
        for engine in ('lcg', 'xorshift'):
            with self.subTest(engine=engine):
                g = ProteinRNG(123, engine)
                chunks = np.concatenate([g.normal(k) for k in (1, 0, 3, 5, 2)])
                whole = ProteinRNG(123, engine).normal(11)
                np.testing.assert_array_equal(chunks, whole)

    def test_box_muller_known_pair(self):
        g = ProteinRNG(123)
        # Подставляем заданные U: радиус равен 1, угол равен pi/2.
        g.uniform = lambda size: np.array([np.exp(-.5), .25])
        np.testing.assert_allclose(g.normal(2), [0., 1.], atol=1e-14)

    def test_empty_outputs(self):
        g = ProteinRNG(123)
        for sample in (g.uniform(0), g.normal(0), g.lognormal(1., .3, 0)):
            self.assertEqual(sample.shape, (0,))
        values, labels = g.mixture(0, [1.], [1.], [.3], return_labels=True)
        self.assertEqual(values.shape, (0,))
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(g.last_proposed, 0)
        self.assertEqual(g.last_rejected, 0)

    def test_sizes_labels_and_positive_values(self):
        for size in (1, 7, 1000):
            with self.subTest(size=size):
                values, labels = ProteinRNG(123).mixture(
                    size, [.8, .2], [1., 2.], [.3, .2], return_labels=True)
                self.assertEqual(values.shape, (size,))
                self.assertEqual(labels.shape, (size,))
                self.assertTrue(np.all(np.isfinite(values) & (values > 0)))
                self.assertTrue(np.all((labels == 0) | (labels == 1)))

    def test_rejection_bounds_and_counters(self):
        # Намеренно узкий интервал заставляет реально выполнить отбор.
        g = ProteinRNG(123)
        values = g.mixture(1000, [1.], [0.], [1.], lower=.5, upper=1.5)
        self.assertTrue(np.all((values >= .5) & (values <= 1.5)))
        self.assertGreater(g.last_rejected, 0)
        self.assertEqual(g.last_proposed, len(values) + g.last_rejected)
        values = g.mixture(1000, [.8, .2], [1., 2.], [.3, .2], upper=25.)
        self.assertTrue(np.all((values > 0) & (values <= 25.)))
        self.assertEqual(g.last_proposed, len(values) + g.last_rejected)

    def test_invalid_parameters(self):
        # Проверяем тип ошибки, а не случайное совпадение текста сообщения.
        cases = [dict(size=-1), dict(weights=[.2, .2]),
                 dict(weights=[-1., 2.]), dict(mu=[1.]),
                 dict(sigma=[0., .2]), dict(mu=[np.nan, 2.]),
                 dict(lower=2., upper=1.)]
        for override in cases:
            with self.subTest(parameters=override):
                args = dict(size=10, weights=[.8, .2], mu=[1., 2.], sigma=[.3, .2])
                args.update(override)
                with self.assertRaises(ValueError):
                    ProteinRNG(123).mixture(**args)
        for engine in (ParkMiller, XorShift64Star):
            with self.assertRaises(ValueError):
                engine(0)
        with self.assertRaises(ValueError):
            ProteinRNG(123, engine='unknown')
        with self.assertRaises(ValueError):
            ProteinRNG(123).normal(-1)
        with self.assertRaises(ValueError):
            ProteinRNG(123).lognormal(0., 0., 10)

    def test_rejection_attempt_limit(self):
        g = ProteinRNG(123)
        # Фиксированные предложения exp(0)=1 не проходят интервал [2, 3].
        g.normal = lambda size: np.zeros(size)
        with self.assertRaises(RuntimeError):
            g.mixture(5, [1.], [0.], [1.], lower=2., upper=3., max_proposals=10)
        self.assertEqual(g.last_proposed, 10)
        self.assertEqual(g.last_rejected, 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
