
import unittest
import numpy as np
from rng_protein import ParkMiller, XorShift64Star, ProteinRNG


class GeneratorTests(unittest.TestCase):
    def test_park_miller_reference(self):
        """Первые пять значений Park–Miller совпадают с контрольными значениями."""
        # Контрольные состояния заранее рассчитаны при seed=1.
        # Это позволяет заметить неверный множитель, модуль или порядок обновления.
        actual = ParkMiller(1).uniform(5)
        expected = np.array([16807, 282475249, 1622650073,
                             984943658, 1144108930]) / (2**31 - 1)
        np.testing.assert_array_equal(actual, expected)

    def test_uniform_open_interval(self):
        """Оба датчика возвращают конечные числа строго между 0 и 1."""
        # Ноль нельзя передавать в ln(U), а единица — конец заявленного интервала.
        # Эта проверка границ не является статистическим тестом равномерности.
        for engine in (ParkMiller, XorShift64Star):
            with self.subTest(engine=engine.__name__):
                u = engine(123).uniform(10000)
                self.assertEqual(u.shape, (10000,))
                self.assertTrue(np.all(np.isfinite(u) & (u > 0) & (u < 1)))

    def test_same_seed_reproduces_mixture(self):
        """Одинаковые начальные условия воспроизводят одну и ту же выборку."""
        # Создаём два независимых экземпляра одного алгоритма с одинаковым seed.
        # Сравниваем их между собой, а не последовательности разных алгоритмов.
        for engine in ('lcg', 'xorshift'):
            with self.subTest(engine=engine):
                a = ProteinRNG(123, engine).mixture(101, [.8, .2], [1., 2.], [.3, .2])
                b = ProteinRNG(123, engine).mixture(101, [.8, .2], [1., 2.], [.3, .2])
                np.testing.assert_array_equal(a, b)

    def test_normal_chunking_preserves_spare(self):
        """Разбиение запроса normal на части не теряет числа из нормальных пар."""
        # Нечётные запросы оставляют второе число пары Бокса—Мюллера в кэше.
        # Нулевой запрос между ними не должен менять поток.
        for engine in ('lcg', 'xorshift'):
            with self.subTest(engine=engine):
                g = ProteinRNG(123, engine)
                chunks = np.concatenate([g.normal(k) for k in (1, 0, 3, 5, 2)])
                whole = ProteinRNG(123, engine).normal(11)
                np.testing.assert_array_equal(chunks, whole)

    def test_box_muller_known_pair(self):
        """Преобразование заданной пары U даёт заранее известную пару Z."""
        g = ProteinRNG(123)
        # Подставляем заданные U: радиус равен 1, угол равен pi/2.
        # Поэтому ожидаем Z=(0, 1); малый допуск учитывает округление cos(pi/2).
        g.uniform = lambda size: np.array([np.exp(-.5), .25])
        np.testing.assert_allclose(g.normal(2), [0., 1.], atol=1e-14)

    def test_empty_outputs(self):
        """Нулевой размер допустим для всех способов генерации и обоих датчиков."""
        # Пустой запрос не должен выдавать фиктивные значения или запускать отбор.
        for engine in ('lcg', 'xorshift'):
            with self.subTest(engine=engine):
                g = ProteinRNG(123, engine)
                for sample in (g.uniform(0), g.normal(0), g.lognormal(1., .3, 0)):
                    self.assertEqual(sample.shape, (0,))
                values, labels = g.mixture(0, [1.], [1.], [.3], return_labels=True)
                self.assertEqual(values.shape, (0,))
                self.assertEqual(labels.shape, (0,))
                self.assertEqual(g.last_proposed, 0)
                self.assertEqual(g.last_rejected, 0)

    def test_sizes_labels_and_positive_values(self):
        """Смесь возвращает нужное число положительных значений и допустимых меток."""
        # Нечётный размер дополнительно задействует работу с нормальными парами.
        # Метки здесь — индексы компонент (0 и 1), а не реальные типы йогуртов.
        for engine in ('lcg', 'xorshift'):
            for size in (1, 7, 1000):
                with self.subTest(engine=engine, size=size):
                    values, labels = ProteinRNG(123, engine).mixture(
                        size, [.8, .2], [1., 2.], [.3, .2], return_labels=True)
                    self.assertEqual(values.shape, (size,))
                    self.assertEqual(labels.shape, (size,))
                    self.assertTrue(np.all(np.isfinite(values) & (values > 0)))
                    self.assertTrue(np.issubdtype(labels.dtype, np.integer))
                    self.assertTrue(np.all((labels == 0) | (labels == 1)))

    def test_zero_weight_component_is_never_selected(self):
        """Компонента с нулевым весом не выбирается, а метка соответствует значению."""
        # При весах [1, 0] выбирается только компонента 0, при [0, 1] — только 1.
        # Подставляем Z=0, чтобы ожидаемые значения были известны: X=exp(mu).
        # Разные mu помогают проверить не только метки, но и выбор параметров.
        mus = np.array([0., np.log(4.)])
        for engine in ('lcg', 'xorshift'):
            for weights, expected_label in (([1., 0.], 0), ([0., 1.], 1)):
                with self.subTest(engine=engine, weights=weights):
                    g = ProteinRNG(123, engine)
                    g.normal = lambda size: np.zeros(size)
                    values, labels = g.mixture(
                        100, weights, mus, [.3, .2], return_labels=True)
                    np.testing.assert_array_equal(labels, np.full(100, expected_label))
                    np.testing.assert_allclose(values, np.full(100, np.exp(mus[expected_label])))

    def test_rejection_bounds_and_counters(self):
        """Отбор соблюдает границы и корректно учитывает принятые и отклонённые значения."""
        # Намеренно узкий интервал заставляет реально выполнить отбор.
        # У рабочей смеси выходы за 25 редки: одной такой проверки было бы мало.
        for engine in ('lcg', 'xorshift'):
            with self.subTest(engine=engine):
                g = ProteinRNG(123, engine)
                values = g.mixture(1000, [1.], [0.], [1.], lower=.5, upper=1.5)
                self.assertTrue(np.all((values >= .5) & (values <= 1.5)))
                self.assertGreater(g.last_rejected, 0)
                self.assertEqual(g.last_proposed, len(values) + g.last_rejected)
                values = g.mixture(1000, [.8, .2], [1., 2.], [.3, .2], upper=25.)
                self.assertTrue(np.all((values > 0) & (values <= 25.)))
                self.assertEqual(g.last_proposed, len(values) + g.last_rejected)

    def test_invalid_parameters(self):
        """Некорректные параметры отклоняются понятным типом исключения."""
        # Проверяем тип ошибки, а не случайное совпадение текста сообщения.
        # Веса — вероятности; mu конечны; sigma положительны; формы совпадают.
        cases = [
            ('отрицательный размер', dict(size=-1)),
            ('пустая смесь', dict(weights=[], mu=[], sigma=[])),
            ('неверная сумма весов', dict(weights=[.2, .2])),
            ('отрицательный вес', dict(weights=[-1., 2.])),
            ('NaN в весах', dict(weights=[np.nan, .2])),
            ('бесконечный вес', dict(weights=[np.inf, .2])),
            ('размер mu не совпадает', dict(mu=[1.])),
            ('размер sigma не совпадает', dict(sigma=[.3])),
            ('NaN в mu', dict(mu=[np.nan, 2.])),
            ('бесконечное mu', dict(mu=[np.inf, 2.])),
            ('нулевое sigma', dict(sigma=[0., .2])),
            ('отрицательное sigma', dict(sigma=[-.3, .2])),
            ('NaN в sigma', dict(sigma=[np.nan, .2])),
            ('бесконечное sigma', dict(sigma=[np.inf, .2])),
            ('обратный порядок границ', dict(lower=2., upper=1.)),
            ('одинаковые границы', dict(lower=1., upper=1.)),
            ('отрицательная нижняя граница', dict(lower=-1.)),
            ('NaN в границе', dict(upper=np.nan)),
        ]
        for engine in ('lcg', 'xorshift'):
            for name, override in cases:
                with self.subTest(engine=engine, case=name):
                    args = dict(size=10, weights=[.8, .2], mu=[1., 2.], sigma=[.3, .2])
                    args.update(override)
                    with self.assertRaises(ValueError):
                        ProteinRNG(123, engine).mixture(**args)

            with self.subTest(engine=engine, case='отрицательный размер normal'):
                with self.assertRaises(ValueError):
                    ProteinRNG(123, engine).normal(-1)
            for mu, sigma in ((0., 0.), (0., -.3), (np.inf, .3),
                              (0., np.inf), (np.nan, .3), (0., np.nan)):
                with self.subTest(engine=engine, lognormal_mu=mu, lognormal_sigma=sigma):
                    with self.assertRaises(ValueError):
                        ProteinRNG(123, engine).lognormal(mu, sigma, 10)

        # После приведения seed по модулю состояние тоже не должно стать нулём.
        for engine, modulus in ((ParkMiller, 2**31 - 1), (XorShift64Star, 2**64)):
            for seed in (0, modulus):
                with self.subTest(engine=engine.__name__, seed=seed):
                    with self.assertRaises(ValueError):
                        engine(seed)
        with self.assertRaises(ValueError):
            ProteinRNG(123, engine='unknown')

    def test_rejection_attempt_limit(self):
        """Невозможный отбор останавливается после заданного числа попыток."""
        # Фиксированные предложения exp(0)=1 не проходят интервал [2, 3].
        # Так проверка не зависит от вероятности редкого события.
        for engine in ('lcg', 'xorshift'):
            with self.subTest(engine=engine):
                g = ProteinRNG(123, engine)
                g.normal = lambda size: np.zeros(size)
                with self.assertRaises(RuntimeError):
                    g.mixture(5, [1.], [0.], [1.], lower=2., upper=3., max_proposals=10)
                self.assertEqual(g.last_proposed, 10)
                self.assertEqual(g.last_rejected, 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
