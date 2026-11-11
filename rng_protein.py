import numpy as np
import operator

def protein_inputs(yog, mix_p):
    protein = yog["proteins_100g"].to_numpy(dtype=float)
    weights = mix_p["w"]
    mus = mix_p["mu"]
    sigmas = mix_p["sd"]
    return protein, weights, mus, sigmas

class ParkMiller:
    period = 2**31 - 2

    def __init__(self, seed=20261109):
        self.state = operator.index(seed) % (2**31 - 1)
        if self.state == 0:
            raise ValueError("seed должен быть ненулевым по модулю 2**31 - 1")

    def uniform(self, size):
        out = np.empty(operator.index(size))

        for i in range(len(out)):
            self.state = (16807 * self.state) % (2**31 - 1)
            out[i] = self.state / (2**31 - 1)

        return out


class XorShift64Star:
    period = 2**64 - 1
    mask = 2**64 - 1

    def __init__(self, seed=20261109):
        self.state = operator.index(seed) & self.mask
        if self.state == 0:
            raise ValueError("seed должен быть ненулевым по модулю 2**64")

    def uniform(self, size):
        out = np.empty(operator.index(size))

        for i in range(len(out)):
            # меняем состояние сдвигами и оставляем 64 бита
            x = self.state
            x ^= x >> 12
            x ^= (x << 25) & self.mask
            x ^= x >> 27
            self.state = x

            word = (x * 2685821657736338717) & self.mask

            # старшие 52 бита переводим в интервал строго между 0 и 1
            out[i] = ((word >> 12) + 0.5) / 2**52

        return out


class ProteinRNG:
    def __init__(self, seed=20261109, engine="xorshift"):
        if engine not in ("xorshift", "lcg"):
            raise ValueError("неизвестный базовый датчик")

        self.base = (
            XorShift64Star if engine == "xorshift" else ParkMiller
        )(seed)

        self._spare = None
        self.last_proposed = 0
        self.last_rejected = 0

    def uniform(self, size):
        return self.base.uniform(size)

    def normal(self, size):
        size = operator.index(size)
        if size < 0:
            raise ValueError("размер выборки должен быть неотрицательным")

        out = np.empty(size)
        start = 0

        # сначала используем число, оставшееся с прошлого вызова
        if size and self._spare is not None:
            out[0] = self._spare
            self._spare = None
            start = 1

        count = size - start

        if count:
            # для каждой нормальной пары нужны два равномерных числа
            pair_count = (count + 1) // 2
            u = self.uniform(2 * pair_count).reshape(-1, 2)

            radius = np.sqrt(-2 * np.log(u[:, 0]))
            angle = 2 * np.pi * u[:, 1]

            z = np.column_stack((
                radius * np.cos(angle),
                radius * np.sin(angle),
            )).ravel()

            out[start:] = z[:count]

            # лишнее число сохраняем до следующего вызова
            if count % 2:
                self._spare = z[-1]

        return out

    def lognormal(self, mu, sigma, size):
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("mu должен быть конечным, sigma положительным")

        return np.exp(mu + sigma * self.normal(size))

    def mixture(
        self, size, weights, mu, sigma,
        lower=0., upper=np.inf,
        return_labels=False, max_proposals=None,
    ):
        size = operator.index(size)
        w = np.asarray(weights, dtype=float)
        m = np.asarray(mu, dtype=float)
        s = np.asarray(sigma, dtype=float)

        if (
            size < 0
            or w.ndim != 1
            or not len(w)
            or m.shape != w.shape
            or s.shape != w.shape
            or not np.all(np.isfinite([w, m, s]))
            or np.any(w < 0)
            or not np.isclose(w.sum(), 1.)
            or np.any(s <= 0)
        ):
            raise ValueError("некорректные параметры смеси")

        if not (0 <= lower < upper):
            raise ValueError("границы должны удовлетворять 0 <= lower < upper")

        if max_proposals is None:
            limit = max(10000, 1000 * size)
        else:
            limit = operator.index(max_proposals)

        chunks = []
        labels = []
        remaining = size
        self.last_proposed = 0
        self.last_rejected = 0

        # накопленные веса задают границы выбора компоненты
        cdf = np.cumsum(w / w.sum())
        cdf[-1] = 1.

        while remaining:
            if self.last_proposed + remaining > limit:
                raise RuntimeError("слишком много отклонений, проверь границы")

            # выбираем компоненты по исходным весам
            component = np.searchsorted(
                cdf, self.uniform(remaining), side="right"
            )

            values = np.exp(
                m[component] + s[component] * self.normal(remaining)
            )

            keep = (values >= lower) & (values <= upper)

            self.last_proposed += remaining
            self.last_rejected += int((~keep).sum())

            chunks.append(values[keep])
            labels.append(component[keep])
            remaining -= int(keep.sum())

        values = np.concatenate(chunks) if chunks else np.empty(0)
        component = (
            np.concatenate(labels) if labels else np.empty(0, dtype=int)
        )

        if return_labels:
            return values, component

        return values