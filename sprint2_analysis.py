from rng_protein import ProteinRNG, ParkMiller, XorShift64Star
import time
import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261109


def protein_summary(protein):
    return {
        "Всего значений": len(protein),
        "Положительных значений": int((protein > 0).sum()),
        "Нулей": int((protein == 0).sum()),
    }
def compare(a, b, bins):
    ca, cb = np.histogram(a, bins)[0], np.histogram(b, bins)[0] #смотрим плотности двух выборок
    use = (ca + cb) > 0 #отсеиваем 0
    table = np.array([ca[use], cb[use]])
    chi, p, df, expected = stats.chi2_contingency(table, correction=False) #хи квадрат, как коэффициет совпадения выборок
    ks = stats.ks_2samp(a, b, method='asymp') #смещение распределение 
    return {'KS D': ks.statistic, 'KS p': ks.pvalue, 'chi2': chi,
            'chi2 df': df, 'chi2 p': p, 'min expected': expected.min()}

def uniform_tests(u): #тест на случайность сгенерированных значений
    n = len(u)
    chi = stats.chisquare(np.histogram(u, np.linspace(0, 1, 21))[0]) #смостим, насколько равномерно генерируются значения
    lag = np.corrcoef(u[:-1], u[1:])[0, 1] #проверка на связь подряд сегрерированных значений
    bits = u > .5
    n1, n0 = int(bits.sum()), int((~bits).sum())
    runs = 1 + np.count_nonzero(bits[1:] != bits[:-1]) #проверка сериями
    mean = 1 + 2*n1*n0/n
    var = 2*n1*n0*(2*n1*n0-n)/(n*n*(n-1))
    runs_p = 2*stats.norm.sf(abs((runs - mean)/np.sqrt(var)))
    pairs = u[:n//2*2].reshape(-1, 2) # проверка парами
    counts = np.histogram2d(*pairs.T, bins=[np.linspace(0, 1, 11)]*2)[0]
    pair = stats.chisquare(counts.ravel())
    return {'chi2 U p': chi.pvalue, 'rho lag 1': lag,
            'lag p (approx)': 2*stats.norm.sf(abs(lag)*np.sqrt(n)),
            'серии p': runs_p, 'пары p': pair.pvalue}

def versions(real, w, mu, sd): #сравнение версий генератора по функциям выше
    rows, samples = [], {}
    bins = np.r_[-np.inf, np.quantile(real, np.arange(.1, 1., .1)), np.inf] #задаем бины для всех выборок, по реальным квантелям
    for name, engine, upper in [('v1: Park-Miller + BM', 'lcg', np.inf),
                                 ('v2: xorshift64* + BM', 'xorshift', np.inf),
                                 ('v3: v2 + отбор', 'xorshift', 25.)]:
        g = ProteinRNG(SEED, engine)
        start = time.perf_counter()
        y = g.mixture(len(real), w, mu, sd, upper=upper)
        elapsed = time.perf_counter() - start
        samples[name] = y
        base = (ParkMiller if engine == 'lcg' else XorShift64Star)(SEED)
        rows.append({'версия': name, 'период': str(base.period),
                     'время, с': elapsed, 'доля > 25': (y > 25).mean(),
                     'отброшено': g.last_rejected,
                     **uniform_tests(base.uniform(100000)), **compare(real, y, bins)})
    return pd.DataFrame(rows).set_index('версия'), samples
