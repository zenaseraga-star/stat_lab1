import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
NUTRIENTS = ['proteins_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'energy-kcal_100g']
COLORS = {'proteins_100g': '#4C72B0', 'sugars_100g': '#DD8452', 'fat_100g': '#55A868'}
LABELS = {
    'proteins_100g': 'белок, г/100 г',
    'sugars_100g': 'сахара, г/100 г',
    'fat_100g': 'жиры, г/100 г',
    'carbohydrates_100g': 'углеводы, г/100 г',
    'energy-kcal_100g': 'энергия, ккал/100 г',
}
SHORT = {'proteins_100g': 'белок', 'fat_100g': 'жиры', 'carbohydrates_100g': 'углеводы', 'sugars_100g': 'сахара'}
def set_style():
    sns.set_theme(style='whitegrid', context='notebook')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 9
def clean_yogurts(raw): #функция для чистки данных
    steps = [('исходная выгрузка', len(raw))]
    df = raw.drop_duplicates('code') #убираем повторы
    steps.append(('без повторов штрихкода', len(df)))
    df = df.dropna(subset=NUTRIENTS) #берем только с заполненными кбжу
    steps.append(('заполнены белки, жиры, углеводы, сахара, энергия', len(df)))
    cat = df['categories_tags'].fillna('')
    df = df[~cat.str.contains('non-dairy|plant-based|dairy-substitutes|soy', regex=True)] #выкидываем растительные йогурты
    steps.append(('без растительных йогуртов', len(df)))
    ok = (
        df['proteins_100g'].between(0, 25)
        & df['fat_100g'].between(0, 30)
        & df['carbohydrates_100g'].between(0, 60)
        & (df['sugars_100g'] >= 0)
        & (df['sugars_100g'] <= df['carbohydrates_100g'] + 0.5)
        & (df[['proteins_100g', 'fat_100g', 'carbohydrates_100g']].sum(axis=1) <= 100) 
    ) #смотрим на здравый смысл бжу, проверяем на сумму
    df = df[ok]
    steps.append(('физически возможные значения', len(df)))
    at = atwater(df) #проверка на формуллу общей каллорийности 
    df = df[(df['energy-kcal_100g'] - at).abs() <= 0.15 * at + 5]
    steps.append(('энергия согласуется с составом (±15%)', len(df)))
    df = df.reset_index(drop=True)
    countries = df['countries_tags'].fillna('') #смотрим на ниличие разных странах
    df['russia'] = countries.str.contains('en:russia')
    df['france'] = countries.str.contains('en:france')
    df['usa'] = countries.str.contains('en:united-states')
    cat = df['categories_tags'].fillna('') #смотрим на катекории йогуртов 
    df['plain'] = cat.str.contains('en:plain-yogurts')
    df['fruit'] = cat.str.contains('en:fruit-yogurts')
    df['greek'] = cat.str.contains('en:greek-style-yogurts')
    return df, pd.DataFrame(steps, columns=['шаг', 'осталось записей']) 

def atwater(df): 
    return 4 * df['proteins_100g'] + 9 * df['fat_100g'] + 4 * df['carbohydrates_100g'] #формула общей каллорийности из энергетической ценности каждого нутриента
def describe(x): #расчитаем статистические параметпы датасета со случайными величинами
    x = pd.Series(x).dropna()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    return {
        'n': len(x),
        'E': x.mean(),
        'D': x.var(),
        'sigma': x.std(),
        'CV, %': x.std() / x.mean() * 100,
        'min': x.min(),
        'Q1': q1,
        'медиана': x.median(),
        'Q3': q3,
        'max': x.max(),
        'IQR': q3 - q1,
        'мода': x.mode().iloc[0],
        'асимметрия': stats.skew(x),
        'эксцесс': stats.kurtosis(x),
    }
def describe_table(columns):
    return pd.DataFrame({name: describe(x) for name, x in columns.items()}).round(3)
def atwater_check(raw):
    d = raw.dropna(subset=NUTRIENTS)
    d = d[d['proteins_100g'].between(0, 25) & d['fat_100g'].between(0, 30) & d['carbohydrates_100g'].between(0, 60)]
    at = atwater(d) #рассчитаем теоретическую каллорийность
    diff = d['energy-kcal_100g'] - at #разница между теоретической и указанной каллорийностью
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    axes[0].scatter(at, d['energy-kcal_100g'], s=2, alpha=0.2, color='#8172B2') #строим график, где можно проследить разницу
    axes[0].plot([0, 400], [0, 400], 'k--', lw=1, label='y = 4Б + 9Ж + 4У') #диагональ
    axes[0].set_xlim(0, 400)
    axes[0].set_ylim(0, 400)
    axes[0].set_title('Энергия на этикетке и формула Этуотера')
    axes[0].set_xlabel('4Б + 9Ж + 4У, ккал/100 г')
    axes[0].set_ylabel(LABELS['energy-kcal_100g'])
    axes[0].legend()
    axes[1].hist(diff.clip(-30, 30), bins=np.arange(-30, 31, 1), color='#8172B2', alpha=0.7) #гистограмма, разницу обрезаем до 30, чтобы избежать влияния выбросов
    axes[1].set_title('Разница: этикетка - формула')
    axes[1].set_xlabel('ккал/100 г (обрезано до ±30)')
    axes[1].set_ylabel('количество')
    plt.tight_layout()
    plt.show()
    return {
        'записей': len(d),
        'медиана |разницы|, ккал': diff.abs().median(),
        'доля |разницы| <= 5 ккал': (diff.abs() <= 5).mean(),
        'доля |разницы| <= 10 ккал': (diff.abs() <= 10).mean(),
    }
def randomness_check(df, col, features, seed=0):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    X = df[features]
    y = df[col]
    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)#создаем лес
    pred = cross_val_predict(model, X, y, cv=5)#создание фолдов и возврат данных с тестов модели(предсказания)
    res = y - pred
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()#считае коэффициент детерминации
    #далее поиск двойников, чтобы убедиться в правдивости полученных результатов
    Z = StandardScaler().fit_transform(X) 
    indices = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[1]
    indices = indices.astype(int)
    twin = np.abs(y.values - y.values[indices[:, 1]])#разница белка по модулю
    rnd = np.abs(y.values - np.random.default_rng(seed).permutation(y.values)) #перемешиваем пары и находи разницу

    c = COLORS[col]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].scatter(y, pred, s=3, alpha=0.2, color=c)#график разницы предсказанных и реальных
    lim = [y.min(), y.max()]
    axes[0].plot(lim, lim, 'k--', lw=1, label='если бы было y = f(x)')
    axes[0].set_title(f'Случайный лес, 5-fold CV, R$^2$ = {r2:.2f}')
    axes[0].set_xlabel('реальное значение')
    axes[0].set_ylabel('предсказание по: ' + ', '.join(SHORT[f] for f in features))
    axes[0].legend()
    axes[1].hist(res, bins=50, density=True, color=c, alpha=0.6)
    axes[1].set_title(f'Остатки модели, sigma = {res.std():.3f}')
    axes[1].set_xlabel('реальное - предсказанное')
    axes[1].set_ylabel('плотность')
    hi = np.quantile(rnd, 0.99)
    axes[2].hist(twin, bins=np.linspace(0, hi, 30), density=True, alpha=0.6, color=c, label='ближайший "двойник"')
    axes[2].hist(rnd, bins=np.linspace(0, hi, 30), density=True, alpha=0.4, color='grey', label='случайная пара')
    axes[2].set_title('|разница| у самых похожих йогуртов')
    axes[2].set_xlabel('модуль разницы значений')
    axes[2].set_ylabel('плотность')
    axes[2].legend()
    plt.tight_layout()
    plt.show()
    return {
        'R2 (CV)': r2,
        'sigma остатков': res.std(),
        'sigma величины': y.std(),
        'медиана |разн.| у двойников': np.median(twin),
        'медиана |разн.| у случайных пар': np.median(rnd),
        'доля двойников с другим значением': (twin > 0).mean(),
    }
def twins_exact(df, col, features): #ищет близнецов
    g = df.groupby(features)[col]
    stats_g = g.agg(['count', 'min', 'max', 'std'])
    stats_g = stats_g[stats_g['count'] >= 5]
    stats_g['размах'] = stats_g['max'] - stats_g['min']
    return stats_g.sort_values('count', ascending=False)

def order_check(x, col, window=500):
    from statsmodels.sandbox.stats.runs import runstest_1samp

    z, p = runstest_1samp(x.values, cutoff='median')
    ac = pd.Series(x.values).autocorr(1)
    shuffled = pd.Series(np.random.default_rng(1).permutation(x.values))#перемешаем весь датасет, чтобы точно избежать скрытого порядка
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(x.values, lw=0.2, color='grey', alpha=0.4)
    ax.plot(pd.Series(x.values).rolling(window).mean().values, color=COLORS[col], lw=2, label=f'скользящее среднее по {window}, как в файле')
    ax.plot(shuffled.rolling(window).mean().values, color='k', lw=1.5, ls='--', label='то же после перемешивания')
    ax.set_title('Значения по порядку записей в файле')
    ax.set_xlabel('номер записи')
    ax.set_ylabel(LABELS[col])
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    return {'критерий серий, z': z, 'критерий серий, p-value': p, 'автокорреляция lag 1': ac,
            'автокорреляция после перемешивания': shuffled.autocorr(1)} #z показывает сколько раз происходит чередование значения относительно медианного

def eda_plots(x, col, bins=60):
    c = COLORS[col]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    sns.histplot(x, bins=bins, kde=True, stat='density', color=c, ax=axes[0])#строим распределение
    axes[0].axvline(x.mean(), color='k', ls='--', lw=1, label=f'E = {x.mean():.2f}')
    axes[0].axvline(x.median(), color='r', ls=':', lw=1.5, label=f'медиана = {x.median():.2f}')
    axes[0].set_title('Гистограмма и KDE')
    axes[0].set_xlabel(LABELS[col])
    axes[0].set_ylabel('плотность')
    axes[0].legend()
    sns.boxplot(x=x, color=c, ax=axes[1], flierprops={'markersize': 2})#строим бокс по процентилям(где по белку лежит основная масса йогуртов)
    axes[1].set_title('Boxplot')
    axes[1].set_xlabel(LABELS[col])
    stats.probplot(x, dist='norm', plot=axes[2])#сравниваем с нормальным распределением по процентилям
    axes[2].get_lines()[0].set(color=c, markersize=2)
    axes[2].get_lines()[1].set(color='k')
    axes[2].set_title('Q-Q plot относительно нормального')
    axes[2].set_xlabel('теоретические квантили N(0, 1)')
    axes[2].set_ylabel('выборочные квантили')
    plt.tight_layout()
    plt.show()


def outliers_iqr(x):
    q1, q3 = x.quantile(0.25), x.quantile(0.75) #считаем границы по процентилям
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return x[(x < low) | (x > high)], low, high

def fit_mixture(x, k=2, log=False, seed=0):
    from sklearn.mixture import GaussianMixture

    v = np.log(x.values) if log else x.values
    g = GaussianMixture(n_components=k, random_state=seed, n_init=5).fit(v.reshape(-1, 1))#ищем холмики(2)
    order = np.argsort(g.means_.ravel())
    w = g.weights_[order]
    mu = g.means_.ravel()[order]
    sd = np.sqrt(g.covariances_.ravel())[order]
    ll = g.score(v.reshape(-1, 1)) * len(v) #расчет, насколько правильно мы описали данные
    if log:
        ll = ll - np.log(x.values).sum()
    name = f'смесь {k} логнормальных' if log else f'смесь {k} нормальных'
    return {'name': name, 'k': k, 'log': log, 'w': w, 'mu': mu, 'sd': sd, 'logL': ll, 'model': g}


def fit_table(x, mixture=None):#смотрим по тому, что мы получили в прошлый раз, какое распределение подходит лучше всего
    rows = {}
    for name, dist in [('нормальное', stats.norm), ('логнормальное', stats.lognorm), ('гамма', stats.gamma)]:
        if name == 'нормальное':
            p = dist.fit(x)
        else:
            p = dist.fit(x, floc=0)
        ll = dist.logpdf(x, *p).sum()
        rows[name] = {'число параметров': 2, 'logL': ll, 'AIC': 4 - 2 * ll, 'BIC': 2 * np.log(len(x)) - 2 * ll}
    if mixture is not None:
        k = 3 * mixture['k'] - 1
        rows[mixture['name']] = {'число параметров': k, 'logL': mixture['logL'], 'AIC': 2 * k - 2 * mixture['logL'],
                                 'BIC': k * np.log(len(x)) - 2 * mixture['logL']}
    return pd.DataFrame(rows).T

def mixture_table(m): #сравним парметры двух полученных компонент
    rows = []
    for i, (w, mu, sd) in enumerate(zip(m['w'], m['mu'], m['sd'])):
        row = {'компонента': i + 1, 'вес w': w, 'mu': mu, 'sigma': sd}
        row['медиана компоненты, г/100 г'] = np.exp(mu) if m['log'] else mu
        rows.append(row)
    return pd.DataFrame(rows).round(3)

def mixture_pdf(grid, m): #возращает сумму колоколов и какждый по отдельности
    total = np.zeros_like(grid, dtype=float)
    parts = []
    for w, mu, sd in zip(m['w'], m['mu'], m['sd']):
        if m['log']:
            p = w * stats.lognorm.pdf(grid, sd, scale=np.exp(mu))
        else:
            p = w * stats.norm.pdf(grid, mu, sd)
        parts.append(p)
        total += p
    return total, parts

def mixture_cdf(grid, m): #рассчитываем вероятности для функции распределения
    total = np.zeros_like(grid, dtype=float)
    for w, mu, sd in zip(m['w'], m['mu'], m['sd']):
        if m['log']:
            total += w * stats.lognorm.cdf(grid, sd, scale=np.exp(mu))
        else:
            total += w * stats.norm.cdf(grid, mu, sd)
    return total

def plot_fits(x, col, m, xmax=None):
    c = COLORS[col]
    xmax = xmax or x.quantile(0.999)
    grid = np.linspace(max(x.min(), 1e-3), xmax, 500)
    pn = stats.norm.fit(x)
    pl = stats.lognorm.fit(x, floc=0)
    total, parts = mixture_pdf(grid, m)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.3))
    axes[0].hist(x[x <= xmax], bins=80, density=True, color=c, alpha=0.4, label='данные')#строим график плотностей для различных распределений
    axes[0].plot(grid, stats.norm.pdf(grid, *pn), 'k--', lw=1, label='нормальное')
    axes[0].plot(grid, stats.lognorm.pdf(grid, *pl), color='grey', lw=1, label='логнормальное')
    axes[0].plot(grid, total, 'r-', lw=2, label=m['name'])
    for i, p in enumerate(parts):
        axes[0].plot(grid, p, 'r:', lw=1.2, label='компоненты смеси' if i == 0 else None)
    axes[0].set_title('Подбор распределения')
    axes[0].set_xlabel(LABELS[col])
    axes[0].set_ylabel('плотность')
    axes[0].legend()
    xs = np.sort(x.values)
    axes[1].step(xs, np.arange(1, len(xs) + 1) / len(xs), where='post', color=c, label='эмпирическая F(x)')#строим сами функции распределения 
    axes[1].plot(grid, stats.norm.cdf(grid, *pn), 'k--', lw=1, label='нормальное')
    axes[1].plot(grid, stats.lognorm.cdf(grid, *pl), color='grey', lw=1, label='логнормальное')
    axes[1].plot(grid, mixture_cdf(grid, m), 'r-', lw=1.5, label=m['name'])
    axes[1].set_xlim(grid[0], xmax)
    axes[1].set_title('Функции распределения')
    axes[1].set_xlabel(LABELS[col])
    axes[1].set_ylabel('F(x)')
    axes[1].legend()
    plt.tight_layout()
    plt.show()