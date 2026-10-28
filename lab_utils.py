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