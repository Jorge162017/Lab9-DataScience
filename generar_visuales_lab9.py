"""
Ejecutar: python3 generar_visuales_lab9.py
"""
from pathlib import Path
import json
import warnings
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'Lab1-DataScience'
OUT = ROOT / 'recursos_lab9'
OUT.mkdir(exist_ok=True)
BLUE, GREEN, GRAY, ORANGE = ('#17324D', '#246B52', '#475569', '#C45A0A')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold', 'text.color': BLUE, 'axes.labelcolor': BLUE, 'figure.facecolor': 'white'})

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=240, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def axis(ax):
    ax.grid(axis='y', alpha=0.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:,.0f} mil'))
    ax.set_ylabel('Ingresos de visitantes')
raw = pd.read_excel(next(SRC.glob('*.xlsx')), sheet_name='Datos')
df = raw.copy()
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype('string').str.strip().str.replace('\\s+', ' ', regex=True)
df['Región dos'] = df['Región dos'].replace('0', 'Sin clasificar')
for col in ['Año', 'Mes cod', 'Viajero']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.drop_duplicates()
df['Fecha'] = pd.to_datetime(dict(year=df['Año'], month=df['Mes cod'], day=1), errors='coerce')
df = df.loc[df.Fecha.notna() & df.Viajero.notna() & df.Viajero.ge(0)]
v = df.loc[df['Tipo de Viajero'].isin(['Turista', 'Excursionista'])]
idx = pd.date_range(v.Fecha.min(), v.Fecha.max(), freq='MS')

def series(data):
    return data.groupby('Fecha').Viajero.sum().reindex(idx, fill_value=0).astype(float)
s = series(v)
ss = {'Total visitantes': s}
for via in ['Aérea', 'Terrestre', 'Marítima']:
    ss[f'Vía — {via}'] = series(v.loc[v.Vía.eq(via)])
for region in v.groupby('Región dos').Viajero.sum().nlargest(3).index:
    ss[f'Región — {region}'] = series(v.loc[v['Región dos'].eq(region)])
years = s.groupby(s.index.year).sum()
shares = v.groupby('Vía').Viajero.sum() / v.Viajero.sum() * 100
kpis = {'periodo': 'enero de 2009 a junio de 2026', 'meses': len(s), 'visitantes_2019': float(years[2019]), 'visitantes_2020': float(years[2020]), 'caida_2020_vs_2019_pct': float((years[2020] / years[2019] - 1) * 100), 'visitantes_2025': float(years[2025]), 'cambio_2025_vs_2019_pct': float((years[2025] / years[2019] - 1) * 100), 'participacion_terrestre_periodo_pct': float(shares['Terrestre']), 'definicion': 'Ingresos de turistas + excursionistas; no personas únicas. 2026 parcial.'}
cards = [('2025', f'{years[2025] / 1000000.0:.2f} millones', 'ingresos de visitantes'), ('2020 frente a 2019', f"{kpis['caida_2020_vs_2019_pct']:.1f} %", 'caída durante la pandemia'), ('2025 frente a 2019', f"{kpis['cambio_2025_vs_2019_pct']:+.1f} %", 'variación frente al nivel prepandemia'), ('Enero 2009–junio 2026', f"{shares['Terrestre']:.1f} %", 'de los ingresos por vía terrestre')]
fig, axs = plt.subplots(1, 4, figsize=(16, 2.5))
for ax, (title, value, label) in zip(axs, cards):
    ax.axis('off')
    ax.text(0.02, 0.86, title, fontsize=11)
    ax.text(0.02, 0.46, value, fontsize=25, weight='bold', color=ORANGE if title.startswith('2020') else BLUE)
    ax.text(0.02, 0.17, label, fontsize=10, wrap=True)
save(fig, '01_indicadores')
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(s, color=BLUE, lw=1.5)
ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'), color=ORANGE, alpha=0.12)
ax.annotate('2020: choque pandémico', xy=(pd.Timestamp('2020-07-01'), s.loc['2020-07-01']), xytext=(pd.Timestamp('2014-01-01'), s.max() * 0.8), arrowprops={'arrowstyle': '->', 'color': ORANGE}, color=ORANGE)
ax.axvline(pd.Timestamp('2023-01-01'), color=GRAY, ls=':', alpha=0.7)
ax.text(pd.Timestamp('2023-02-01'), s.max() * 0.55, '2023: cambio\nmetodológico', fontsize=9)
ax.set_title('Las llegadas a Guatemala: caída y recuperación')
axis(ax)
ax.text(0, -0.21, 'Turistas + excursionistas · 2026 incluye solo enero–junio.', transform=ax.transAxes, fontsize=9)
save(fig, '02_historia')
rows = []
decs = {}
for name, x in ss.items():
    dec = seasonal_decompose(x, model='additive', period=12, extrapolate_trend='freq')
    decs[name] = dec
    pre = x.loc[:'2019-12-01']
    rows.append({'Serie': name, 'Fuerza estacional': max(0, 1 - np.var(dec.resid) / np.var(dec.resid + dec.seasonal)), 'Tendencia absoluta pre2020': np.polyfit(np.arange(len(pre)), pre, 1)[0], 'Volatilidad relativa pre2020': pre.pct_change().replace([np.inf, -np.inf], np.nan).dropna().std(), 'Cambio 2020 vs 2019 pct': (x.loc['2020'].sum() / x.loc['2019'].sum() - 1) * 100 if x.loc['2019'].sum() > 0 else np.nan})
comp = pd.DataFrame(rows).set_index('Serie')
pat = decs['Total visitantes'].seasonal.groupby(s.index.month).mean()
months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.bar(months, pat, color=[GREEN if x >= 0 else GRAY for x in pat])
ax.axhline(0, color=BLUE, lw=0.7)
ax.set_title('¿Qué meses suelen superar el nivel habitual?')
ax.set_ylabel('Desviación estacional (ingresos)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:,.0f} mil'))
ax.text(0, -0.23, 'Componente estacional aditivo, período 12 · No representa el volumen observado ni prueba causas.', transform=ax.transAxes, fontsize=9)
save(fig, '03_estacionalidad')
fig, ax = plt.subplots(figsize=(12, 4.1))
ax.axis('off')
cell = [[name.replace('Vía — ', '').replace('Región — ', ''), f'{r.iloc[0]:.3f}', f'{r.iloc[1]:,.1f}', f'{r.iloc[2]:.3f}', f'{r.iloc[3]:+.1f} %' if np.isfinite(r.iloc[3]) else 'No comparable'] for name, r in comp.iterrows()]
t = ax.table(cellText=cell, colLabels=['Serie', 'Fuerza\nestacional', 'Pendiente pre2020\n(ingresos/mes)', 'Volatilidad\nrelativa', '2020 vs\n2019'], loc='center', cellLoc='center', colWidths=[0.27, 0.15, 0.23, 0.17, 0.18])
t.auto_set_font_size(False)
t.set_fontsize(10)
t.scale(1, 2)
for (r, c), cl in t.get_celld().items():
    cl.set_edgecolor('#E2E8F0')
    if r == 0:
        cl.set_facecolor(BLUE)
        cl.set_text_props(color='white', weight='bold')
ax.set_title('Vías y mercados: patrones distintos', pad=18)
ax.text(0, -0.09, 'Volatilidad: desviación de cambios porcentuales antes de 2020. Marítima: cobertura limitada; interpretar con cautela.', transform=ax.transAxes, fontsize=9)
save(fig, '04_comparacion_series')
nb = json.loads((SRC / 'Lab1_AnalisisSeriesTiempo.ipynb').read_text())
metricas = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'filas_metricas = []' in ''.join(cell['source']):
        for output in cell.get('outputs', []):
            h = ''.join(output.get('data', {}).get('text/html', []))
            soup = BeautifulSoup(h, 'html.parser')
            table = soup.find('table')
            if table is None:
                continue
            headers = [th.get_text(strip=True) for th in table.find('thead').find_all('tr')[0].find_all('th')][1:]
            if 'Serie' not in headers:
                continue
            data = [[td.get_text(strip=True) for td in tr.find_all('td')] for tr in table.find('tbody').find_all('tr')]
            metricas = pd.DataFrame(data, columns=headers)
if metricas is None:
    raise RuntimeError('No se encontró la tabla de métricas guardada en el notebook.')
for col in ['MAE', 'RMSE']:
    metricas[col] = metricas[col].str.replace(',', '', regex=False).astype(float)
total = metricas.loc[metricas.Serie.eq('Total visitantes')].sort_values('RMSE')
fig, ax = plt.subplots(figsize=(10, 3.8))
bars = ax.barh(total.Modelo, total.RMSE, color=[GREEN] + [GRAY] * (len(total) - 1))
ax.invert_yaxis()
ax.set_xlim(0, total.RMSE.max() * 1.4)
for bar, (_, r) in zip(bars, total.iterrows()):
    ax.text(bar.get_width() + total.RMSE.max() * 0.02, bar.get_y() + bar.get_height() / 2, f'RMSE {r.RMSE:,.0f}\nMAE {r.MAE:,.0f}', va='center', fontsize=9)
ax.set_title('Para el total, un modelo sencillo tuvo menor error')
ax.set_xlabel('RMSE en ingresos de visitantes · menor es mejor')
ax.text(0, -0.25, 'Prueba: abril 2021–junio 2026 · Métricas guardadas en el Lab 1, redondeadas a unidades.', transform=ax.transAxes, fontsize=9)
save(fig, '05_modelos_total')
cut = int(np.floor(0.7 * len(s)))
train, test = (s.iloc[:cut], s.iloc[cut:])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fit = SimpleExpSmoothing(train, initialization_method='estimated').fit(optimized=True)
pred = np.maximum(np.asarray(fit.forecast(len(test))), 0)
mae = float(np.mean(abs(test.values - pred)))
rmse = float(np.sqrt(np.mean((test.values - pred) ** 2)))
ref = total.loc[total.Modelo.eq('Suavizamiento exponencial')].iloc[0]
assert abs(mae - ref.MAE) < 1 and abs(rmse - ref.RMSE) < 1, 'El ajuste no reproduce el Lab 1'
alpha = float(fit.params['smoothing_level'])
res = np.asarray(fit.resid)[1:]
sigma2 = float(np.sum(res ** 2) / (len(res) - 2))
h = np.arange(1, len(test) + 1)
se = np.sqrt(sigma2 * (1 + (h - 1) * alpha ** 2))
z = 1.959963984540054
lower = np.maximum(pred - z * se, 0)
upper = np.maximum(pred + z * se, 0)
forecast = pd.DataFrame({'Real': test, 'Estimacion': pred, 'Limite_inferior_95': lower, 'Limite_superior_95': upper})
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(train.iloc[-24:], color=GRAY, lw=1, label='Entrenamiento (últimos 24 meses)')
ax.plot(test, color=BLUE, lw=1.6, label='Observado en prueba')
ax.fill_between(test.index, lower, upper, color=ORANGE, alpha=0.17, label='Intervalo de predicción 95 % (condicional)')
ax.plot(test.index, pred, color=ORANGE, lw=2, ls='--', label='Estimación: suavizamiento exponencial')
ax.axvline(test.index[0], color=GRAY, ls=':', lw=1)
ax.set_title('Estimar no es adivinar: la recuperación desafió al modelo')
axis(ax)
ax.legend(fontsize=8, loc='upper left')
ax.text(0, -0.24, 'Modelo entrenado hasta marzo 2021 · Evaluación retrospectiva, no proyección posterior a junio 2026.\nBanda basada en errores de entrenamiento; no garantiza cubrir rupturas como la recuperación pandémica.', transform=ax.transAxes, fontsize=9)
save(fig, '06_pronostico_intervalo95')
assert len(s) == 210 and len(train) == 147 and (len(test) == 63)
assert np.isfinite(forecast.values).all() and (lower <= pred).all() and (pred <= upper).all()
