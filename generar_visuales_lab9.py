"""Genera recursos del Lab 9 desde la raíz, sin modificar el Lab 1.
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
BLUE, GREEN, GRAY, ORANGE = '#17324D', '#246B52', '#475569', '#C45A0A'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.titleweight':'bold', 'text.color':BLUE,
                     'axes.labelcolor':BLUE, 'figure.facecolor':'white'})

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=240, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def axis(ax):
    ax.grid(axis='y', alpha=.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: f'{x/1000:,.0f} mil'))
    ax.set_ylabel('Ingresos de visitantes')

raw=pd.read_excel(next(SRC.glob('*.xlsx')),sheet_name='Datos')
df=raw.copy();df.columns=df.columns.str.strip()
for col in df.select_dtypes(include=['object','string']).columns:
    df[col]=df[col].astype('string').str.strip().str.replace(r'\s+',' ',regex=True)
df['Región dos']=df['Región dos'].replace('0','Sin clasificar')
for col in ['Año','Mes cod','Viajero']:df[col]=pd.to_numeric(df[col],errors='coerce')
df=df.drop_duplicates()
df['Fecha']=pd.to_datetime(dict(year=df['Año'],month=df['Mes cod'],day=1),errors='coerce')
df=df.loc[df.Fecha.notna() & df.Viajero.notna() & df.Viajero.ge(0)]
v=df.loc[df['Tipo de Viajero'].isin(['Turista','Excursionista'])]
idx=pd.date_range(v.Fecha.min(),v.Fecha.max(),freq='MS')
def series(data):return data.groupby('Fecha').Viajero.sum().reindex(idx,fill_value=0).astype(float)
s=series(v)
ss={'Total visitantes':s}
for via in ['Aérea','Terrestre','Marítima']:ss[f'Vía — {via}']=series(v.loc[v.Vía.eq(via)])
for region in v.groupby('Región dos').Viajero.sum().nlargest(3).index:
    ss[f'Región — {region}']=series(v.loc[v['Región dos'].eq(region)])
series_df=pd.DataFrame(ss);series_df.to_csv(OUT/'series_mensuales.csv',index_label='Fecha')
years=s.groupby(s.index.year).sum()
shares=v.groupby('Vía').Viajero.sum()/v.Viajero.sum()*100
kpis={
 'periodo':'enero de 2009 a junio de 2026', 'meses':len(s),
 'visitantes_2019':float(years[2019]),'visitantes_2020':float(years[2020]),
 'caida_2020_vs_2019_pct':float((years[2020]/years[2019]-1)*100),
 'visitantes_2025':float(years[2025]),
 'cambio_2025_vs_2019_pct':float((years[2025]/years[2019]-1)*100),
 'participacion_terrestre_periodo_pct':float(shares['Terrestre']),
 'definicion':'Ingresos de turistas + excursionistas; no personas únicas. 2026 parcial.'}
(OUT/'indicadores.json').write_text(json.dumps(kpis,ensure_ascii=False,indent=2))
cards=[('2025',f'{years[2025]/1e6:.2f} millones','ingresos de visitantes'),
       ('2020 frente a 2019',f'{kpis["caida_2020_vs_2019_pct"]:.1f} %','caída durante la pandemia'),
       ('2025 frente a 2019',f'{kpis["cambio_2025_vs_2019_pct"]:+.1f} %','variación frente al nivel prepandemia'),
       ('Enero 2009–junio 2026',f'{shares["Terrestre"]:.1f} %','de los ingresos por vía terrestre')]
fig,axs=plt.subplots(1,4,figsize=(16,2.5))
for ax,(title,value,note) in zip(axs,cards):
 ax.axis('off');ax.text(.02,.86,title,fontsize=11);ax.text(.02,.46,value,fontsize=25,weight='bold',color=ORANGE if title.startswith('2020') else BLUE);ax.text(.02,.17,note,fontsize=10,wrap=True)
save(fig,'01_indicadores')
fig,ax=plt.subplots(figsize=(12,4));ax.plot(s,color=BLUE,lw=1.5)
ax.axvspan(pd.Timestamp('2020-03-01'),pd.Timestamp('2020-12-31'),color=ORANGE,alpha=.12)
ax.annotate('2020: choque pandémico',xy=(pd.Timestamp('2020-07-01'),s.loc['2020-07-01']),xytext=(pd.Timestamp('2014-01-01'),s.max()*.8),arrowprops={'arrowstyle':'->','color':ORANGE},color=ORANGE)
ax.axvline(pd.Timestamp('2023-01-01'),color=GRAY,ls=':',alpha=.7)
ax.text(pd.Timestamp('2023-02-01'),s.max()*.55,'2023: cambio\nmetodológico',fontsize=9)
ax.set_title('Las llegadas a Guatemala: caída y recuperación');axis(ax)
ax.text(0,-.21,'Turistas + excursionistas · 2026 incluye solo enero–junio.',transform=ax.transAxes,fontsize=9)
save(fig,'02_historia')
rows=[];decs={}
for name,x in ss.items():
 dec=seasonal_decompose(x,model='additive',period=12,extrapolate_trend='freq');decs[name]=dec
 pre=x.loc[:'2019-12-01']
 rows.append({'Serie':name,'Fuerza estacional':max(0,1-np.var(dec.resid)/np.var(dec.resid+dec.seasonal)),
 'Tendencia absoluta pre2020':np.polyfit(np.arange(len(pre)),pre,1)[0],
 'Volatilidad relativa pre2020':pre.pct_change().replace([np.inf,-np.inf],np.nan).dropna().std(),
 'Cambio 2020 vs 2019 pct':((x.loc['2020'].sum()/x.loc['2019'].sum()-1)*100 if x.loc['2019'].sum()>0 else np.nan)})
comp=pd.DataFrame(rows).set_index('Serie');comp.to_csv(OUT/'comparacion_series.csv')
pat=decs['Total visitantes'].seasonal.groupby(s.index.month).mean()
pat.to_csv(OUT/'perfil_estacional.csv',index_label='Mes',header=['Desviacion_visitantes'])
months=['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
fig,ax=plt.subplots(figsize=(10,3.8));ax.bar(months,pat,color=[GREEN if x>=0 else GRAY for x in pat]);ax.axhline(0,color=BLUE,lw=.7)
ax.set_title('¿Qué meses suelen superar el nivel habitual?');ax.set_ylabel('Desviación estacional (ingresos)');ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:,.0f} mil'))
ax.text(0,-.23,'Componente estacional aditivo, período 12 · No representa el volumen observado ni prueba causas.',transform=ax.transAxes,fontsize=9)
save(fig,'03_estacionalidad')
# Tabla legible: conserva las magnitudes y definiciones del laboratorio.
fig,ax=plt.subplots(figsize=(12,4.1));ax.axis('off')
cell=[[name.replace('Vía — ','').replace('Región — ',''),f'{r.iloc[0]:.3f}',f'{r.iloc[1]:,.1f}',f'{r.iloc[2]:.3f}',(f'{r.iloc[3]:+.1f} %' if np.isfinite(r.iloc[3]) else 'No comparable')] for name,r in comp.iterrows()]
t=ax.table(cellText=cell,colLabels=['Serie','Fuerza\nestacional','Pendiente pre2020\n(ingresos/mes)','Volatilidad\nrelativa','2020 vs\n2019'],loc='center',cellLoc='center',colWidths=[.27,.15,.23,.17,.18]);t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2)
for (r,c),cl in t.get_celld().items():
 cl.set_edgecolor('#E2E8F0')
 if r==0:cl.set_facecolor(BLUE);cl.set_text_props(color='white',weight='bold')
ax.set_title('Vías y mercados: patrones distintos',pad=18)
ax.text(0,-.09,'Volatilidad: desviación de cambios porcentuales antes de 2020. Marítima: cobertura limitada; interpretar con cautela.',transform=ax.transAxes,fontsize=9)
save(fig,'04_comparacion_series')
# Recupera métricas YA calculadas del notebook: no inventa ni reentrena Prophet.
nb=json.loads((SRC/'Lab1_AnalisisSeriesTiempo.ipynb').read_text())
metricas=None
for cell in nb['cells']:
 if cell['cell_type']=='code' and 'filas_metricas = []' in ''.join(cell['source']):
  for output in cell.get('outputs',[]):
   h=''.join(output.get('data',{}).get('text/html',[]))
   soup=BeautifulSoup(h,'html.parser');table=soup.find('table')
   if table is None:continue
   headers=[th.get_text(strip=True) for th in table.find('thead').find_all('tr')[0].find_all('th')][1:]
   if 'Serie' not in headers:continue
   data=[[td.get_text(strip=True) for td in tr.find_all('td')] for tr in table.find('tbody').find_all('tr')]
   metricas=pd.DataFrame(data,columns=headers)
if metricas is None:raise RuntimeError('No se encontró la tabla de métricas guardada en el notebook.')
for col in ['MAE','RMSE']:metricas[col]=metricas[col].str.replace(',','',regex=False).astype(float)
metricas.to_csv(OUT/'metricas_modelos_lab1.csv',index=False)
total=metricas.loc[metricas.Serie.eq('Total visitantes')].sort_values('RMSE')
fig,ax=plt.subplots(figsize=(10,3.8));bars=ax.barh(total.Modelo,total.RMSE,color=[GREEN]+[GRAY]*(len(total)-1));ax.invert_yaxis();ax.set_xlim(0,total.RMSE.max()*1.4)
for bar,(_,r) in zip(bars,total.iterrows()):ax.text(bar.get_width()+total.RMSE.max()*.02,bar.get_y()+bar.get_height()/2,f'RMSE {r.RMSE:,.0f}\nMAE {r.MAE:,.0f}',va='center',fontsize=9)
ax.set_title('Para el total, un modelo sencillo tuvo menor error');ax.set_xlabel('RMSE en ingresos de visitantes · menor es mejor')
ax.text(0,-.25,'Prueba: abril 2021–junio 2026 · Métricas guardadas en el Lab 1, redondeadas a unidades.',transform=ax.transAxes,fontsize=9)
save(fig,'05_modelos_total')
cut=int(np.floor(.7*len(s)));train,test=s.iloc[:cut],s.iloc[cut:]
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 fit=SimpleExpSmoothing(train,initialization_method='estimated').fit(optimized=True)
pred=np.maximum(np.asarray(fit.forecast(len(test))),0)
mae=float(np.mean(abs(test.values-pred)));rmse=float(np.sqrt(np.mean((test.values-pred)**2)))
ref=total.loc[total.Modelo.eq('Suavizamiento exponencial')].iloc[0]
assert abs(mae-ref.MAE)<1 and abs(rmse-ref.RMSE)<1, 'El ajuste no reproduce el Lab 1'
# Modelo de innovaciones ETS(A,N,N): y_t=l_(t-1)+e_t; l_t=l_(t-1)+alpha*e_t.
# Varianza a h pasos: sigma² [1+(h-1)alpha²]. Solo usa residuos de entrenamiento.
alpha=float(fit.params['smoothing_level']);res=np.asarray(fit.resid)[1:]
sigma2=float(np.sum(res**2)/(len(res)-2));h=np.arange(1,len(test)+1)
se=np.sqrt(sigma2*(1+(h-1)*alpha**2));z=1.959963984540054
lower=np.maximum(pred-z*se,0);upper=np.maximum(pred+z*se,0)
forecast=pd.DataFrame({'Real':test,'Estimacion':pred,'Limite_inferior_95':lower,'Limite_superior_95':upper})
forecast.to_csv(OUT/'pronostico_total_intervalo95.csv',index_label='Fecha')
coverage=float(np.mean((test.values>=lower)&(test.values<=upper)))
fig,ax=plt.subplots(figsize=(12,4.5));ax.plot(train.iloc[-24:],color=GRAY,lw=1,label='Entrenamiento (últimos 24 meses)');ax.plot(test,color=BLUE,lw=1.6,label='Observado en prueba')
ax.fill_between(test.index,lower,upper,color=ORANGE,alpha=.17,label='Intervalo de predicción 95 % (condicional)');ax.plot(test.index,pred,color=ORANGE,lw=2,ls='--',label='Estimación: suavizamiento exponencial')
ax.axvline(test.index[0],color=GRAY,ls=':',lw=1);ax.set_title('Estimar no es adivinar: la recuperación desafió al modelo');axis(ax);ax.legend(fontsize=8,loc='upper left')
ax.text(0,-.24,'Modelo entrenado hasta marzo 2021 · Evaluación retrospectiva, no proyección posterior a junio 2026.\nBanda basada en errores de entrenamiento; no garantiza cubrir rupturas como la recuperación pandémica.',transform=ax.transAxes,fontsize=9)
save(fig,'06_pronostico_intervalo95')
assert len(s)==210 and len(train)==147 and len(test)==63
assert np.isfinite(forecast.values).all() and (lower<=pred).all() and (pred<=upper).all()
info={'modelo':'Suavizamiento exponencial simple / ETS(A,N,N)', 'alpha':alpha,'sigma2_entrenamiento':sigma2,'MAE':mae,'RMSE':rmse,'cobertura_observada_prueba':coverage,'nivel_nominal':.95,'inicio_prueba':str(test.index[0].date()),'fin_prueba':str(test.index[-1].date())}
(OUT/'diagnostico_intervalo.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
notes=f'''# Recursos para la infografía del Lab 9

Ejecutar desde cualquier ubicación: `python3 generar_visuales_lab9.py`.
Las rutas se resuelven respecto del script. No se modifica el Lab 1.
Cada gráfico se exporta en PNG (240 dpi) y SVG editable, dentro de recursos_lab9/.

## Indicadores
Las cifras se calculan del Excel: turistas + excursionistas. Son ingresos, no personas únicas.
Se proponen 2025 completo, caída 2020/2019, cambio 2025/2019 y participación terrestre del período.
No se compara 2026 parcial contra años completos. El perfil mensual usa descomposición aditiva,
no sumas mensuales con distinta cobertura. No atribuye los picos a aguinaldo o vacaciones sin evidencia.

## Métricas
La comparación de cinco algoritmos se extrae de la tabla guardada del notebook del Lab 1;
por tanto las métricas están redondeadas a unidades. No se reentrenan esos cinco algoritmos.
Se vuelve a ajustar el ganador del total y se verifica que MAE/RMSE coincidan con ese redondeo.

## Intervalo de predicción del 95 %
Ganador del total: suavizamiento exponencial simple, equivalente a ETS(A,N,N).
Se adopta el modelo de innovaciones aditivas normales, independientes, de media cero y varianza constante.
Para h pasos: var(error_h) = sigma² [1 + (h-1) alpha²].
Sigma² se estima con residuos de entrenamiento, excluyendo la primera inicialización y usando n-2 grados de libertad.
Los límites normales se recortan a cero para respetar el soporte de conteos; eso modifica la interpretación de las colas.
Es una banda puntual (no simultánea), condicional al modelo y parámetros estimados: no incorpora
incertidumbre de selección de modelo, parámetros o cambios estructurales. Ningún dato de prueba calibra la banda.
No es un intervalo de confianza de la media; es un intervalo de predicción para observaciones.

La cobertura observada en los 63 meses de prueba fue {coverage:.1%}, frente al 95 % nominal.
Esta cifra se informa para transparentar el efecto de la ruptura y no para ajustar retrospectivamente la banda.
El entrenamiento termina en marzo de 2021: no presentar este gráfico como pronóstico después de junio de 2026.

## Evidencia y archivos
comparacion_series.csv reproduce las definiciones del Lab 1; pendiente es crecimiento absoluto,
volatilidad es desviación estándar de cambios porcentuales pre2020 y fuerza estacional está entre 0 y 1.
La cobertura marítima es limitada y sus ceros no implican ausencia de actividad portuaria.
Los CSV y JSON permiten consultar cifras sin leer etiquetas de gráficos.
La propuesta PDF anterior todavía no incorpora estos indicadores definitivos ni esta metodología del intervalo.
'''
(OUT/'LEEME.md').write_text(notes)
print(json.dumps(kpis,ensure_ascii=False,indent=2));print(json.dumps(info,ensure_ascii=False,indent=2))
print('Mes estacional máximo:',months[int(pat.idxmax())-1],'; mínimo:',months[int(pat.idxmin())-1])
print('Recursos generados en',OUT)
