import requests
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

response = requests.get('https://en.wikipedia.org/wiki/List_of_highest-grossing_films')
df = pd.read_html(io.StringIO(response.text))[0]
df.columns = df.columns.droplevel()
df['Worldwide gross'] = df['Worldwide gross'].str.replace(r'[$,T\n]', '', regex=True).astype(float)
df['Year'] = df['Year'].astype(int)

q1 = len(df[(df['Worldwide gross'] >= 2000) & (df['Year'] < 2000)])
q2_df = df[df['Worldwide gross'] > 1500].sort_values(by='Year').iloc[0]
q2 = q2_df['Film']

correlation = df['Rank'].corr(df['Worldwide gross'])

sns.set_theme()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rank', y='Worldwide gross', data=df)
slope, intercept, r_value, p_value, std_err = linregress(df['Rank'], df['Worldwide gross'])
sns.lineplot(x=df['Rank'], y=slope * df['Rank'] + intercept, color='red', linestyle='--')
plt.xlabel('Rank')
plt.ylabel('Worldwide Gross (Billions)')
plt.title('Rank vs. Worldwide Gross with Regression Line')
buf = io.BytesIO()
plt.savefig(buf, format='png')
data = base64.b64encode(buf.getbuffer()).decode('ascii')
img_uri = f'data:image/png;base64,{data}'
plt.close()
print(f'["{q1}", "{q2}", "{correlation}", "{img_uri}"]')