import pandas as pd
import requests
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

url = 'https://en.wikipedia.org/wiki/List_of_highest-grossing_films'
response = requests.get(url)
df = pd.read_html(io.StringIO(response.text))[0]
df.columns = df.columns.str.replace('\[.*\]', '', regex=True)
df.columns = ['Rank', 'Title', 'Worldwide gross', 'Year', 'Notes', 'Ref']
df = df[['Rank', 'Title', 'Worldwide gross', 'Year']]
df['Worldwide gross'] = df['Worldwide gross'].replace({'\$': '', ',': ''}, regex=True).astype(float, errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

billion_2_movies_before_2000 = df[(df['Worldwide gross'] >= 2000000000) & (df['Year'] < 2000)]
count_2bn = len(billion_2_movies_before_2000)

earliest_over_1_5bn = df[df['Worldwide gross'] > 1500000000].sort_values(by='Year').iloc[0]['Title']

correlation = df['Rank'].corr(df['Worldwide gross'])

sns.set_theme()
plt.figure(figsize=(10, 6))
sns.regplot(x='Rank', y='Worldwide gross', data=df, scatter_kws={'s': 5})
plt.xlabel('Rank')
plt.ylabel('Worldwide Gross')
plt.title('Rank vs Worldwide Gross')
buf = BytesIO()
plt.savefig(buf, format='png')
data = base64.b64encode(buf.getbuffer()).decode()
uri = 'data:image/png;base64,' + data
plt.close()


print([str(count_2bn), earliest_over_1_5bn, str(correlation), uri])