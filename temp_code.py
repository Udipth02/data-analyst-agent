import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def analyze_high_courts():
    duckdb.execute("INSTALL httpfs; LOAD httpfs;")
    duckdb.execute("INSTALL parquet; LOAD parquet;")
    query = """
    SELECT * FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
    """
    df = duckdb.query(query).df()
    df['decision_date'] = pd.to_datetime(df['decision_date'])
    df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], dayfirst=True)
    df['delay_days'] = (df['decision_date'] - df['date_of_registration']).dt.days
    high_court_counts = df[df['decision_date'].dt.year.between(2019, 2022)].groupby('court')['cnr'].count()
    most_cases_court = high_court_counts.idxmax()
    df_filtered = df[df['court'] == most_cases_court]
    df_filtered['year'] = df_filtered['decision_date'].dt.year
    df_filtered['days_delay'] = (df_filtered['decision_date'] - df_filtered['date_of_registration']).dt.days
    df_filtered['date_of_registration'] = pd.to_datetime(df_filtered['date_of_registration'], errors='coerce', dayfirst=True)
    df_filtered = df_filtered.dropna(subset=['date_of_registration'])
    df_filtered['year'] = df_filtered['decision_date'].dt.year
    df_filtered['days_delay'] = (df_filtered['decision_date'] - df_filtered['date_of_registration']).dt.days
    slope = None
    if len(df_filtered) >= 2:
        from sklearn.linear_model import LinearRegression
        X = df_filtered[['year']]
        y = df_filtered['days_delay']
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
    plt.figure(figsize=(8,6))
    sns.regplot(x='year', y='days_delay', data=df_filtered)
    plt.xlabel('Year of Decision')
    plt.ylabel('Days of Delay')
    plt.title(f'Delay over Years for {most_cases_court}')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    data_uri = f"data:image/png;base64,{base64_str}"
    answer = {
        "Which high court disposed the most cases from 2019 - 2022?": most_cases_court,
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": slope,
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": data_uri
    }
    return answer

if __name__ == "__main__":
    result = analyze_high_courts()
    print(result)