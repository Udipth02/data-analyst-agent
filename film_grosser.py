import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import json
import logging
from scipy.stats import pearsonr
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FilmAnalyzer:
    """A class to analyze highest-grossing films data from Wikipedia."""
    
    def __init__(self):
        self.target_url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        self.gross_threshold_2bn = 2_000_000_000
        self.gross_threshold_1_5bn = 1_500_000_000
        self.release_year_threshold = 2000
        self.df = None
    
    def fetch_data(self) -> BeautifulSoup:
        """Fetch and parse Wikipedia page."""
        try:
            response = requests.get(self.target_url, timeout=10)
            response.raise_for_status()
            logger.info("Successfully fetched Wikipedia page")
            return BeautifulSoup(response.content, "html.parser")
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data: {e}")
            raise
    
    def extract_table_data(self, soup: BeautifulSoup) -> pd.DataFrame:
        """Extract data from the Wikipedia table."""
        try:
            # Try multiple table selectors in case the structure changes
            table = (soup.find("table", {"class": "wikitable sortable"}) or 
                    soup.find("table", {"class": "wikitable"}) or
                    soup.find("table"))
            
            if not table:
                raise ValueError("Could not find data table on the page")
            
            rows = table.find_all("tr")
            if len(rows) < 2:
                raise ValueError("Table appears to be empty or malformed")
            
            # Extract headers
            headers = [th.text.strip() for th in rows[0].find_all(["th", "td"])]
            logger.info(f"Found headers: {headers}")
            
            # Extract data rows
            data = []
            for row in rows[1:]:
                cols = row.find_all(["td", "th"])
                if len(cols) >= len(headers):  # Ensure we have enough columns
                    row_data = [col.text.strip() for col in cols[:len(headers)]]
                    data.append(row_data)
            
            df = pd.DataFrame(data, columns=headers)
            logger.info(f"Extracted {len(df)} rows of data")
            return df
            
        except Exception as e:
            logger.error(f"Failed to extract table data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the extracted data."""
        try:
            # Create a copy to avoid modifying original
            df_clean = df.copy()
            
            # Find the gross revenue column (could be named differently)
            gross_col = None
            possible_gross_names = ['Gross', 'Worldwide gross', 'Box office', 'Revenue', 'Total gross']
            for col_name in possible_gross_names:
                if col_name in df_clean.columns:
                    gross_col = col_name
                    break
            
            if not gross_col:
                # Try to find column containing 'gross' in the name
                gross_col = next((col for col in df_clean.columns if 'gross' in col.lower()), None)
            
            if gross_col:
                logger.info(f"Using '{gross_col}' as gross revenue column")
                df_clean['Gross'] = (df_clean[gross_col]
                                   .str.replace(r'[$,\s]', '', regex=True)
                                   .str.replace(r'[^\d.]', '', regex=True))
                df_clean['Gross'] = pd.to_numeric(df_clean['Gross'], errors='coerce')
            else:
                raise ValueError(f"Could not find gross revenue column. Available columns: {list(df_clean.columns)}")
            
            # Find and clean year column
            year_col = None
            if 'Year' in df_clean.columns:
                year_col = 'Year'
            else:
                year_col = next((col for col in df_clean.columns if 'year' in col.lower()), None)
            
            if year_col:
                logger.info(f"Using '{year_col}' as year column")
                df_clean['Year'] = pd.to_numeric(
                    df_clean[year_col].str.extract(r'(\d{4})')[0], 
                    errors='coerce'
                )
            else:
                logger.warning("No year column found")
                df_clean['Year'] = None
            
            # Find and clean title column
            title_col = None
            possible_title_names = ['Film', 'Title', 'Movie', 'Name']
            for col_name in possible_title_names:
                if col_name in df_clean.columns:
                    title_col = col_name
                    break
            
            if title_col and title_col != 'Film':
                df_clean['Film'] = df_clean[title_col]
                logger.info(f"Using '{title_col}' as film title column")
            
            # Clean rank and peak columns
            for col in ['Rank', 'Peak']:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Fill missing peak values with rank if available
            if 'Peak' in df_clean.columns and 'Rank' in df_clean.columns:
                df_clean['Peak'] = df_clean['Peak'].fillna(df_clean['Rank'])
            
            # Remove rows with critical missing data
            required_cols = ['Gross']
            if 'Year' in df_clean.columns:
                required_cols.append('Year')
            
            df_clean = df_clean.dropna(subset=required_cols)
            
            logger.info(f"Cleaned data: {len(df_clean)} valid rows remaining")
            logger.info(f"Final columns: {list(df_clean.columns)}")
            return df_clean
            
        except Exception as e:
            logger.error(f"Failed to clean data: {e}")
            raise
    
    def analyze_data(self) -> List[str]:
        """Perform the main analysis and return results."""
        if self.df is None:
            raise ValueError("No data available for analysis")
        
        results = []
        
        try:
            # 1. Movies over $2B before 2000
            two_bn_before_2000 = len(self.df[
                (self.df['Gross'] >= self.gross_threshold_2bn) & 
                (self.df['Year'] < self.release_year_threshold)
            ])
            results.append(str(two_bn_before_2000))
            logger.info(f"Found {two_bn_before_2000} movies over $2B before 2000")
            
            # 2. Earliest $1.5B movie
            big_earners = self.df[self.df['Gross'] >= self.gross_threshold_1_5bn]
            if not big_earners.empty and 'Year' in self.df.columns:
                earliest_idx = big_earners['Year'].idxmin()
                film_col = 'Film' if 'Film' in big_earners.columns else 'Title'
                if film_col in big_earners.columns:
                    earliest_film = big_earners.loc[earliest_idx, film_col]
                    earliest_year = big_earners.loc[earliest_idx, 'Year']
                    results.append(f"{earliest_film} ({int(earliest_year)})")
                    logger.info(f"Earliest $1.5B film: {earliest_film} ({earliest_year})")
                else:
                    results.append("Film title column not found")
            else:
                results.append("No films found over $1.5B or year data missing")
            
            # 3. Correlation between Rank and Peak
            if 'Rank' in self.df.columns and 'Peak' in self.df.columns:
                valid_data = self.df[['Rank', 'Peak']].dropna()
                if len(valid_data) > 1:
                    correlation, p_value = pearsonr(valid_data['Rank'], valid_data['Peak'])
                    results.append(f"Correlation: {correlation:.3f} (p-value: {p_value:.3f})")
                    logger.info(f"Rank-Peak correlation: {correlation:.3f}")
                else:
                    results.append("Insufficient data for correlation")
            else:
                results.append("Rank or Peak columns not available")
            
            # 4. Generate scatter plot
            plot_uri = self.create_scatter_plot()
            results.append(plot_uri)
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def create_scatter_plot(self) -> str:
        """Create scatter plot with regression line."""
        try:
            plt.style.use('default')  # Ensure consistent styling
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data
            plot_data = self.df[['Rank', 'Peak']].dropna()
            
            if len(plot_data) < 2:
                # Create a placeholder plot if insufficient data
                ax.text(0.5, 0.5, 'Insufficient data for scatter plot', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel("Rank")
                ax.set_ylabel("Peak")
                ax.set_title("Rank vs. Peak (Insufficient Data)")
            else:
                # Create scatter plot
                ax.scatter(plot_data['Rank'], plot_data['Peak'], 
                          alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5)
                
                # Add regression line
                coeffs = np.polyfit(plot_data['Rank'], plot_data['Peak'], 1)
                x_line = np.linspace(plot_data['Rank'].min(), plot_data['Rank'].max(), 100)
                y_line = np.polyval(coeffs, x_line)
                ax.plot(x_line, y_line, color='red', linestyle='--', 
                       linewidth=2, label=f'Regression line (slope: {coeffs[0]:.2f})')
                
                ax.set_xlabel("Rank", fontsize=12)
                ax.set_ylabel("Peak Position", fontsize=12)
                ax.set_title("Rank vs. Peak Position", fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Failed to create plot: {e}")
            return "Plot generation failed"
    
    def run_analysis(self) -> List[str]:
        """Main method to run the complete analysis pipeline."""
        try:
            # Fetch and process data
            soup = self.fetch_data()
            raw_df = self.extract_table_data(soup)
            self.df = self.clean_data(raw_df)
            
            # Run analysis
            results = self.analyze_data()
            
            logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            return [f"Analysis failed: {str(e)}"]

# Usage
if __name__ == "__main__":
    analyzer = FilmAnalyzer()
    results = analyzer.run_analysis()
    
    # Output as proper JSON
    print(json.dumps(results, indent=2))