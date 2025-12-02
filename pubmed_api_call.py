#!/usr/bin/env python3
"""
Real-time PubMed Publications Analysis: LLMs and VLMs in Radiology
Fetches live data from PubMed E-utilities API
Author: PubMed Analysis Tool
"""

import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import time
import json
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple

class PubMedAnalyzer:
    """Class to fetch and analyze PubMed publication data"""
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = "..."  #
        self.api_key = "..."  
        self.delay = 0.34  
        
    def search_pubmed(self, query: str, date_from: str, date_to: str) -> int:
        search_url = f"{self.base_url}esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'datetype': 'pdat',  # Publication date
            'mindate': date_from,
            'maxdate': date_to,
            'retmode': 'json',
            'retmax': '0',  # We only need the count
            'email': self.email
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
            
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract count from response
            count = int(data['esearchresult']['count'])
            
            # Print progress
            print(f"  Query: {query[:50]}, Period: {date_from} to {date_to}, Count: {count}")
            
            # Respect rate limits
            time.sleep(self.delay)
            
            return count
            
        except Exception as e:
            print(f"  Error fetching data: {e}")
            return 0
    
    def fetch_yearly_data(self, start_year: int = 2016, end_year: int = 2025) -> Dict:
        results = {
            'years': list(range(start_year, end_year + 1)),
            'llm_radiology': [],
            'llm_vlm_radiology': [],
            'vlm_all': [],
            'ai_radiology': []
        }
        
        for year in results['years']:
            print(f"\nYear {year}:")
            print("-" * 40)
            
            # Set date range for the year
            date_from = f"{year}/01/01"
            
            # For current year, use today's date
            if year == datetime.now().year:
                date_to = datetime.now().strftime("%Y/%m/%d")
            else:
                date_to = f"{year}/12/31"
            
            # Query 1: LLM radiology
            count_ai_rad = self.search_pubmed(
                '"AI" AND (radiology OR radiography OR "medical imaging" OR "Radiology")',
                date_from, date_to) 
            results['ai_radiology'].append(count_ai_rad)
            count_llm = self.search_pubmed(
                '"LLM" AND (radiology OR radiography OR "medical imaging" OR "Radiology") OR "Large Language Models"',
                date_from, date_to
            )
            results['llm_radiology'].append(count_llm)
            
            # Query 2: LLM OR VLM radiology
            count_llm_vlm = self.search_pubmed(
                '("LLM" OR "VLM" OR "large language model" OR "vision language model" OR "Large Language Models" OR "Vision Language Models") AND (radiology OR radiography OR "medical imaging" OR "Radiology")',
                date_from, date_to
            )
            results['llm_vlm_radiology'].append(count_llm_vlm)
            
            # Query 3: VLM all fields
            count_vlm = self.search_pubmed(
                '"VLM" OR "vision language model OR "Vision Language Model OR "vision-language model" OR "Vision-Language Model" OR "Vision Language Models"',
                date_from, date_to
            )
            results['vlm_all'].append(count_vlm)
        
        # Also fetch total counts for the entire period
        print("\n" + "="*70)
        print("FETCHING TOTAL COUNTS FOR ENTIRE PERIOD")
        print("="*70)
        
        total_date_from = f"{start_year}/01/01"
        total_date_to = datetime.now().strftime("%Y/%m/%d")
        
        results['total_llm'] = self.search_pubmed(
            '"LLM" AND (radiology OR radiography OR "medical imaging")',
            total_date_from, total_date_to
        )
        
        results['total_llm_vlm'] = self.search_pubmed(
            '("LLM" OR "VLM" OR "large language model" OR "vision language model") AND (radiology OR radiography OR "medical imaging")',
            total_date_from, total_date_to
        )
        
        results['total_vlm'] = self.search_pubmed(
            '"VLM" OR "vision language model"',
            total_date_from, total_date_to
        )
        
        results['total_ai_rad'] = self.search_pubmed(
            '"AI" AND (radiology OR radiography OR "medical imaging")',
            total_date_from, total_date_to
        )
        return results
    
    def create_visualization(self, data: Dict, save_path: str = "./thesis/pubmed_outputs/") -> pd.DataFrame:
        """
        Create comprehensive visualization from the fetched data
        """
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        years = data['years']
        llm_counts = data['llm_radiology']
        llm_vlm_counts = data['llm_vlm_radiology']
        vlm_counts = data['vlm_all']
        ai_rad_counts = data['ai_radiology']
        
        # Create the comprehensive figure
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # Main time series plot
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot lines with markers
        line1 = ax1.plot(years, llm_counts, 'o-', linewidth=3, markersize=10, 
                         color='#0077BE', label='LLMs in Radiology', 
                         markeredgecolor='white', markeredgewidth=2, alpha=0.9)
        
        line2 = ax1.plot(years, llm_vlm_counts, 's-', linewidth=3, markersize=10, 
                         color='#FF6B35', label='LLMs or VLMs in Radiology', 
                         markeredgecolor='white', markeredgewidth=2, alpha=0.9)
        
        line3 = ax1.plot(years, vlm_counts, '^-', linewidth=3, markersize=10, 
                         color='#2E7D32', label='VLMs (All Fields)', 
                         markeredgecolor='white', markeredgewidth=2, alpha=0.9)
        line4 = ax1.plot(years, ai_rad_counts, 'D--', linewidth=2, markersize=8, 
                         color='#8B008B', label='AI in Radiology', 
                         markeredgecolor='white', markeredgewidth=2, alpha=0.7)
        
        
        # Fill areas under curves
        ax1.fill_between(years, 0, llm_counts, alpha=0.15, color='#0077BE')
        ax1.fill_between(years, 0, llm_vlm_counts, alpha=0.15, color='#FF6B35')
        ax1.fill_between(years, 0, vlm_counts, alpha=0.15, color='#2E7D32')
        ax1.fill_between(years, 0, ai_rad_counts, alpha=0.1, color='#8B008B')
        
        # Styling
        ax1.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of PubMed Publications', fontsize=14, fontweight='bold')
        
        current_date = datetime.now().strftime("%B %d, %Y")
        ax1.set_title(f'Rise of Large Language and Vision-Language Models in Radiology Literature \n (PubMed Publications)', 
                      fontsize=16, fontweight='bold', pad=20)
        
        # Enhanced grid
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.set_axisbelow(True)
        
        # Legend
        ax1.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, 
                   shadow=True, framealpha=0.95, borderpad=1)
        
        # Set x-axis
        ax1.set_xticks(years)
        ax1.set_xticklabels(years, rotation=0)
        ax1.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax1.set_ylim(bottom=-5)
        
        # Add annotations for key milestones if we have significant growth
        if len(years) > 5 and llm_counts[-3] > 0:  # Check 2023 data
            # ChatGPT release annotation
            ax1.annotate('ChatGPT\nRelease', xy=(2022.9, llm_counts[years.index(2022)] if 2022 in years else 0), 
                         xytext=(2022.5, max(llm_counts)*0.3),
                         arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1.5),
                         fontsize=10, color='gray', ha='center', style='italic',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        
        # Add value labels for peaks
        for i, year in enumerate(years):
            if llm_counts[i] > 50:  # Only label significant values
                ax1.annotate(str(llm_counts[i]), xy=(year, llm_counts[i]), 
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold')
            if llm_vlm_counts[i] > 60:
                ax1.annotate(str(llm_vlm_counts[i]), xy=(year, llm_vlm_counts[i]), 
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold')
        
        # Highlight current year as partial if it's the last year
        if years[-1] == datetime.now().year:
            ax1.axvspan(years[-1] - 0.5, years[-1] + 0.5, alpha=0.1, color='orange')
            ax1.text(years[-1], ax1.get_ylim()[1]*0.95, 'Partial\nYear', 
                     ha='center', va='top', fontsize=9, style='italic', color='darkorange')
        
        # Growth rate plot
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Calculate year-over-year growth rates
        def calculate_growth_rates(counts):
            rates = []
            for i in range(1, len(counts)):
                if counts[i-1] > 0:
                    rate = ((counts[i] - counts[i-1]) / counts[i-1]) * 100
                else:
                    rate = 100 if counts[i] > 0 else 0
                rates.append(min(rate, 500))  # Cap at 500% for visualization
            return rates
        
        plt.tight_layout()
        
        # Save figures
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #make directory if not exists
        import os
        path = "./thesis/pubmed_outputs/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f'{save_path}pubmed_realtime_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}pubmed_realtime_analysis_{timestamp}.pdf', bbox_inches='tight')
        
        # Create data summary CSV
        summary_df = pd.DataFrame({
            'Year': years,
            'LLMs_in_Radiology': llm_counts,
            'LLMs_or_VLMs_in_Radiology': llm_vlm_counts,
            'VLMs_All_Fields': vlm_counts,
        })
        
        summary_df.to_csv(f'{save_path}pubmed_realtime_data_{timestamp}.csv', index=False)
        
        # Save raw data as JSON for future reference
        with open(f'{save_path}pubmed_raw_data_{timestamp}.json', 'w') as f:
            json.dump(data, f, indent=2)

        max_llm_year = years[llm_counts.index(max(llm_counts))]
        max_llm_vlm_year = years[llm_vlm_counts.index(max(llm_vlm_counts))]
        plt.show()
        
        return summary_df

def main():
    analyzer = PubMedAnalyzer()
    data = analyzer.fetch_yearly_data(start_year=2015, end_year=2025)
    print("\nCreating visualizations...")
    summary = analyzer.create_visualization(data, save_path="./thesis/pubmed_outputs/")

if __name__ == "__main__":
    main()