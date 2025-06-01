import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
import random

# Constants
NUM_ENTRIES = 10_000  # Substantial dataset
ENTITIES = ['COVID-19', 'Climate Change', 'AI Regulation', 'Global Economy', 'Ukraine War', 
            'Quantum Computing', 'CRISPR', 'Renewable Energy', 'Space Exploration', 'Cryptocurrency']
CATEGORIES = ['Health', 'Technology', 'Politics', 'Economy', 'Environment', 'Science', 'Military']
COUNTRIES = ['US', 'China', 'UK', 'Germany', 'France', 'Japan', 'India', 'Brazil', 'Canada', 'Australia']
DATE_RANGE = pd.date_range(start='2020-01-01', end='2025-12-31')

def generate_temporal_kb():
    data = {
        'entity': np.random.choice(ENTITIES, NUM_ENTRIES),
        'category': np.random.choice(CATEGORIES, NUM_ENTRIES),
        'country': np.random.choice(COUNTRIES + [None], NUM_ENTRIES, p=[0.7] + [0.03]*10),
        'timestamp': np.random.choice(DATE_RANGE, NUM_ENTRIES),
        'fact': [generate_fact(i) for i in range(NUM_ENTRIES)],
        'source_confidence': np.round(np.random.uniform(0.5, 1.0, NUM_ENTRIES), 2),
        'embedding': [np.random.randn(768).tolist() for _ in range(NUM_ENTRIES)],
        'metadata': [generate_metadata(i) for i in range(NUM_ENTRIES)]
    }
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def generate_fact(index):
    entities = random.sample(ENTITIES, k=random.randint(1, 2))
    year = 2020 + index % 6
    templates = [
        f"In {year}, {entities[0]} was found to have significant impact on {random.choice(['public health', 'global markets', 'technology'])}",
        f"Between {year} and {year+1}, {entities[0]} policies changed dramatically in {random.choice(COUNTRIES)}",
        f"New research in {year} revealed unexpected connections between {entities[0]} and {entities[1] if len(entities)>1 else 'other factors'}",
        f"The {entities[0]} {random.choice(['outbreak', 'crisis', 'boom'])} of {year} led to {random.choice(['policy changes', 'economic shifts', 'technological advancements'])}",
        f"By Q{random.randint(1,4)} {year}, {entities[0]} had {random.choice(['increased', 'decreased'])} by {random.randint(5,95)}% compared to previous period"
    ]
    return random.choice(templates)

def generate_metadata(index):
    return {
        'source': random.choice(['WHO', 'UN', 'CDC', 'IMF', 'World Bank', 'Nature', 'Science']),
        'citations': random.randint(1, 50),
        'update_frequency': random.choice(['daily', 'weekly', 'monthly', 'yearly']),
        'verified': random.choice([True, False]),
        'related_entities': random.sample(ENTITIES, k=random.randint(1, 3))
    }

# Generate and save the knowledge base
kb_df = generate_temporal_kb()

# Save as Parquet with multiple row groups for efficient querying
table = pa.Table.from_pandas(kb_df)
pq.write_table(
    table, 
    'data/temporal_kb.parquet',
    row_group_size=1000,
    compression='ZSTD',
    version='2.6'
)

print(f"Generated temporal knowledge base with {len(kb_df)} entries")
print("Sample entries:")
print(kb_df.head(3).to_markdown())
