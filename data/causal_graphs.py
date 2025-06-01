import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq

# Set random seed for reproducibility
np.random.seed(42)

# Define entity domains
domains = {
    "politics": ["election", "policy", "sanction", "treaty", "scandal"],
    "health": ["pandemic", "vaccine", "treatment", "outbreak", "mortality"],
    "economics": ["inflation", "recession", "trade", "employment", "gdp"],
    "technology": ["ai", "blockchain", "quantum", "5g", "cybersecurity"]
}

# Generate causal relationships
data = []
entity_id = 0
relationship_id = 0

for domain, entities in domains.items():
    for entity in entities:
        # Create main entity
        entity_id += 1
        data.append({
            "entity_id": entity_id,
            "entity_name": entity,
            "domain": domain,
            "description": f"The {entity} entity in {domain} domain",
            "first_observed": "2020-01-01",
            "last_updated": "2025-06-01",
            "embedding": np.random.randn(64).tolist(),
            "is_causal_node": True
        })
        
        # Create 3-5 causal relationships per entity
        for _ in range(np.random.randint(3, 6)):
            relationship_id += 1
            target_entity = np.random.choice(entities)
            if target_entity == entity:
                continue  # Skip self-relationships
                
            # Time-varying causal strength
            causal_strength = []
            for year in range(2020, 2026):
                for month in range(1, 13):
                    if month % 12 == 0:
                        year += 1
                    date = f"{year}-{month:02d}-01"
                    # Base strength with some temporal variation
                    strength = np.clip(
                        np.random.normal(0.5, 0.2) + 0.1 * np.sin(year - 2020),
                        0.1, 0.9
                    )
                    causal_strength.append({
                        "date": date,
                        "strength": float(strength),
                        "confidence": float(np.random.uniform(0.7, 0.95))
                    })
            
            data.append({
                "entity_id": relationship_id + 10000,  # Offset for relationships
                "entity_name": f"causal_relation_{relationship_id}",
                "domain": domain,
                "description": f"Causal relationship between {entity} and {target_entity}",
                "source_entity": entity,
                "target_entity": target_entity,
                "relationship_type": np.random.choice(["direct", "indirect", "moderated"]),
                "causal_strength": causal_strength,
                "first_observed": "2020-01-01",
                "last_updated": "2025-06-01",
                "embedding": np.random.randn(64).tolist(),
                "is_causal_node": False,
                "metadata": {
                    "sources": [
                        f"source_{np.random.randint(1, 4)}",
                        f"study_{np.random.randint(1, 6)}"
                    ],
                    "p_value": float(np.random.uniform(0, 0.05)),
                    "effect_size": float(np.random.uniform(0.1, 0.8))
                }
            })

# Create DataFrame
df = pd.DataFrame(data)

# Add temporal evolution features
def add_temporal_features(row):
    if row['is_causal_node']:
        return row
    
    # Simulate concept drift
    for i, strength in enumerate(row['causal_strength']):
        if i > len(row['causal_strength']) // 2:
            # Apply drift to later time periods
            row['causal_strength'][i]['strength'] *= np.random.uniform(0.8, 1.2)
    return row

df = df.apply(add_temporal_features, axis=1)

# Save to Parquet with advanced features
table = pa.Table.from_pandas(df)
pq.write_table(table, 'data/causal_graphs.parquet', 
              compression='ZSTD', 
              version='2.6',
              data_page_version='2.0',
              use_dictionary=True,
              write_statistics=True)

print(f"Saved causal graphs with {len(df)} entries ({(table.nbytes / (1024*1024)):.2f} MB)")
