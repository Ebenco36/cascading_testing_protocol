"""
Generate Sample AMR Surveillance Dataset (FIXED VERSION)
=========================================================

Creates LARGER realistic AMR dataset with better antibiotic coverage.
This version generates 5,000 rows to ensure sufficient data for:
  - Filtering without removing all antibiotics
  - Statistical testing
  - Cascade discovery

Features:
  - 5,000 realistic episodes (vs 1,000 in original)
  - More balanced pathogen distribution
  - Better antibiotic testing coverage
  - Realistic resistance patterns
  - Complete format compliance

Usage:
  python generate_sample_dataset_FIXED.py
  
Output:
  sample_amr_surveillance_FIXED.parquet (much larger, ~250 MB)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration - INCREASED FOR BETTER DATA
n_rows = 5000  # INCREASED from 1000

# Define dimensions
pathogens = [
    'Escherichia coli',
    'Staphylococcus aureus',
    'Klebsiella pneumoniae',
    'Streptococcus pyogenes',
    'Pseudomonas aeruginosa',
    'Acinetobacter baumannii',
]

pathogen_groups = {
    'Escherichia coli': 'Gram-negative: Enterobacterales',
    'Staphylococcus aureus': 'Gram-positive: Staphylococci',
    'Klebsiella pneumoniae': 'Gram-negative: Enterobacterales',
    'Streptococcus pyogenes': 'Gram-positive: Streptococci',
    'Pseudomonas aeruginosa': 'Gram-negative: Pseudomonadales',
    'Acinetobacter baumannii': 'Gram-negative: Moraxellales',
}

gram_types = {
    'Escherichia coli': 'Gram-negative',
    'Staphylococcus aureus': 'Gram-positive',
    'Klebsiella pneumoniae': 'Gram-negative',
    'Streptococcus pyogenes': 'Gram-positive',
    'Pseudomonas aeruginosa': 'Gram-negative',
    'Acinetobacter baumannii': 'Gram-negative',
}

pathogen_genus = {
    'Escherichia coli': 'Escherichia',
    'Staphylococcus aureus': 'Staphylococcus',
    'Klebsiella pneumoniae': 'Klebsiella',
    'Streptococcus pyogenes': 'Streptococcus',
    'Pseudomonas aeruginosa': 'Pseudomonas',
    'Acinetobacter baumannii': 'Acinetobacter',
}

specimen_types = ['Urine', 'Blood', 'Wound', 'Respiratory', 'Cerebrospinal fluid', 'Other body fluids']
sexes = ['Man', 'Woman']
age_groups = ['18-29', '30-44', '45-59', '60-74', '75-89', '90+']
care_types = ['In-Patient', 'Out-Patient', 'Ambulatory']
hospital_levels = ['Level 1 - Primary Care', 'Level 2 - Regular Care', 'Level 3 - Specialized Care']
ward_types = ['Intensive Care Unit', 'Normal Ward', 'Emergency Department', 'Surgical Ward']
regions = ['North', 'South', 'East', 'West', 'Central']
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
labs = [f'Lab {i}' for i in range(1, 6)]

# Core antibiotics (most commonly tested)
core_antibiotics = [
    'AMC - Amoxicillin/clavulanic acid',
    'AMK - Amikacin',
    'AMP - Ampicillin',
    'AMS - Ampicillin/Sulbactam',
    'CAZ - Ceftazidim',
    'CIP - Ciprofloxacin',
    'CRO - Ceftriaxon',
    'CTX - Cefotaxim',
    'GEN - Gentamicin',
    'IMP - Imipenem',
    'LEV - Levofloxacin',
    'MER - Meropenem',
    'PEN - Penicillin',
    'PIT - Piperacillin/Tazobactam',
    'SXT - Co-Trimoxazol',
    'VAN - Vancomycin',
]

# Extended antibiotic list
all_antibiotics = core_antibiotics + [
    'AZT - Aztreonam',
    'CEP - Cefepim',
    'CEZ - Cefazolin',
    'CLI - Clindamycin',
    'CMP - Chloramphenicol',
    'COL - Colistin',
    'CXM - Cefuroxim',
    'DOX - Doxycyclin',
    'DPT - Daptomycin',
    'ERT - Ertapenem',
    'ERY - Erythromycin',
    'FLU - Flucloxacillin',
    'FOS - Fosfomycin',
    'FUS - Fusidic acid',
    'LIZ - Linezolid',
    'MOX - Moxifloxacin',
    'MUP - Mupirocin',
    'NFT - Nitrofurantoin',
    'OXA - Oxacillin',
    'PIP - Piperacillin',
    'RAM - Rifampicin',
    'TET - Tetracyclin',
    'TGC - Tigecyclin',
    'TOB - Tobramycin',
    'TPL - Teicoplanin',
    'TRP - Trimethoprim',
]

def generate_row(row_id):
    """Generate a single data row."""
    pathogen = np.random.choice(pathogens)
    date = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 730))
    month = date.month
    
    # Initialize row
    row = {
        'OrgType': 'Hospital',
        'Anonymized_Lab': np.random.choice(labs),
        'Pathogen': pathogen,
        'PathogengroupL1': pathogen_groups[pathogen],
        'GramType': gram_types[pathogen],
        'Sex': np.random.choice(sexes),
        'Date': date.strftime('%Y-%m-%d'),
        'PathogenGenus': pathogen_genus[pathogen],
        'TextMaterialgroupRkiL0': np.random.choice(specimen_types),
        'AgeGroup': np.random.choice(age_groups),
        'AgeRange': np.random.choice(['18-44', '45-64', '65-79', '80-89', '90+']),
        'CareType': np.random.choice(care_types),
        'ARS_HospitalLevelManual': np.random.choice(hospital_levels),
        'ARS_WardType': np.random.choice(ward_types),
        'ARS_Region': np.random.choice(regions),
        'Year': date.year,
        'Month': month,
        'MonthName': date.strftime('%B'),
        'YearMonth': date.strftime('%Y-%m'),
        'SeasonCode': (month - 1) // 3 + 1,
        'SeasonName': seasons[(month - 1) // 3],
        'TotalAntibioticsTested': np.random.randint(15, 28),  # INCREASED testing
        'PathogenSummary': pathogen_genus[pathogen] + 'ales' if 'ales' not in pathogen_groups[pathogen] else 'Streptococci',
        'BroadAgeGroup': 'Adult' if int(row_id) % 2 == 0 else 'Elderly',
        'HighLevelAgeRange': '65+' if int(row_id) % 3 == 0 else '18-64',
        'Hospital_Priority': np.random.choice(['High', 'Medium', 'Low']),
        'Care_Complexity': np.random.choice(['Tertiary & Specialized', 'Primary/Secondary']),
        'Facility_Function': np.random.choice(['Referral/Advanced Hospital', 'General Hospital']),
        'CSQ': 'Erstisolat',
        'CSQMG': 'Erstisolat',
        'CSY': 'Erstisolat',
        'CSYMG': 'Erstisolat',
        'IsSpecificlyExcluded_Screening': 'False',
        'TypeOrganisation': 'Krankenhaus',
        'IsSpecificlyExcluded_Pathogen': 'False',
        'IsSpecificlyExcluded_PathogenevidenceNegative': 'False',
    }
    
    # Generate antibiotic testing pattern - IMPROVED COVERAGE
    # More episodes test more antibiotics to ensure coverage passes filtering
    tested_count = np.random.randint(15, 26)  # Test 15-25 out of 44 antibiotics
    tested_antibiotics = set(np.random.choice(all_antibiotics, size=tested_count, replace=False))
    
    # Ensure core antibiotics are often tested
    for ab in np.random.choice(core_antibiotics, size=min(12, tested_count), replace=False):
        tested_antibiotics.add(ab)
    
    # Add tested columns
    for ab in all_antibiotics:
        row[f'{ab}_Tested'] = 1 if ab in tested_antibiotics else 0
    
    # Generate outcomes for tested antibiotics
    # Pathogen-specific resistance patterns - REALISTIC RATES
    resistance_rate = {
        'Escherichia coli': 0.35,
        'Staphylococcus aureus': 0.25,
        'Klebsiella pneumoniae': 0.40,
        'Streptococcus pyogenes': 0.15,
        'Pseudomonas aeruginosa': 0.50,
        'Acinetobacter baumannii': 0.60,
    }[pathogen]
    
    for ab in all_antibiotics:
        if ab in tested_antibiotics:
            rand = np.random.random()
            if rand < resistance_rate:
                outcome = 'R'
            elif rand < resistance_rate + 0.10:
                outcome = 'I'
            else:
                outcome = 'S'
        else:
            outcome = ''  # Not tested
        
        row[f'{ab}_Outcome'] = outcome
    
    return row

# Generate dataset
print("Generating LARGER sample AMR surveillance dataset...")
print(f"Target: {n_rows} rows (increased from 1000)")
rows = []
for i in range(n_rows):
    if (i + 1) % 500 == 0:
        print(f"  Generated {i + 1} rows...")
    rows.append(generate_row(i))

df = pd.DataFrame(rows)

# Save to parquet
output_file = 'sample_amr_surveillance_FIXED.parquet'
df.to_parquet(output_file, index=False, compression='snappy')

print(f"\n✓ Dataset created: {output_file}")
print(f"  Shape: {df.shape}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")
print(f"  File size: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
print(f"\nSample rows:")
print(df[['Pathogen', 'Date', 'ARS_WardType', 'TotalAntibioticsTested']].head(10))

# Print summary statistics
print(f"\nDataset Summary:")
print(f"  Pathogens: {df['Pathogen'].nunique()} unique")
print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"  Regions: {df['ARS_Region'].nunique()} unique")
print(f"  Wards: {df['ARS_WardType'].nunique()} unique")
print(f"  Avg antibiotics tested per episode: {df[[c for c in df.columns if '_Tested' in c]].sum(axis=1).mean():.1f}")

# Count tested and resistant
tested_cols = [c for c in df.columns if '_Tested' in c]
outcome_cols = [c for c in df.columns if '_Outcome' in c]

print(f"\n  Antibiotic columns:")
print(f"    Total: {len(all_antibiotics)}")
print(f"    Tested fields: {len(tested_cols)}")
print(f"    Outcome fields: {len(outcome_cols)}")

# Check antibiotic coverage
print(f"\nAntibiotic Coverage Analysis:")
for ab in sorted(all_antibiotics)[:5]:
    tested = df[f'{ab}_Tested'].sum()
    resistant = (df[f'{ab}_Outcome'] == 'R').sum()
    pct_tested = tested / len(df) * 100
    pct_resistant = resistant / tested * 100 if tested > 0 else 0
    print(f"  {ab}: {tested:4d} tested ({pct_tested:5.1f}%), {resistant:3d} resistant ({pct_resistant:5.1f}%)")
print(f"  ...")

print(f"\n✓ This dataset should work with pipeline!")
print(f"✓ Use with:")
print(f"    python Run.py --data ./{output_file} --out ./output")
print(f"\nLoad with:")
print(f"  import pandas as pd")
print(f"  df = pd.read_parquet('{output_file}')")
