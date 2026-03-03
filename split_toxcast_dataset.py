import pandas as pd


file_path = 'dataset/raw/raw_data.xlsx'
raw_file = pd.ExcelFile(file_path)

sheet_names = raw_file.sheet_names

toxcast = raw_file.parse('ToxCast raw data', skiprows = [1, 2])

dart_low_perform = raw_file.parse('DART tired model (F1<0.5)')
dart_low_perform_assay = dart_low_perform['Assay Name'].unique()

dart_lp_tier1 = dart_low_perform[dart_low_perform.Tier == 1]
dart_lp_tier2 = dart_low_perform[dart_low_perform.Tier == 2]
dart_lp_tier3 = dart_low_perform[dart_low_perform.Tier == 3]
assert dart_low_perform.shape[0] == dart_lp_tier1.shape[0] + dart_lp_tier2.shape[0] + dart_lp_tier3.shape[0]

dart_lp_tier1_assays = list(set(dart_lp_tier1['Assay Name']))
dart_lp_tier2_assays = list(set(dart_lp_tier2['Assay Name']))
dart_lp_tier3_assays = list(set(dart_lp_tier3['AEID']))               # tg dataset (tg 421, 416)
tier_dup = [x for x in dart_lp_tier2_assays if x in dart_lp_tier1_assays]
assert len(dart_low_perform_assay) == len(dart_lp_tier1_assays) + len(dart_lp_tier2_assays) + len(dart_lp_tier3_assays) - len(tier_dup)



toxcast_tier1_assay_idx = [i for i in range(2, toxcast.shape[1]) if toxcast.iloc[:, i].name in dart_lp_tier1_assays]
toxcast_tier2_assay_idx = [i for i in range(2, toxcast.shape[1]) if toxcast.iloc[:, i].name in dart_lp_tier2_assays]

assert len(dart_lp_tier1_assays) == len(toxcast_tier1_assay_idx)
assert len(dart_lp_tier2_assays) == len(toxcast_tier2_assay_idx)

toxcast_tier1_lp = toxcast.iloc[:, [1] + toxcast_tier1_assay_idx]

tmp = toxcast.iloc[:, toxcast_tier2_assay_idx]

result = [x for x in dart_lp_tier2_assays if x not in tmp.columns]

len(toxcast_tier2_assay_idx) + len(result) == len(dart_lp_tier2_assays)