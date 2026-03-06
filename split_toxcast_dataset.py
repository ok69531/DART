import os
import pandas as pd

if __name__ == '__main__':
    file_path = 'dataset/raw/raw_data.xlsx'
    raw_file = pd.ExcelFile(file_path)

    # sheet_names = raw_file.sheet_names

    toxcast = raw_file.parse('ToxCast raw data', skiprows = [1, 2])

    # drop the assays where F1 score is nan and the number of chemicals is less than 100
    dart_low_perform = raw_file.parse('DART tired model (F1<0.5)')
    dart_low_perform = dart_low_perform.dropna(subset = 'F1')
    drop_idx = dart_low_perform['No. of chemicals'] < 100
    dart_low_perform = dart_low_perform[~drop_idx].reset_index(drop = True)

    dart_low_perform_assay = dart_low_perform['Assay Name'].unique()

    # split dataset per tier
    dart_lp_tier1 = dart_low_perform[dart_low_perform.Tier == 1]
    dart_lp_tier2 = dart_low_perform[dart_low_perform.Tier == 2]
    dart_lp_tier3 = dart_low_perform[dart_low_perform.Tier == 3]
    assert dart_low_perform.shape[0] == dart_lp_tier1.shape[0] + dart_lp_tier2.shape[0] + dart_lp_tier3.shape[0]

    # assay name in each tier
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
    toxcast_tier2_lp = toxcast.iloc[:, [1] + toxcast_tier2_assay_idx]

    # #%%
    # ''' 
    #     tier2
    #     f1 score가 0.5보다 작은 assay를 모아놓은 sheet에는 존재하지만 
    #     toxcast raw data 시트에는 존재하지 않는 어세들이 있음.
    #     -> 이거 처리해야함
    # '''
    # tmp = toxcast.iloc[:, toxcast_tier2_assay_idx]
    # result = [x for x in dart_lp_tier2_assays if x not in tmp.columns]
    # assert len(toxcast_tier2_assay_idx) + len(result) == len(dart_lp_tier2_assays)
    # #%%

    tc_tier1_lp_clean = toxcast_tier1_lp.dropna(subset = ['SMILES']).dropna(subset = dart_lp_tier1_assays, how = 'all')
    tc_tier2_lp_clean = toxcast_tier2_lp.dropna(subset = ['SMILES']).dropna(subset = toxcast_tier2_lp.columns.drop('SMILES'), how = 'all')


    tier_num = 1
    save_path = f'dataset/raw/ToxCast_Tier{tier_num}_LowPerform'
    file_name = f'ToxCast_Tier{tier_num}_LowPerform.csv'

    if os.path.exists(save_path):pass
    else: os.makedirs(save_path)
    tc_tier1_lp_clean.to_csv(os.path.join(save_path, file_name), header = True, index = False)
    with open(os.path.join(save_path, 'assay_names.txt'), 'w') as f:
        f.writelines('\n'.join(dart_lp_tier1_assays))
    
    tier_num = 2
    save_path = f'dataset/raw/ToxCast_Tier{tier_num}_LowPerform'
    file_name = f'ToxCast_Tier{tier_num}_LowPerform.csv'

    if os.path.exists(save_path):pass
    else: os.makedirs(save_path)
    tc_tier2_lp_clean.to_csv(os.path.join(save_path, file_name), header = True, index = False)
    with open(os.path.join(save_path, 'assay_names.txt'), 'w') as f:
        f.writelines('\n'.join(dart_lp_tier2_assays))
