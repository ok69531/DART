import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


# --- 표준화 유틸(선택) ---
_normalizer = rdMolStandardize.Normalizer()
_reionizer = rdMolStandardize.Reionizer()
_lf = rdMolStandardize.LargestFragmentChooser()


def standardize_mol(mol):
    """염/용매 제거 → 규칙 기반 정규화 → 재이오나이즈(중화) 순."""
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol)
    mol = _lf.choose(mol)
    mol = _normalizer.normalize(mol)
    mol = _reionizer.reionize(mol)
    return mol


# --- canonical 변환 ---
def to_canonical_smiles(smi, isomeric=True, do_standardize=False):
    """
    RDKit canonical SMILES로 변환.
    - isomeric=True 이면 입체/동위원소 보존
    - do_standardize=True 이면 표준화 후 canonicalization
    반환: 문자열 또는 None(파싱 실패)
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            return None
        if do_standardize:
            mol = standardize_mol(mol)
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
        return can
    except Exception:
        return None


# --- CSV 로드/처리 ---
def process_df(df, smiles_col="SMILES", standardize=True, isomeric=True):
    out = df.copy()
    out["rdkit_canonical"] = out[smiles_col].apply(
        lambda s: to_canonical_smiles(s, isomeric=isomeric, do_standardize=standardize)
    )
    out["is_canonical_rdkit"] = out[smiles_col] == out["rdkit_canonical"]
    return out


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tg', type = str, default = 414, help = '414 or 421')
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    path = f'dart_tg{args.tg}.xlsx'
    df = pd.read_excel(path)
    smiles_na_idx = df[df.SMILES.apply(lambda x: len(x) == 1)].index
    df.SMILES[smiles_na_idx] = df.QSAR_READY_SMILES[smiles_na_idx]
    
    df.to_excel(path, header = True, index = False)
    
