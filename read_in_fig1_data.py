import re, pandas as pd, numpy as np
from pathlib import Path

output_path = Path("data/")      
input_path = Path("data/Data_Figure _1-2_G-Flamp1_Sensor_Shock_Bouton_Axon")  

# --- Regex: "g" before 1-5 is optional (matches B-g5_1 and B-5_1; with/without fly prefix) ---
ROI_RE_FLIES = re.compile(r'^(?P<struct>[AB])-(?:g)?(?P<g>[1-5])_(?P<roi>\d+)(?:\.(?P<sfx>\d+))?$')
ROI_RE_WIDE  = re.compile(r'^fly(?P<fly>\d+)_(?P<struct>[AB])-(?:g)?(?P<g>[1-5])_(?P<roi>\d+)(?:\.(?P<sfx>\d+))?$')

NON_DATA_COLS = {"Mittelwert","Standardabweichung","SE des Mittelwerts","Mittelwert.1"}

def is_roi_col(c: str, dataset: str) -> bool:
    if not isinstance(c, str): return False
    if c == "time": return False
    if c.startswith("Unnamed"): return False
    if c.startswith("Integrated"): return False
    if c in NON_DATA_COLS: return False
    if dataset in {"Bouton","Axon"}:
        return ROI_RE_WIDE.match(c) is not None
    if dataset == "Flies":
        return ROI_RE_FLIES.match(c) is not None
    return False

def genotype_from_name(name: str) -> str:
    return "dnc-KD" if "dncRNAi" in name else "dnc-wt"

def tidy_sheet(df, sheet, wb_name, dataset):
    if "time" not in df.columns:
        return pd.DataFrame()

    # time → numeric; drop unit/header rows
    t = pd.to_numeric(df["time"], errors="coerce")
    df = df.loc[t.notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["time_s"] = t[t.notna()].values

    # pick only ROI columns
    roi_cols = [c for c in df.columns if is_roi_col(c, dataset)]
    if not roi_cols:
        return pd.DataFrame()

    long = df[["time_s"] + roi_cols].melt(
        id_vars="time_s", var_name="roi_label", value_name="value"
    )

    # --- parse ROI metadata (DON'T filter out None; keep row alignment) ---
    meta_rows = []
    for lab in long["roi_label"]:
        m = (ROI_RE_WIDE.match(lab) if dataset in {"Bouton","Axon"} else ROI_RE_FLIES.match(lab))
        if not m:
            meta_rows.append(None)
            continue
        d = m.groupdict()
        meta_rows.append({
            "structure": {"A": "Axon", "B": "Bouton"}[d["struct"]],
            "compartment": f"g{d['g']}",
            "roi_index": int(d["roi"]),
            "fly_id": int(d["fly"]) if d.get("fly") else None,
            "suffix": d.get("sfx"),
        })

    meta = pd.DataFrame(meta_rows, index=long.index)

    # --- fill fly_id for Flies from sheet name BEFORE computing "good" ---
    if dataset == "Flies":
        m_sheet = re.match(r"fly(\d+)", sheet, re.I)
        fly_from_sheet = int(m_sheet.group(1)) if m_sheet else None
        meta["fly_id"] = meta["fly_id"].fillna(fly_from_sheet)

    # --- require only the fields that must exist ---
    must_have = ["structure", "compartment", "roi_index"]
    if dataset in {"Bouton", "Axon"}:
        must_have.append("fly_id")   # for Bouton/Axon the fly id is in the label
    good = meta[must_have].notna().all(axis=1)

    if not good.any():
        return pd.DataFrame()

    out = pd.concat([long.loc[good].reset_index(drop=True),
                     meta.loc[good].reset_index(drop=True)], axis=1)

    # annotate ids
    out["dataset"]  = dataset
    out["workbook"] = wb_name
    out["sheet"]    = sheet
    out["genotype"] = genotype_from_name(wb_name)

    # measure & fly_id source
    if dataset == "Flies":
        out["measure"] = np.where(out["suffix"].notna(), "dff", "raw")
    else:
        out["measure"] = "dff"

    return out[[
        "genotype","dataset","workbook","sheet","fly_id",
        "structure","compartment","roi_index","roi_label",
        "time_s","measure","value"
    ]]

def run_all(xlsx_paths, csv_filename=None, output_dir=None):
    frames = []
    total_est_rows = 0

    for wb in xlsx_paths:
        dataset = "Flies" if "Flies" in wb.name else ("Bouton" if "Bouton" in wb.name else "Axon")
        xls = pd.ExcelFile(wb)
        for sheet in xls.sheet_names:
            # --- USE keep to actually skip ---
            if dataset == "Flies":
                keep = re.match(r"fly\d+", sheet, re.I) is not None
            else:  # Bouton / Axon
                keep = sheet in {"g1","g2","g3","g4","g5"}
            if not keep:
                continue

            print(f"Processing {wb.name} - {sheet} ({dataset})")
            df = pd.read_excel(wb, sheet_name=sheet)

            # diagnostics: how many numeric timepoints, how many ROI columns
            t_num = pd.to_numeric(df.get("time", pd.Series(dtype=float)), errors="coerce")
            n_time = int(t_num.notna().sum()) if "time" in df else 0
            roi_cols = [c for c in df.columns if is_roi_col(c, dataset)]
            n_roi = len(roi_cols)
            print(f"  timepoints={n_time}, roi_cols={n_roi}, est_long_rows≈{n_time*n_roi}")
            if n_roi > 0:
                print(f"  sample roi cols: {roi_cols[:6]}")

            long = tidy_sheet(df, sheet, wb.name, dataset)
            print(f"  extracted rows: {len(long)}")
            total_est_rows += n_time * n_roi

            if not long.empty:
                frames.append(long)

    if not frames:
        raise RuntimeError("No data extracted. Check regex/filters and sheet names.")

    long = pd.concat(frames, ignore_index=True)

    wide = long.pivot_table(
        index=["genotype","dataset","workbook","sheet","fly_id","structure",
               "compartment","roi_index","roi_label","time_s"],
        columns="measure", values="value", aggfunc="first"
    ).reset_index()

    # write
    if csv_filename:
        Path(csv_filename).parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(csv_filename, index=False)
    else:
        outdir = Path(output_dir) if output_dir else Path(".")
        outdir.mkdir(parents=True, exist_ok=True)
        wide.to_csv(outdir/"camp_long.csv", index=False)

    print(f"\nTOTAL estimated long rows (time×ROI sum): {total_est_rows}")
    print(f"TOTAL extracted rows (after parsing): {len(long)}")
    print(f"Rows in wide table: {len(wide)}")

    return wide


# Example:
files = [input_path/"Flies.xlsx", input_path/"Flies+dncRNAi.xlsx", input_path/"Bouton.xlsx", input_path/"Bouton+dncRNAi.xlsx", input_path/"Axon.xlsx", input_path/"Axon+dncRNAi.xlsx"] 
wide = run_all(files,output_dir=Path("data/"))
