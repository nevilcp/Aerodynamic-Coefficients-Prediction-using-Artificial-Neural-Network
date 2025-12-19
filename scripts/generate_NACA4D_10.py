#!/usr/bin/env python3

import os
import time
import csv
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from collections import defaultdict

JAVA_BIN = "/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java"
JAVAFOIL_JAR = "/home/nevilcp/MH-AeroTools/JavaFoil/javafoil.jar"
MHCLASSES_JAR = "/home/nevilcp/MH-AeroTools/JavaFoil/mhclasses.jar"

BASE_DIR = Path.home() / "ML_Aero"
RESULTS_BASE = BASE_DIR / "results" / "NACA4D_10"
MACRO_DIR = RESULTS_BASE / "macros"
XML_DIR   = RESULTS_BASE / "xml"
MASTER_CSV = RESULTS_BASE / "NACA4D_10.csv"

CAMBER_VALUES       = list(range(0, 10))
CAMBER_LOC_VALUES   = list(range(5, 76, 10))
THICKNESS_VALUES    = list(range(5, 36, 5))
MACH_VALUES         = [0.1, 0.2, 0.3]

RE_START, RE_END, RE_STEP = 1e5, 5e5, 1e5
AOA_START, AOA_END, AOA_STEP = -10.0, 10.0, 1.0

N_INNER = 10
NUM_POINTS = N_INNER + 2
CSV_BATCH_SIZE = 500
POST_WAIT = 0.05
TIMEOUT_JAVAFOIL = 120
MAX_WORKERS = min(4, os.cpu_count() or 2)

MACRO_TEMPLATE = r"""
Options.Country(0);
Options.GroundEffect(0);
Options.AspectRatio(0);
Options.StallModel(2);
Options.TransitionModel(8);
Modify.SetPivot(0.25, 0);
Modify.PointCount(400);

Geometry.CreateAirfoil(
    0, 101, {thickness}, 30, {camber}, {camber_loc},
    0.0, 0.0, 0.0, 0.0, 0
);

Geometry.Save("{geom_xml}");

Options.MachNumber({mach});

Polar.Analyze(
    {re_start}, {re_end}, {re_step},
    {aoa_start}, {aoa_end}, {aoa_step},
    1.0, 1.0, 0.0, false
);

Polar.Save("{polar_xml}");
JavaFoil.Exit();
"""

# Run JavaFoil macro and return exit code
def run_javafoil_macro(macro_path: Path):
    cmd = [
        JAVA_BIN, "-cp", str(MHCLASSES_JAR),
        "-jar", str(JAVAFOIL_JAR),
        f"Script={str(macro_path)}"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding="utf-8", timeout=TIMEOUT_JAVAFOIL)
        return proc.returncode
    except subprocess.TimeoutExpired:
        return -1

# Parse 101-point geometry XML and return upper/lower lists
def parse_geometry_xml(xml_path: Path):
    try:
        tree = ET.parse(str(xml_path))
        pts = []
        for p in tree.findall(".//coordinates/point"):
            x_node = p.find("x")
            y_node = p.find("y")
            if x_node is None or y_node is None:
                continue
            x = float(x_node.text)
            y = float(y_node.text)
            pts.append((x, y))
        if len(pts) < 5:
            return None, None
        xs = [pt[0] for pt in pts]
        le_index = int(np.argmin(xs))
        upper = sorted(pts[:le_index + 1], key=lambda z: z[0])
        lower = sorted(pts[le_index:], key=lambda z: z[0])
        return upper, lower
    except Exception:
        return None, None

# Parse polar XML and return a list of (alpha, Re, CL, CD, Cm) rows
def parse_polar_xml(xml_path: Path):
    def _strip_namespace(tree):
        root = tree.getroot()
        for elem in root.iter():
            if isinstance(elem.tag, str) and "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]
        return root

    try:
        tree = ET.parse(str(xml_path))
        root = _strip_namespace(tree)
        polars_out = []
        for polar in root.findall(".//polar"):
            rnode = polar.find("reynoldsnumber")
            if rnode is None:
                continue
            try:
                Re = int(float(rnode.text))
            except Exception:
                continue
            vars_ = [(v.text or "").strip().lower()
                     for v in polar.findall(".//variables/variable")]
            try:
                ia = vars_.index("alpha")
                icl = vars_.index("cl")
                icd = vars_.index("cd")
                icm = vars_.index("cm")
            except ValueError:
                continue
            for dp in polar.findall(".//datapoint"):
                vals = dp.findall("value")
                if len(vals) <= max(ia, icl, icd, icm):
                    continue
                try:
                    a = float(vals[ia].text)
                    cl = float(vals[icl].text)
                    cd = float(vals[icd].text)
                    cm = float(vals[icm].text)
                    polars_out.append((a, Re, cl, cd, cm))
                except Exception:
                    continue
        return polars_out if polars_out else None
    except Exception:
        return None

# Resample upper and lower surfaces using cosine spacing
def resample_geometry(upper_pts, lower_pts, n_points):
    try:
        up = np.array(sorted(upper_pts, key=lambda z: z[0]))
        lo = np.array(sorted(lower_pts, key=lambda z: z[0]))
        minx = min(up[:, 0].min(), lo[:, 0].min())
        maxx = max(up[:, 0].max(), lo[:, 0].max())
        span = maxx - minx
        if span <= 1e-12:
            return None
        xu = (up[:, 0] - minx) / span
        xl = (lo[:, 0] - minx) / span
        xt = np.array([0.5 * (1 - np.cos(np.pi * i / (n_points - 1)))
                       for i in range(n_points)])
        yu = np.interp(xt, xu, up[:, 1])
        yl = np.interp(xt, xl, lo[:, 1])
        return xt, yu, yl
    except Exception:
        return None

# Check that trailing edge is closed within tolerances
def check_true_te(upper, lower):
    try:
        yU = upper[-1][1]
        yL = lower[-1][1]
        tol_y   = 0.01
        tol_gap = 0.015
        return (
            abs(yU) <= tol_y and
            abs(yL) <= tol_y and
            abs(yU - yL) <= tol_gap
        )
    except Exception:
        return False

# Generate, parse and clean a single case; returns CSV rows or None
def process_case(idx: int, params: tuple):
    camber, camber_loc, thickness, mach = params
    prefix = f"{idx:04d}"

    geom_xml  = XML_DIR / f"{prefix}_geom.xml"
    polar_xml = XML_DIR / f"{prefix}_polar.xml"
    macro_js  = MACRO_DIR / f"{prefix}_macro.js"

    for f in (geom_xml, polar_xml):
        try:
            if f.exists():
                f.unlink()
        except:
            pass

    macro = MACRO_TEMPLATE.format(
        thickness=thickness, camber=camber, camber_loc=camber_loc,
        mach=mach,
        re_start=int(RE_START), re_end=int(RE_END), re_step=int(RE_STEP),
        aoa_start=AOA_START, aoa_end=AOA_END, aoa_step=AOA_STEP,
        geom_xml=str(geom_xml).replace("\\","/"),
        polar_xml=str(polar_xml).replace("\\","/")
    )
    macro_js.write_text(macro, encoding="utf-8")

    rc = run_javafoil_macro(macro_js)
    time.sleep(POST_WAIT)

    upper, lower = parse_geometry_xml(geom_xml)
    if upper is None or lower is None:
        return None

    if not check_true_te(upper, lower):
        return None

    polars = parse_polar_xml(polar_xml)
    if not polars:
        return None

    res = resample_geometry(upper, lower, NUM_POINTS)
    if res is None:
        return None
    _, yu, yl = res
    yU = [f"{float(round(v,8)):.8f}" for v in yu[1:-1]]
    yL = [f"{float(round(v,8)):.8f}" for v in yl[1:-1]]

    rows = []
    for a, Re, cl, cd, cm in polars:
        if not all(map(np.isfinite, (a, cl, cd, cm))):
            continue
        rows.append([thickness, camber, camber_loc] + yU + yL +
                    [a, mach, Re, cl, cd, cm])

    return rows if rows else None

# Main: enumerates parameter sets, runs cases in parallel, writes CSV
def main():
    for d in (MASTER_CSV.parent, MACRO_DIR, XML_DIR):
        d.mkdir(parents=True, exist_ok=True)

    geom_cols = [f"yU{k}" for k in range(1, N_INNER+1)] + \
                [f"yL{k}" for k in range(1, N_INNER+1)]
    header = ["t","m","p"] + geom_cols + ["alpha","M","Re","CL","CD","Cm"]

    with MASTER_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    params = list(product(CAMBER_VALUES, CAMBER_LOC_VALUES,
                          THICKNESS_VALUES, MACH_VALUES))

    batch = []
    accepted_cases = 0
    rejected_cases = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_case, idx, p): (idx, p)
                   for idx, p in enumerate(params, 1)}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx, p = futures[fut]
            try:
                out = fut.result()
            except Exception:
                rejected_cases += 1
                continue

            if out:
                accepted_cases += 1
                batch.extend(out)
                if len(batch) >= CSV_BATCH_SIZE:
                    with MASTER_CSV.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerows(batch)
                    batch.clear()
            else:
                rejected_cases += 1

    if batch:
        with MASTER_CSV.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(batch)

    print(f"\nDone. Accepted cases: {accepted_cases}, Rejected cases: {rejected_cases}")
    print("Cleaned dataset saved to:", MASTER_CSV)

if __name__ == "__main__":
    main()

