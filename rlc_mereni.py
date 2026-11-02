# Zpracování dokumentu - automatizace měření kondenzátoru na RLC metru
#!/usr/bin/env python3
"""
Zpracování log CSV z R&S LCX200 (Dynamic Impedance Measurement / Logging).

Co program dělá:
  - načte CSV (hlavičku s '#' přeskočí, najde řádek 'Timestamp,...')
  - odstraní záznam před spuštěním sweepu (opakované body na stejné frekvenci)
  - nahradí hodnoty ±inf (Cs nad rezonancí) za NaN a dopočítá D, reaktanci X, Ls
  - najde rezonanční frekvenci (SRF) třemi způsoby a porovná je:
        1) průchod fáze nulou (nejpřesnější)
        2) minimum |Z|
        3) proložení modelu C–ESL–ESR (X·ω = ω²·L − 1/C)
  - vykreslí grafy zvlášť (PNG) i dohromady (jeden obrázek)
  - uloží zpracovaná data a textový souhrn

Použití:
    python lcx_graf.py log-20261102T131839_021.csv
    python lcx_graf.py soubor.csv --lin          (lineární osa frekvence)
    python lcx_graf.py soubor.csv --out grafy

Instalace: pip install pandas numpy matplotlib
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- načtení
def nacti_log(cesta: Path):
    radky = cesta.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    meta = {}
    zacatek = None
    for i, r in enumerate(radky):
        if r.startswith("#"):
            k, _, v = r[1:].partition(",")
            meta[k.strip()] = v.strip()
        elif r.lower().startswith("timestamp"):
            zacatek = i
            break
    if zacatek is None:
        raise SystemExit("V souboru nebyl nalezen řádek s hlavičkou 'Timestamp,...'.")
    df = pd.read_csv(cesta, skiprows=zacatek, encoding="utf-8-sig")
    # sloupce podle pořadí/názvu (θ může být kódované různě)
    nazvy = {}
    for c in df.columns:
        n = c.lower()
        if n.startswith("timestamp"):
            nazvy[c] = "t"
        elif n.startswith("cs") or n.startswith("cp"):
            nazvy[c] = "Cs"
        elif n.startswith("rs") or n.startswith("rp"):
            nazvy[c] = "Rs"
        elif n.startswith("z["):
            nazvy[c] = "Z"
        elif n.startswith("frequency") or n.startswith("freq"):
            nazvy[c] = "f"
        elif n.startswith("voltage[") :
            nazvy[c] = "U"
        elif "[°]" in n or n.startswith("θ") or n.startswith("theta"):
            nazvy[c] = "th"
    df = df.rename(columns=nazvy)
    for povinny in ("f", "Z", "th"):
        if povinny not in df.columns:
            raise SystemExit(f"Chybí sloupec '{povinny}'. Nalezeno: {list(df.columns)}")
    for c in ("Cs", "Rs", "Z", "th", "f", "U"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, meta


def oriznout_sweep(df):
    """Zahodí záznam před spuštěním sweepu (frekvence pak skočí zpět dolů)."""
    skoky = np.where(df["f"].diff() < 0)[0]
    if len(skoky):
        return df.iloc[skoky[-1]:].reset_index(drop=True), skoky[-1]
    return df.reset_index(drop=True), 0


# ---------------------------------------------------------------- výpočty
def dopocitej(df):
    f = df["f"].to_numpy(float)
    w = 2 * np.pi * f
    th = np.radians(df["th"].to_numpy(float))
    z = df["Z"].to_numpy(float)
    df["X"] = z * np.sin(th)                       # reaktance
    df["R"] = z * np.cos(th)                       # sériový odpor z |Z| a θ
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Cs_calc"] = -1 / (w * df["X"])         # záporné = induktivní oblast
        df["Ls_calc"] = df["X"] / w
        rs = df["Rs"].to_numpy(float) if "Rs" in df else df["R"].to_numpy(float)
        cs = df["Cs"].to_numpy(float) if "Cs" in df else df["Cs_calc"].to_numpy(float)
        df["D"] = rs * w * cs
        df["D_z_fazi"] = np.abs(1 / np.tan(th))    # tan δ = |cot θ|
    return df


def srf_faze(f, th):
    idx = np.where((th[:-1] < 0) & (th[1:] >= 0))[0]
    if not len(idx):
        return None
    i = idx[0]
    return float(f[i] + (0 - th[i]) * (f[i + 1] - f[i]) / (th[i + 1] - th[i]))


def srf_min_z(f, z):
    i = int(np.nanargmin(z))
    if 0 < i < len(z) - 1:
        # parabola přes tři body kolem minima (v log f)
        x = np.log(f[i - 1:i + 2])
        y = np.log(z[i - 1:i + 2])
        a, b, _ = np.polyfit(x, y, 2)
        if a > 0:
            return float(math.exp(-b / (2 * a))), float(z[i])
    return None, float(z[i])


def fit_model(f, x, f_min):
    """Proložení X = ω·L − 1/(ω·C) (relativní váhy, naškálované sloupce) → L, C, f0"""
    m = (f >= f_min) & np.isfinite(x) & (np.abs(x) > 0)
    w0 = 2 * np.pi * 1e6
    u = 2 * np.pi * f[m] / w0
    A = np.vstack([u, 1 / u]).T
    wt = 1 / np.abs(x[m])
    (a, b), *_ = np.linalg.lstsq(A * wt[:, None], x[m] * wt, rcond=None)
    L, C = a / w0, (-1 / (b * w0) if b < 0 else None)
    if L <= 0 or not C or C <= 0:
        return None
    return L, C, 1 / (2 * math.pi * math.sqrt(L * C))


# ---------------------------------------------------------------- grafy
PANELY = [
    # klíč, popisek, barva, logY
    ("Cs",  "Cs [nF]",              "tab:blue",   False),
    ("Rs",  "Rs = ESR [mΩ]",        "tab:red",    False),
    ("Z",   "|Z| [Ω]",              "tab:purple", True),
    ("th",  "Fáze θ [°]",           "tab:gray",   False),
    ("D",   "D (tan δ)",            "tab:orange", False),
    ("X",   "Reaktance X [Ω]",      "tab:green",  None),   # symlog
]


def hodnoty(df, klic):
    if klic == "Cs":
        return df["Cs"].to_numpy(float) * 1e9 if "Cs" in df else df["Cs_calc"] * 1e9
    if klic == "Rs":
        return (df["Rs"] if "Rs" in df else df["R"]).to_numpy(float) * 1e3
    return df[klic].to_numpy(float)


def kresli(ax, df, klic, popisek, barva, logy, srf, lin, fmin_graf, ymax_cs=None):
    f = df["f"].to_numpy(float)
    y = hodnoty(df, klic)
    m = f >= fmin_graf
    ax.plot(f[m] / 1e6, y[m], "-", color=barva, lw=1.4)
    ax.set_xscale("linear" if lin else "log")
    if klic == "Z":
        ax.set_yscale("log")
    elif klic == "X":
        ax.set_yscale("symlog", linthresh=0.01)
        ax.axhline(0, color="k", lw=0.6)
    if klic == "Cs":
        ref = np.nanmedian(y[m & (f < 0.3 * srf)]) if srf else np.nanmedian(y[m])
        ax.set_ylim(0.8 * np.nanmin(y[m]), 3 * ref)      # u SRF Cs roste do nekonečna
        ax.text(0.02, 0.95, "osa oříznuta (Cs → ∞ u SRF)", transform=ax.transAxes,
                fontsize=8, va="top")
    if klic == "D":
        ax.set_ylim(0, 1)                                  # u SRF D diverguje
        ax.text(0.02, 0.95, "osa oříznuta (D → ∞ u SRF)", transform=ax.transAxes,
                fontsize=8, va="top")
    if srf:
        ax.axvline(srf / 1e6, color="k", ls="--", lw=1)
        ax.text(srf / 1e6, 0.97, f"  SRF {srf / 1e6:.3f} MHz", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=8)
    ax.set_ylabel(popisek)
    ax.set_xlabel("f [MHz]")
    ax.grid(True, which="both", alpha=0.3)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default=None, help="Cesta k CSV souboru")
    p.add_argument("--out", help="výstupní složka (výchozí: <název>_grafy)")
    p.add_argument("--lin", action="store_true", help="lineární osa frekvence")
    p.add_argument("--fmin-graf", type=float, default=1e3,
                   help="body pod touto frekvencí se do grafů nekreslí (výchozí 1 kHz)")
    a = p.parse_args()

    # Pokud nebyl soubor zadán přes terminál, otevře se dialog pro výběr
    if not a.csv:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        a.csv = filedialog.askopenfilename(
            title="Vyberte CSV soubor",
            filetypes=[("CSV soubory", "*.csv"), ("Všechny soubory", "*.*")]
        )
        if not a.csv:
            raise SystemExit("Nebyl vybrán žádný soubor.")

    cesta = Path(a.csv)
    df, meta = nacti_log(cesta)
    n_vse = len(df)
    df, n_pred = oriznout_sweep(df)
    df = dopocitej(df)
    print(f"Načteno {n_vse} řádků, z toho {n_pred} před sweepem (zahozeno); "
          f"sweep má {len(df)} bodů, {df['f'].min():g} – {df['f'].max():g} Hz.")

    f = df["f"].to_numpy(float)
    th = df["th"].to_numpy(float)
    z = df["Z"].to_numpy(float)
    m_ok = f >= a.fmin_graf

    # ---- rezonance
    srf1 = srf_faze(f[m_ok], th[m_ok])
    srf2, zmin = srf_min_z(f[m_ok], z[m_ok])
    fit = fit_model(f, df["X"].to_numpy(float), 5e3)
    srf = srf1 or srf2

    radky = []
    radky.append(f"Soubor: {cesta.name}")
    radky.append(f"Přístroj: {meta.get('Device', '?')} (SN {meta.get('Serial No.', '?')}), "
                 f"datum {meta.get('Date', '?')}")
    if "U" in df:
        radky.append(f"Měřicí napětí: {df['U'].median():g} V")
    radky.append("")
    if srf1:
        radky.append(f"SRF (průchod fáze nulou):   {srf1 / 1e6:.4f} MHz  ({srf1:,.0f} Hz)")
    if srf2:
        radky.append(f"SRF (minimum |Z|):          {srf2 / 1e6:.4f} MHz  ({srf2:,.0f} Hz)")
    if fit:
        L, C, f0 = fit
        radky.append(f"SRF (model C–ESL, fit):     {f0 / 1e6:.4f} MHz  "
                     f"(C = {C * 1e9:.1f} nF, ESL = {L * 1e9:.2f} nH)")
    if srf:
        i = int(np.nanargmin(np.abs(f - srf)))
        esr = df["Rs"].iloc[i] if "Rs" in df else df["R"].iloc[i]
        radky.append(f"ESR na rezonanci:           {esr * 1e3:.2f} mΩ")
        radky.append(f"|Z| na rezonanci (min.):    {zmin * 1e3:.2f} mΩ")
        # kapacita z oblasti hluboko pod SRF, opravená o vliv ESL
        sel = (f > 50e3) & (f < 0.2 * srf) & np.isfinite(df["Cs_calc"])
        if sel.any():
            c0 = np.nanmedian(df["Cs_calc"][sel] * (1 - (f[sel] / srf) ** 2))
            radky.append(f"Kapacita (50 kHz – 0,2·SRF):  {c0 * 1e9:.1f} nF")
            esl = 1 / ((2 * math.pi * srf) ** 2 * c0)
            radky.append(f"ESL z SRF a C:              {esl * 1e9:.2f} nH")
        # použitelné kapacitní pásmo (Cs do ±5 % oproti referenci)
        ref = df.iloc[(df["f"] - 100e3).abs().argmin()]
        cs_ref = ref["Cs_calc"]
        band = (np.abs(df["Cs_calc"] / cs_ref - 1) <= 0.05) & (f >= a.fmin_graf)
        if band.any():
            radky.append(f"Kapacita v toleranci ±5 % (vůči 100 kHz): "
                         f"{f[band].min() / 1e3:.0f} kHz – {f[band].max() / 1e6:.3f} MHz")
    souhrn = "\n".join(radky)
    print("\n" + souhrn)

    # ---- výstupy
    out = Path(a.out) if a.out else cesta.with_name(cesta.stem + "_grafy")
    out.mkdir(parents=True, exist_ok=True)

    # zvlášť
    for klic, popisek, barva, logy in PANELY:
        fig, ax = plt.subplots(figsize=(8, 5))
        kresli(ax, df, klic, popisek, barva, logy, srf, a.lin, a.fmin_graf)
        ax.set_title(f"{cesta.stem}: {popisek}")
        fig.tight_layout()
        fig.savefig(out / f"{klic}.png", dpi=150)
        plt.close(fig)

    # dohromady
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (klic, popisek, barva, logy) in zip(axs.flat, PANELY):
        kresli(ax, df, klic, popisek, barva, logy, srf, a.lin, a.fmin_graf)
        ax.set_title(popisek)
    fig.suptitle(f"{cesta.stem}"
                 + (f"   –   SRF ≈ {srf / 1e6:.3f} MHz" if srf else ""), fontsize=13)
    fig.tight_layout()
    spolecny = out / "spolecny_graf.png"
    fig.savefig(spolecny, dpi=150)
    plt.close(fig)

    # zpracovaná data + souhrn
    sloupce = [c for c in ("f", "Cs", "Cs_calc", "Rs", "Z", "th", "X", "Ls_calc", "D") if c in df]
    df[sloupce].to_csv(out / "zpracovano.csv", index=False, sep=";", float_format="%.6g")
    (out / "souhrn.txt").write_text(souhrn + "\n", encoding="utf-8")
    print(f"\nUloženo do složky: {out}")


if __name__ == "__main__":
    main()