#!/usr/bin/env python3
"""disco_mappa.py — mappa pezzo × tempo dai log di `disco leggi`.

Legge i log di `DISCO_ISTO=1 DISCO_BINS=32 disco leggi <file> s16 ...`
presi a orari diversi e stampa:

  1. la tabella pezzo per tempo: V = veloce, L = lento, ? = incerto;
  2. per ogni orario quanti pezzi (e quanti GB) sono ancora veloci;
  3. la velocità dello spostamento in GB all'ora (retta ai minimi quadrati
     e velocità fra un log e il successivo);
  4. l'ordine in cui i pezzi sono diventati lenti.

Solo libreria standard. Python >= 3.9.

Uso tipico:
    python3 disco_mappa.py --gb 22 log/*.txt
    python3 disco_mappa.py --gb 22 --soglia 89 --ora-da nome log/*.txt
    python3 disco_mappa.py --gb 22 --csv mappa.csv --md log/*.txt

Formato dei log accettato (tollerante):
  * una riga per pezzo, prima dell'istogramma, con l'indice del pezzo e la
    mediana in microsecondi. Esempi che vanno bene:
        pezzo 12: mediana 77.3 us
        pezzo 12  0.63-1.26 GB  med 77.3 us  min 61 max 410
        [12] 77.3
        12 77.3
  * l'istogramma è riconosciuto dalla parola "istogramma"/"histogram" e
    tutto quello che segue viene ignorato;
  * l'orario del log viene (in quest'ordine, `--ora-da auto`): dal nome
    del file (es. leggi-2026-10-23T19-55.txt oppure leggi-1955.txt), da
    una riga nel log con data e ora ISO (2026-10-23 19:55:12) o con
    "ora: 19:55", altrimenti dall'ora di modifica del file;
  * la dimensione del file misurato: `--gb 22`, oppure una riga del log tipo
    "dim 22.0 GB" / "size: 23622320128 byte".

Se il tuo `disco` stampa in un altro modo, passa `--regex` con due gruppi
(indice, mediana), per esempio:
    --regex '^pezzo\\s+(\\d+).*?(\\d+(?:\\.\\d+)?)\\s*us'
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# lettura di un log
# ----------------------------------------------------------------------------

UNITA = r"(?:us|µs|μs|usec|microsec\w*)"
NUM = r"(\d+(?:[.,]\d+)?)"

# riga "con parola chiave": pezzo/piece/chunk/bin/blocco + indice, mediana in us
RX_PAROLA = re.compile(
    r"^\s*(?:pezzo|piece|chunk|bin|blocco|parte)\s*#?\s*(\d+)\b(.*)$", re.I)
# riga "[12] ..." oppure "12 77.3" (due colonne nude)
RX_QUADRE = re.compile(r"^\s*\[\s*(\d+)\s*\](.*)$")
RX_NUDA = re.compile(r"^\s*(\d+)\s*[:;,\t ]\s*" + NUM + r"\s*" + UNITA + r"?\s*$")
RX_FINE = re.compile(r"istogr|histog", re.I)

# orario dentro al log
RX_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T _]+(\d{2}):(\d{2})(?::(\d{2}))?")
RX_ORA = re.compile(r"\b(?:ora|time|orario)\s*[:=]?\s*(\d{1,2}):(\d{2})(?::(\d{2}))?", re.I)
# orario nel nome del file
RX_NOME_ISO = re.compile(r"(\d{4})-?(\d{2})-?(\d{2})[T_\-. ]?(\d{2})[-:h.]?(\d{2})(?:[-:.]?(\d{2}))?")
RX_NOME_HHMM = re.compile(r"(?<!\d)(\d{2})[-:h.]?(\d{2})(?:[-:.]?(\d{2}))?(?!\d)")
# dimensione del file misurato
RX_DIM = re.compile(
    r"(?:dim\w*|size|byte|bytes|grandezza|lunghezza)\D{0,12}?" + NUM +
    r"\s*(GiB|GB|MiB|MB|KiB|KB|B|byte|bytes)?\b", re.I)
RX_DIM2 = re.compile(NUM + r"\s*(GiB|GB|MiB|MB)\b")


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _mediana_da_resto(resto: str) -> Optional[float]:
    """Dal resto della riga estrae la mediana in us.

    Preferisce il numero seguito da 'us'; con più numeri+us prende quello dopo
    la parola mediana/median/med se c'è, altrimenti il primo. Senza unità prende
    l'ultimo numero della riga.
    """
    con_unita = list(re.finditer(NUM + r"\s*" + UNITA + r"\b", resto, re.I))
    if con_unita:
        m = re.search(r"(?:mediana|median|med)\D{0,6}" + NUM + r"\s*" + UNITA, resto, re.I)
        if m:
            return _num(m.group(1))
        return _num(con_unita[0].group(1))
    m = re.search(r"(?:mediana|median|med)\D{0,6}" + NUM, resto, re.I)
    if m:
        return _num(m.group(1))
    # senza unità e senza parola chiave: il primo numero che non sia una
    # dimensione (GB/MB/KB), una percentuale o l'inizio di un intervallo "a-b"
    candidati = []
    for m in re.finditer(NUM + r"(?=(\s*(?:GiB|GB|MiB|MB|KiB|KB|%|-|–)))?", resto, re.I):
        if m.group(2) is None:
            candidati.append(_num(m.group(1)))
    if candidati:
        return candidati[0]
    numeri = re.findall(NUM, resto)
    if numeri:
        return _num(numeri[-1])
    return None


class Log:
    def __init__(self, path: str):
        self.path = path
        self.ora: Optional[dt.datetime] = None
        self.ora_da: str = ""
        self.mediane: Dict[int, float] = {}
        self.gb: Optional[float] = None
        self.file_misurato: Optional[str] = None
        self.avvisi: List[str] = []


def leggi_log(path: str, regex: Optional[re.Pattern]) -> Log:
    L = Log(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        righe = f.read().splitlines()
    for riga in righe:
        if RX_FINE.search(riga):
            break
        if L.ora is None:
            m = RX_ISO.search(riga)
            if m:
                y, mo, d, h, mi, s = m.groups()
                L.ora = dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0))
                L.ora_da = "riga"
        if L.gb is None:
            m = RX_DIM.search(riga)
            if m and m.group(2):
                L.gb = _a_gb(_num(m.group(1)), m.group(2))
            elif m and m.group(2) is None and _num(m.group(1)) > 1e8:
                L.gb = _a_gb(_num(m.group(1)), "B")
        if L.file_misurato is None:
            m = re.search(r"(/[^\s'\"]+)", riga)
            if m and re.search(r"file|leggi|path", riga, re.I):
                L.file_misurato = m.group(1)
        idx, med = None, None
        if regex is not None:
            m = regex.search(riga)
            if m:
                idx, med = int(m.group(1)), _num(m.group(2))
        else:
            m = RX_PAROLA.match(riga) or RX_QUADRE.match(riga)
            if m:
                idx = int(m.group(1))
                med = _mediana_da_resto(m.group(2))
            else:
                m = RX_NUDA.match(riga)
                if m:
                    idx, med = int(m.group(1)), _num(m.group(2))
        if idx is not None and med is not None:
            if idx in L.mediane:
                L.avvisi.append(f"pezzo {idx} ripetuto, tengo il primo")
            else:
                L.mediane[idx] = med
    # orario senza data: prendo la data dall'ora di modifica
    if L.ora is None:
        for riga in righe:
            m = RX_ORA.search(riga)
            if m:
                base = dt.datetime.fromtimestamp(os.stat(path).st_mtime)
                L.ora = base.replace(hour=int(m.group(1)), minute=int(m.group(2)),
                                     second=int(m.group(3) or 0), microsecond=0)
                L.ora_da = "riga"
                break
    return L


def _a_gb(v: float, unita: str) -> float:
    u = (unita or "B").lower()
    if u == "gib":
        return v * (1 << 30) / 1e9
    if u == "gb":
        return v
    if u == "mib":
        return v * (1 << 20) / 1e9
    if u == "mb":
        return v / 1e3
    if u == "kib":
        return v * 1024 / 1e9
    if u == "kb":
        return v / 1e6
    return v / 1e9


def ora_dal_nome(path: str, mtime: float) -> Optional[Tuple[dt.datetime, str]]:
    nome = os.path.splitext(os.path.basename(path))[0]
    m = RX_NOME_ISO.search(nome)
    if m:
        y, mo, d, h, mi, s = m.groups()
        try:
            return dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0)), "nome"
        except ValueError:
            pass
    m = None
    for m in RX_NOME_HHMM.finditer(nome):
        pass  # l'ultimo HHMM del nome
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if h < 24 and mi < 60 and s < 60:
            base = dt.datetime.fromtimestamp(mtime)
            return base.replace(hour=h, minute=mi, second=s, microsecond=0), "nome"
    return None


def assegna_ore(logs: List[Log], modo: str) -> None:
    for L in logs:
        mtime = os.stat(L.path).st_mtime
        scelta = None
        if modo in ("auto", "nome"):
            scelta = ora_dal_nome(L.path, mtime)
        if scelta is None and modo in ("auto", "riga") and L.ora is not None:
            scelta = (L.ora, "riga")
        if scelta is None and modo in ("auto", "mtime"):
            scelta = (dt.datetime.fromtimestamp(mtime), "mtime")
        if scelta is None:
            scelta = (dt.datetime.fromtimestamp(mtime), "mtime")
            L.avvisi.append(f"orario non trovato con --ora-da {modo}, uso mtime")
        L.ora, L.ora_da = scelta


# ----------------------------------------------------------------------------
# soglia veloce/lento
# ----------------------------------------------------------------------------

def soglia_automatica(valori: List[float]) -> Tuple[float, float, float]:
    """Due gruppi (k-means a 1 dimensione). Torna (soglia, centro_veloce, centro_lento)."""
    v = sorted(valori)
    if len(v) < 2:
        return (v[0] if v else 0.0), (v[0] if v else 0.0), (v[0] if v else 0.0)
    a, b = v[0], v[-1]
    for _ in range(100):
        g1 = [x for x in v if abs(x - a) <= abs(x - b)]
        g2 = [x for x in v if abs(x - a) > abs(x - b)]
        if not g1 or not g2:
            break
        na, nb = sum(g1) / len(g1), sum(g2) / len(g2)
        if na == a and nb == b:
            break
        a, b = na, nb
    return (a + b) / 2.0, a, b


def classifica(x: float, soglia: float, margine: float) -> str:
    if abs(x - soglia) <= soglia * margine / 100.0:
        return "?"
    return "V" if x < soglia else "L"


# ----------------------------------------------------------------------------
# retta ai minimi quadrati
# ----------------------------------------------------------------------------

def retta(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """y = a + b x. Torna (a, b). Con meno di 2 punti distinti b = 0."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0), 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


# ----------------------------------------------------------------------------
# programma
# ----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Mappa pezzo × tempo dai log di `disco leggi`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Uso tipico:", 1)[1])
    ap.add_argument("log", nargs="+", help="file di log (uno per orario)")
    ap.add_argument("--gb", type=float, default=None,
                    help="dimensione del file misurato in GB (altrimenti letta dal log)")
    ap.add_argument("--pezzi", type=int, default=32, help="pezzi attesi per log (32)")
    ap.add_argument("--soglia", type=float, default=None,
                    help="soglia in us: sotto = veloce, sopra = lento (automatica se assente)")
    ap.add_argument("--margine", type=float, default=3.0,
                    help="percento attorno alla soglia segnato '?' (3)")
    ap.add_argument("--ora-da", choices=["auto", "nome", "riga", "mtime"], default="auto",
                    help="da dove prendere l'orario di ogni log")
    ap.add_argument("--regex", default=None,
                    help="espressione con due gruppi (indice, mediana) per righe non standard")
    ap.add_argument("--csv", default=None, help="scrive la matrice delle mediane in CSV")
    ap.add_argument("--md", action="store_true", help="stampa la tabella anche in markdown")
    ap.add_argument("--senza-colori", action="store_true", help="niente colori ANSI")
    args = ap.parse_args(argv)

    regex = re.compile(args.regex) if args.regex else None
    logs: List[Log] = []
    for p in args.log:
        try:
            L = leggi_log(p, regex)
        except OSError as e:
            print(f"[salto] {p}: {e}", file=sys.stderr)
            continue
        if not L.mediane:
            print(f"[salto] {p}: nessuna riga 'pezzo N ... us' riconosciuta", file=sys.stderr)
            continue
        logs.append(L)
    if not logs:
        print("nessun log leggibile", file=sys.stderr)
        return 2

    assegna_ore(logs, args.ora_da)
    logs.sort(key=lambda L: L.ora)

    # indici dei pezzi: unione di tutti i log (normalmente 0..31)
    indici = sorted(set(i for L in logs for i in L.mediane))
    if indici and indici[0] == 1 and len(indici) == args.pezzi:
        pass  # numerazione da 1: va bene lo stesso
    n_pezzi = len(indici)
    for L in logs:
        if len(L.mediane) != args.pezzi:
            L.avvisi.append(f"{len(L.mediane)} pezzi invece di {args.pezzi}")

    # dimensione
    gb = args.gb
    if gb is None:
        for L in logs:
            if L.gb:
                gb = L.gb
                break
    gb_pezzo = (gb / n_pezzi) if (gb and n_pezzi) else None

    # soglia
    tutti = [v for L in logs for v in L.mediane.values()]
    s_auto, c_v, c_l = soglia_automatica(tutti)
    soglia = args.soglia if args.soglia is not None else s_auto
    bimodale = (c_l - c_v) > 0.08 * (c_l + c_v) / 2  # gruppi distanti più dell'8 %

    col = (lambda s, c: s) if (args.senza_colori or not sys.stdout.isatty()) else (
        lambda s, c: f"\033[{c}m{s}\033[0m")

    # ---- intestazione ----
    print("disco_mappa — mappa pezzo × tempo")
    print(f"log letti: {len(logs)}   pezzi: {n_pezzi}   "
          f"file misurato: {gb:.2f} GB" if gb else
          f"log letti: {len(logs)}   pezzi: {n_pezzi}   file misurato: ? (passa --gb)")
    if gb_pezzo:
        print(f"un pezzo = {gb_pezzo:.3f} GB")
    print(f"soglia: {soglia:.1f} us  (centri trovati: veloce {c_v:.1f} us, lento {c_l:.1f} us"
          f"{'' if bimodale else ' — ATTENZIONE: gruppi poco distinti'})")
    for L in logs:
        for a in L.avvisi:
            print(f"  avviso {os.path.basename(L.path)}: {a}")
    print()

    # ---- tabella ----
    ore = [L.ora for L in logs]
    t0 = ore[0]
    intest = [L.ora.strftime("%H:%M") for L in logs]
    # giorno diverso dal primo: segnalo con il giorno
    for k, L in enumerate(logs):
        if L.ora.date() != t0.date():
            intest[k] = L.ora.strftime("%d/%H:%M")
    larg = max(5, max(len(s) for s in intest))
    print("pezzo " + " ".join(s.rjust(larg) for s in intest) + "   lento da")
    quando_lento: Dict[int, Optional[dt.datetime]] = {}
    stati: Dict[int, List[str]] = {}
    for i in indici:
        riga_stati = []
        for L in logs:
            v = L.mediane.get(i)
            riga_stati.append(classifica(v, soglia, args.margine) if v is not None else " ")
        stati[i] = riga_stati
        # primo L dopo il quale non torna più V
        ultimo_v = max((k for k, s in enumerate(riga_stati) if s == "V"), default=-1)
        primo_l_dopo = next((k for k, s in enumerate(riga_stati) if s == "L" and k > ultimo_v), None)
        if riga_stati and riga_stati[0] == "L" and ultimo_v < 0:
            quando_lento[i] = None  # già lento dal primo log
            da = "già lento"
        elif primo_l_dopo is None:
            quando_lento[i] = None
            da = "ancora veloce" if ultimo_v >= 0 else "?"
        else:
            quando_lento[i] = logs[primo_l_dopo].ora
            da = intest[primo_l_dopo]
        celle = []
        for s in riga_stati:
            if s == "V":
                celle.append(col("V".rjust(larg), "32"))
            elif s == "L":
                celle.append(col("L".rjust(larg), "31"))
            else:
                celle.append(s.rjust(larg))
        print(f"{i:5d} " + " ".join(celle) + f"   {da}")
    print()

    # ---- conteggio per orario ----
    print("orario    veloci  incerti   GB veloci   mediana veloci   mediana lenti")
    gb_veloci: List[float] = []
    ore_h: List[float] = []
    for k, L in enumerate(logs):
        nv = sum(1 for i in indici if stati[i][k] == "V")
        nq = sum(1 for i in indici if stati[i][k] == "?")
        vv = sorted(L.mediane[i] for i in indici if stati[i][k] == "V" and i in L.mediane)
        ll = sorted(L.mediane[i] for i in indici if stati[i][k] == "L" and i in L.mediane)
        mv = vv[len(vv) // 2] if vv else float("nan")
        ml = ll[len(ll) // 2] if ll else float("nan")
        g = nv * gb_pezzo if gb_pezzo else float("nan")
        gb_veloci.append(g)
        ore_h.append((L.ora - t0).total_seconds() / 3600.0)
        print(f"{intest[k]:>8}  {nv:6d}  {nq:7d}   {g:9.2f}   {mv:14.1f}   {ml:13.1f}")
    print()

    # ---- velocità ----
    if gb_pezzo and len(logs) >= 2:
        # la retta si calcola finché c'è qualcosa di veloce: dopo il primo log
        # a zero la curva è piatta e abbasserebbe la pendenza
        fine_k = next((k for k, g in enumerate(gb_veloci) if g <= 0), len(logs) - 1)
        xs, ys = ore_h[:fine_k + 1], gb_veloci[:fine_k + 1]
        a, b = retta(xs, ys)
        durata = ore_h[-1] - ore_h[0]
        print(f"durata osservata: {durata:.2f} h   (da {intest[0]} a {intest[-1]})")
        print(f"zona veloce: {gb_veloci[0]:.2f} GB -> {gb_veloci[-1]:.2f} GB")
        if fine_k < len(logs) - 1:
            print(f"zona veloce finita nel log delle {intest[fine_k]}")
        if xs[-1] > xs[0]:
            media = (ys[0] - ys[-1]) / (xs[-1] - xs[0])
            print(f"spostamento medio (dal primo log a quello delle {intest[fine_k]}): "
                  f"{media:.2f} GB/h")
        print(f"spostamento (retta ai minimi quadrati sugli stessi log): {-b:.2f} GB/h")
        if b < 0 and gb_veloci[-1] > 0:
            resto_h = gb_veloci[-1] / (-b)
            fine = logs[-1].ora + dt.timedelta(hours=resto_h)
            print(f"a questo ritmo la zona veloce finisce fra {resto_h:.1f} h "
                  f"(verso le {fine.strftime('%H:%M')})")
        print()
        print("fra un log e il successivo:")
        print("  da     a       ore     GB passati   GB/h")
        for k in range(1, len(logs)):
            dh = ore_h[k] - ore_h[k - 1]
            dg = gb_veloci[k - 1] - gb_veloci[k]
            r = dg / dh if dh > 0 else float("nan")
            print(f"  {intest[k-1]:>5}  {intest[k]:>5}  {dh:6.2f}   {dg:10.2f}   {r:6.2f}")
        print()
    elif not gb_pezzo:
        print("velocità in GB/h non calcolabile: manca la dimensione (passa --gb)\n")

    # ---- ordine ----
    diventati = sorted(((t, i) for i, t in quando_lento.items() if t is not None),
                       key=lambda x: (x[0], x[1]))
    if diventati:
        print("ordine in cui i pezzi sono diventati lenti:")
        seq = [i for _, i in diventati]
        gruppi: List[Tuple[dt.datetime, List[int]]] = []
        for t, i in diventati:
            if gruppi and gruppi[-1][0] == t:
                gruppi[-1][1].append(i)
            else:
                gruppi.append((t, [i]))
        for t, ps in gruppi:
            print(f"  {t.strftime('%H:%M')}: pezzi {', '.join(str(p) for p in ps)}")
        inv = sum(1 for x in range(len(seq)) for y in range(x + 1, len(seq)) if seq[x] > seq[y])
        tot = len(seq) * (len(seq) - 1) // 2
        if tot:
            print(f"  inversioni rispetto all'ordine crescente: {inv}/{tot} "
                  f"({'ordine di scrittura rispettato' if inv <= 0.1 * tot else 'ordine diverso'})")
        print()
    gia = [i for i in indici if stati[i] and stati[i][0] == "L" and all(s != "V" for s in stati[i])]
    anc = [i for i in indici if stati[i] and stati[i][-1] == "V"]
    if gia:
        print(f"già lenti dal primo log: {', '.join(str(p) for p in gia)}")
    if anc:
        print(f"ancora veloci nell'ultimo log: {', '.join(str(p) for p in anc)}")

    # ---- markdown ----
    if args.md:
        print()
        print("| pezzo | " + " | ".join(intest) + " |")
        print("|---:|" + "|".join(":-:" for _ in intest) + "|")
        for i in indici:
            print(f"| {i} | " + " | ".join(stati[i]) + " |")

    # ---- csv ----
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pezzo"] + [L.ora.isoformat(timespec="minutes") for L in logs])
            for i in indici:
                w.writerow([i] + [L.mediane.get(i, "") for L in logs])
        print(f"\nscritto {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
