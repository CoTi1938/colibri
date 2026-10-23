#!/usr/bin/env python3
"""scrivi_prova.py — scrive un file di prova con dati casuali e misura la velocità.

Serve alle prove del piano (piano-prove.md). Scrive N GiB in pezzi da 1 GiB,
stampa i MB/s di ogni pezzo (il crollo di velocità segna la fine della cache
SLC) e alla fine forza la scrittura fisica (fsync + F_FULLFSYNC).

Solo libreria standard. Niente sudo. Su macOS scrive con F_NOCACHE, così i
dati non restano nella memoria del sistema e `disco leggi` legge dal disco.

Uso:
    python3 scrivi_prova.py <file> <GiB>
    python3 scrivi_prova.py <file> 4 --ordine 3,1,4,2
    python3 scrivi_prova.py <file> 9 --pausa 0

--ordine: scrive i pezzi da 1 GiB in quell'ordine di POSIZIONE nel file
          (3,1,4,2 = prima il terzo GiB, poi il primo, il quarto, il secondo).
          Serve a capire se il disco sposta i dati per ordine di scrittura o
          per posizione nel file.
--pausa:  secondi di pausa fra un pezzo e l'altro (0).

Attenzione: non scrivere mai più di (spazio libero - 4 GB).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

GIB = 1 << 30
BLOCCO = 64 << 20  # 64 MiB per chiamata write


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("file")
    ap.add_argument("gib", type=int, help="quanti GiB scrivere")
    ap.add_argument("--ordine", default=None,
                    help="ordine dei pezzi da 1 GiB per posizione, es. 3,1,4,2")
    ap.add_argument("--pausa", type=float, default=0.0, help="secondi fra i pezzi")
    ap.add_argument("--con-cache", action="store_true",
                    help="NON usare F_NOCACHE (di norma lo usiamo)")
    args = ap.parse_args()

    if args.gib < 1:
        print("GiB deve essere >= 1", file=sys.stderr)
        return 2
    ordine = list(range(1, args.gib + 1))
    if args.ordine:
        ordine = [int(x) for x in args.ordine.split(",")]
        if sorted(ordine) != list(range(1, args.gib + 1)):
            print(f"--ordine deve contenere ogni numero da 1 a {args.gib} una volta",
                  file=sys.stderr)
            return 2

    st = os.statvfs(os.path.dirname(os.path.abspath(args.file)) or ".")
    liberi = st.f_bavail * st.f_frsize
    if liberi - args.gib * GIB < 4 * 10**9:
        print(f"spazio libero {liberi/1e9:.1f} GB: scrivere {args.gib} GiB lascerebbe "
              f"meno di 4 GB. Mi fermo.", file=sys.stderr)
        return 3

    fd = os.open(args.file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    if sys.platform == "darwin" and not args.con_cache:
        import fcntl
        F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
        fcntl.fcntl(fd, F_NOCACHE, 1)

    # un blocco casuale generato una volta: i byte sono casuali (niente
    # compressione possibile) e rigenerarli ogni volta falserebbe i tempi
    blocco = os.urandom(BLOCCO)
    per_gib = GIB // BLOCCO

    print(f"file: {args.file}")
    print(f"scrivo {args.gib} GiB in pezzi da 1 GiB, ordine di posizione: "
          f"{','.join(str(x) for x in ordine)}")
    print(f"inizio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("pezzo  posizione(GiB)   secondi    MB/s")
    t_tot0 = time.monotonic()
    velocita = []
    for k, pos in enumerate(ordine, 1):
        off = (pos - 1) * GIB
        t0 = time.monotonic()
        for b in range(per_gib):
            n = os.pwrite(fd, blocco, off + b * BLOCCO)
            if n != BLOCCO:
                print(f"scrittura corta ({n} byte) al pezzo {pos}", file=sys.stderr)
                return 4
        os.fsync(fd)
        dt_s = time.monotonic() - t0
        mbs = GIB / 1e6 / dt_s if dt_s > 0 else float("inf")
        velocita.append((pos, mbs))
        print(f"{k:5d}  {pos-1:7d}-{pos:<7d}   {dt_s:7.2f}  {mbs:7.0f}", flush=True)
        if args.pausa > 0 and k < len(ordine):
            time.sleep(args.pausa)
    if sys.platform == "darwin":
        import fcntl
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)   # il disco scrive davvero sulle celle
    else:
        os.fsync(fd)
    os.close(fd)
    tot = time.monotonic() - t_tot0
    print(f"fine: {time.strftime('%Y-%m-%d %H:%M:%S')}   totale {tot:.1f} s   "
          f"media {args.gib*GIB/1e6/tot:.0f} MB/s")

    # stima del "gradino": primo pezzo sotto la metà della velocità del primo
    if len(velocita) >= 2:
        v0 = velocita[0][1]
        gradino = next((k for k, (_, v) in enumerate(velocita, 1) if v < 0.5 * v0), None)
        if gradino:
            print(f"gradino di velocità al pezzo {gradino} (circa {gradino-1} GiB scritti "
                  f"veloci): possibile fine della cache")
        else:
            print("nessun gradino: tutti i pezzi alla stessa velocità")
    return 0


if __name__ == "__main__":
    sys.exit(main())
