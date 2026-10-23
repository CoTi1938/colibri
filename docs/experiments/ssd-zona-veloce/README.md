# La zona veloce del disco del Mac: studio e piano

Fascicolo per il MacBook Air M1 (8 GB, SSD 256 GB, circa 13 GB liberi, macOS 26.6.2).
Niente è stato fatto sul Mac: qui ci sono solo lettura, conti, un piano e due script.

| File | Cosa contiene |
|---|---|
| [`ricerca.md`](ricerca.md) | Che cosa si sa del controllore ANS2, delle cache SLC dinamiche e dei metodi per misurare un SSD da fuori. Ogni fatto con il link; [V] letto, [D] dedotto. |
| [`piano-prove.md`](piano-prove.md) | Sei prove da fare sul Mac, solo misure da fuori, al massimo 28 GB scritti in tutto. Per ognuna: cosa scrive, quanto dura, cosa aspettarsi. |
| [`motore.md`](motore.md) | Conviene far leggere al motore gli esperti più usati da un file nella zona veloce? Conti con i tuoi numeri e verdetto. |
| [`disco_mappa.py`](disco_mappa.py) | Legge i log di `disco leggi` presi a orari diversi; stampa la tabella pezzo per tempo (V/L) e la velocità dello spostamento in GB/h. Solo libreria standard. |
| [`scrivi_prova.py`](scrivi_prova.py) | Scrive i file di prova (byte casuali, F_NOCACHE, F_FULLFSYNC) e misura i MB/s di ogni GiB. Si rifiuta se lascerebbe meno di 4 GB liberi. |

## In due righe

- Quello che hai misurato (77-80 µs contro 98-101 µs, zona di circa 10 GB che cala in ordine di scrittura
  a 2,5-4 GB/h) è coerente con una cache SLC svuotata in ordine di scrittura. Nessuna fonte lo conferma
  per il disco Apple: nessuno lo ha misurato così. Le prove A-D lo chiudono.
- Sfruttarla nel motore non vale la pena: al massimo 6 ms su 100 per passo, nella pratica 1-3, per poche ore,
  e con 13 GB liberi probabilmente meno. Le leve migliori sono la percentuale di hit in RAM e più letture in volo.

## Come usare gli script

```sh
# mappa ogni 10 minuti (vedi piano-prove.md per le regole)
DISCO_ISTO=1 DISCO_BINS=32 disco leggi "$F" s16 30000 0 0 1 > log/leggi-$(date +%Y-%m-%dT%H-%M).txt

# tabella e GB/h
python3 disco_mappa.py --gb 22 log/*.txt

# file di prova da 8 GiB, con i MB/s di ogni GiB
python3 scrivi_prova.py file/c.bin 8
```

`disco_mappa.py` riconosce righe come `pezzo 12: mediana 77.3 us`, `[12] 77.3` o `12 77.3`,
si ferma alla parola "istogramma", e prende l'ora dal nome del log (o da una riga con data e ora, o dalla
data di modifica). Se il tuo `disco` stampa in un altro modo: `--regex` con due gruppi (indice, mediana).
