# Piano di prove: la zona veloce del disco, solo misure da fuori

Mac: MacBook Air M1, 8 GB, SSD 256 GB (`APPLE SSD AP0256Q`), circa 13 GB liberi, macOS 26.6.2.
Strumenti: `disco leggi`, `disco residenza` (tuoi), `scrivi_prova.py` e `disco_mappa.py` (in questa cartella).
Niente sudo, niente firmware, niente estensioni. Solo file nostri in una cartella temporanea, poi nel Cestino.

Legenda: [V] letto in una fonte (link in `ricerca.md`), [D] dedotto, [M] misurato da te il 19 o il 23 ottobre.

## 0. Regole comuni a tutte le prove

### Dove
- Cartella: `~/prova-disco.noindex/`, dentro `file/` e `log/`.
- Il suffisso `.noindex` tiene lontano Spotlight [V, Oakley 2024-07-09].
  Senza, Spotlight legge ogni file nuovo e aggiunge lavoro al disco.
- Fine di ogni prova: file nel Cestino, poi **svuota il Cestino**.
  Finché il file è nel Cestino lo spazio resta occupato e il disco non riceve il TRIM [D].
- Istantanee locali di Time Machine: `tmutil listlocalsnapshots /` (senza sudo).
  Se ce ne sono, i blocchi dei file cancellati restano vivi finché l'istantanea non sparisce [D].
  Prima delle prove metti Time Machine in pausa.

### Spazio
- `df -h /` prima e dopo ogni prova. Scrivi il numero nel log.
- Mai sotto 4 GB liberi: `scrivi_prova.py` si rifiuta da solo.
- Tutte le prove insieme scrivono al massimo 26 GiB (28 GB). Tabella in fondo.

### Il Mac
- Collegato alla corrente, coperchio aperto.
- `caffeinate -i` in un terminale per tutta la prova: altrimenti il Mac dorme, e con lui il disco.
- Niente motore, niente Foto o iCloud in sincronia, Time Machine in pausa.
- "Disco fermo" lo verifichi con `iostat -d -w 10 disk0` [V, man page]:
  colonne KB/t, tps, MB/s. A riposo MB/s deve stare vicino a zero.
  macOS scrive sempre qualcosa (log, Spotlight): qualche decina di KB/s è normale.

### Scrivere
- `python3 scrivi_prova.py file/a.bin 8` scrive 8 GiB di byte casuali, a pezzi da 1 GiB.
- Usa F_NOCACHE (i dati non restano in RAM) e alla fine fsync più F_FULLFSYNC,
  che "chiede al disco di scrivere tutti i dati bufferizzati sul supporto" [V, man fcntl].
- Stampa i MB/s di ogni GiB. Il crollo di velocità segna dove finisce la cache: è una seconda misura, gratis.
- Byte casuali: niente compressione, niente scorciatoie del controllore.

### Leggere
- Prima: `disco residenza file/a.bin` deve dire 0 % (o quasi). Se no stai misurando la RAM.
- Poi una mappa, col nome del log che porta l'ora (lo script la legge da lì):
  ```sh
  F=file/a.bin
  DISCO_ISTO=1 DISCO_BINS=32 disco leggi "$F" s16 30000 0 0 1 > log/leggi-$(date +%Y-%m-%dT%H-%M).txt
  ```
  30000 letture da 16 KiB = 0,49 GB letti, circa 3 secondi.
- Ogni 10 minuti per 4 ore (24 giri):
  ```sh
  for i in $(seq 1 24); do
    DISCO_ISTO=1 DISCO_BINS=32 disco leggi "$F" s16 30000 0 0 1 > log/leggi-$(date +%Y-%m-%dT%H-%M).txt
    sleep 597
  done
  ```
- Analisi: `python3 disco_mappa.py --gb 8.59 log/*.txt` (8 GiB = 8,59 GB; per il contenitore `--gb 22`).
  Stampa la tabella pezzo per ora, i GB veloci a ogni ora, i GB/h e l'ordine.

### Cosa dicono i numeri
- Letture da 16 KiB a caso, una per volta: 77-80 µs = zona veloce, 98-101 µs = zona lenta [M].
- La soglia automatica dello script cade a metà, circa 89 µs. Se serve: `--soglia 89`.
- Una mappa ogni 10 minuti disturba poco (0,5 GB letti) ma non è "disco fermo" assoluto.
  Se vuoi il fermo puro, fai una mappa ogni 30 minuti (`sleep 1797`, 8 giri).

## 1. Prova A: quanto dura la zona veloce senza scritture

Scrive: 0 GB se parti stasera sul contenitore (alle 20:31 aveva ancora circa 7,5 GB veloci [M]).
Altrimenti 8 GiB con `scrivi_prova.py`, poi aspetta 2 minuti.
Dura: 4 ore (24 mappe) più 5 minuti di analisi.
Come: regole comuni; disco fermo per tutte le 4 ore, solo le mappe.

Cosa aspettarsi:
- Ipotesi di lavoro (cache SLC svuotata in ordine di scrittura): i pezzi diventano lenti uno dopo
  l'altro partendo dal primo scritto; velocità fra 2,5 e 4 GB/h
  (oggi: 0,8 GB in 13 minuti, poi 1,5 GB in 36 minuti [M]). Con 7,5 GB restano 2-3 ore.
- Se il disco fermo svuota molto più in fretta (i dischi da recensione svuotano 50 GB in 4 minuti [V, Oakley 2025-03-17]):
  la zona sparisce nella prima ora, vedrai 2-3 mappe con qualcosa di veloce e poi niente.
- Se in 4 ore non si muove nulla: lo spostamento avviene solo quando il disco lavora. Lo dice la Prova B.
- Se i pezzi passano da veloce a lento e ritorno senza ordine: non è uno spostamento FIFO.
  Forse è una cache di mappatura del controllore. Lo chiarisce la Prova F.

Risultato: GB/h dalla retta, ora di fine prevista, ordine dei pezzi.

## 2. Prova B: il disco fermo svuota più in fretta del disco occupato?

Scrive: 8 GiB, file nuovo, a cache vuota (almeno 4 ore dopo l'ultima grossa scrittura,
o dopo che la Prova A ha mostrato la zona esaurita).
Dura: 4 ore: 1 h fermo, 1 h occupato, 1 h fermo, 1 h occupato. Mappa ogni 10 minuti per tutte e 4.

Occupato = un altro processo legge in continuazione un ALTRO file, con F_NOCACHE.
Va bene il file casuale da 6 GiB del 23 ottobre (dovrebbe essere tutto lento, vedi Prova 0).
Per non misurare in coda al carico, il carico si ferma 5 secondi prima di ogni mappa:
```sh
ALTRO=file/casuale6.bin     # il file da 6 GiB di oggi, se lo hai ancora
F=file/b.bin
for i in $(seq 1 6); do                      # un'ora occupata = 6 giri da 10 minuti
  ( while :; do disco leggi "$ALTRO" s16 30000 0 0 1 > /dev/null; done ) &
  B=$!
  sleep 570
  kill $B; sleep 5
  DISCO_ISTO=1 DISCO_BINS=32 disco leggi "$F" s16 30000 0 0 1 > log/leggi-$(date +%Y-%m-%dT%H-%M).txt
done
```
Il carico è leggero (letture da 16 KiB una per volta: circa 80 MB/s [D]).
Variante pesante: nell'ora occupata fai girare il motore con una risposta lunga.
È il carico vero, ma il motore usa anche la RAM e il processore: segna quale variante hai usato.

Cosa aspettarsi:
- Spostamento di sottofondo a disco fermo: GB/h nelle ore ferme > GB/h nelle ore occupate (oggi, occupato: 2,5 GB/h [M]).
- Spostamento "a traino" dell'attività: GB/h occupato maggiore o uguale a fermo.
- Uguali: lo spostamento va a tempo e non guarda l'attività.

Lettura: righe "fra un log e il successivo" dello script, sommate per ora.

## 3. Prova C: quanto è grande con 13 GB liberi

Scrive: 8 GiB (restano circa 4,4 GB liberi).
Prima: cache vuota, cioè almeno 4 ore dopo l'ultima grossa scrittura.
Se il contenitore ha ancora pezzi veloci, segna quanti GB: la cache ne sta già usando una parte [D].
Dura: 1 ora.

Come:
1. `df -h /`, segna i GB liberi.
2. `python3 scrivi_prova.py file/c.bin 8` e tieni l'uscita: è la tabella MB/s per GiB.
3. Aspetta 2 minuti. `disco residenza file/c.bin`, poi una mappa con `--gb 8.59`.
4. Dopo 30 minuti un'altra mappa, per vedere se è già in movimento.
5. Cancella il file, svuota il Cestino, `df -h /`.

Cosa aspettarsi:
- Nella scrittura: i primi X GiB veloci (un Air M1 da 256 GB scrive circa 2,2 GB/s [V, Notebookcheck 2022]),
  poi un gradino verso il basso. X è la grandezza della cache adesso.
- Nella mappa: pezzi veloci fino a X, lenti dopo. Le due misure devono dire lo stesso X.
- X fra 3 e 5 GB: cache dinamica, circa un terzo dello spazio libero
  (una cella usata come SLC tiene 1 bit invece di 3) [D].
- Nessun gradino in 8 GiB: cache di almeno 8 GB anche con 13 GB liberi: c'è una parte fissa
  o la parte dinamica usa anche la riserva nascosta del disco. Serve la Prova E.
- X = 0 (scrittura lenta da subito, tutto lento): con 13 GB liberi la cache non c'è più.
  Il 23 ottobre c'era perché prima di scrivere i 28 GB lo spazio libero era circa 41 GB [D].
- Oggi: 9,8 GB veloci su 22 scritti (0,8 + 9) [M], cioè cache di circa 10 GB con 41 GB liberi [D].
  La copia del contenitore andava a 0,48 GB/s (22 GB in 46 s [M]): o la sorgente era lenta,
  o la cache si è riempita a metà copia. La scrittura locale di byte casuali toglie il dubbio.

## 4. Prova D: in che ordine sposta i dati

Scrive: 4 GiB.
Dura: 2-3 ore (dipende dai GB/h della Prova A).
Prima: cache vuota, come per la Prova C.

Come: `python3 scrivi_prova.py file/d.bin 4 --ordine 3,1,4,2`.
Il terzo GiB del file viene scritto per primo, poi il primo, il quarto, il secondo.
APFS mette i blocchi sul disco nell'ordine in cui arrivano, quindi ordine di scrittura e posizione nel file sono diversi [D].
Poi mappe ogni 10 minuti con `--gb 4.29`. Con 32 pezzi ogni GiB sono 8 pezzi:
pezzi 0-7 = GiB 1, 8-15 = GiB 2, 16-23 = GiB 3, 24-31 = GiB 4.

Cosa aspettarsi:
- Ordine di scrittura (FIFO): diventano lenti prima i pezzi 16-23, poi 0-7, poi 24-31, poi 8-15.
- Ordine di posizione nel file: 0-7, poi 8-15, 16-23, 24-31.
- Dentro ogni GiB: in fila dal primo byte all'ultimo, oppure a blocchi sparsi
  (il controllore sposta blocchi fisici, non byte del file) [D].
- Nessun ordine: pezzi che cambiano a caso.
- Se la cache è più piccola di 4 GiB (lo dice la Prova C), i GiB scritti per ultimi nascono lenti:
  anche questo è ordine di scrittura, visibile già nella prima mappa.

Bonus a costo zero: subito dopo aver cancellato il file e svuotato il Cestino, scrivi 1 GiB
e guarda la velocità. Se è di nuovo veloce, il disco butta via i dati cancellati senza
spostarli, e la cache si libera col TRIM [D].

## 5. Prova E (facoltativa): quanto è grande con più spazio libero

Scrive: 6 GiB.
Prima: cancella i file delle prove e il file casuale da 6 GiB del 23 ottobre, svuota il Cestino.
Spazio libero atteso: 19-25 GB. Aspetta 4 ore o fai la prova la mattina dopo.
Dura: 1 ora.
Come: come la Prova C, con `scrivi_prova.py file/e.bin 6`.

Cosa aspettarsi:
- X cresce rispetto alla Prova C: cache dinamica che segue lo spazio libero.
- X uguale: cache di grandezza fissa.
- Due punti (13 GB e 19-25 GB liberi) bastano per una retta approssimativa [D].

## 6. Prova F (facoltativa, 0 GB): è il posto fisico o una cache del controllore?

Dura: 30 minuti.
Come:
1. Mappa di un file con pezzi veloci.
2. Leggi 20 GB di altri file dal disco, per esempio il contenitore:
   `disco leggi contenitore s16 1200000 0 0 1 > /dev/null` (circa 2 minuti, 20 GB letti).
3. Rifai la mappa.

Cosa aspettarsi:
- I pezzi veloci restano veloci: è il posto fisico dei dati (celle SLC), non una cache.
  I dati di oggi vanno già in questa direzione: 36 minuti di letture di altri file hanno
  tolto solo 1,5 GB, e in ordine di scrittura [M].
- Diventano tutti lenti insieme: è una cache del controllore svuotata dal traffico.
  Allora "zona veloce" vuol dire "mappatura appena usata", e il modo di sfruttarla cambia.

Variante sonno: chiudi il coperchio per un'ora fra due mappe, senza `caffeinate`.
Se la zona non si è mossa, il disco non lavora mentre il Mac dorme.

## 7. Prova 0 (0 GB, 5 minuti): due controlli prima di tutto

- Mappa del file casuale da 6 GiB del 23 ottobre.
  Atteso: tutto lento, perché è stato scritto a cache piena [D].
  Se invece è veloce, la cache non è FIFO o è più grande di quanto pensiamo: il piano cambia.
- `system_profiler SPNVMeDataType` e `diskutil info disk0` (senza sudo): modello del disco,
  TRIM, dimensione dei blocchi (4096 byte [V]). Da mettere nel fascicolo.

## 8. Tabella riassuntiva

| Prova | Scrive | Dura | Risposta che dà |
|---|---:|---:|---|
| 0 | 0 | 5 min | il file di oggi è lento? |
| A | 0 (contenitore) oppure 8 GiB | 4 h | durata, GB/h, ordine |
| B | 8 GiB | 4 h | fermo contro occupato |
| C | 8 GiB | 1 h | grandezza con 13 GB liberi |
| D | 4 GiB | 2-3 h | ordine di spostamento |
| E | 6 GiB | 1 h più attesa | grandezza con più spazio |
| F | 0 | 30 min | posto fisico o cache |

Totale scritture: 26 GiB = 28 GB (34 GiB se la Prova A richiede un file suo: in quel caso salta la E).

## 9. Sequenza consigliata

1. Stasera (23 ottobre): Prova 0, poi Prova A sul contenitore, 4 ore con `caffeinate -i`.
2. Domani mattina (cache vuota dopo la notte): Prova C, poi il bonus TRIM, poi Prova D.
3. Domani sera o dopodomani: Prova B (8 GiB nuovi, 4 ore).
4. Poi Prova E e Prova F se c'è tempo.

Dopo ogni prova: file nel Cestino, Cestino svuotato, `df -h /`, log in `log/` con il nome dell'ora.

## 10. Come cambiare l'ipotesi in base ai risultati

| Se vedi | Allora |
|---|---|
| FIFO lento (2-4 GB/h), uguale fermo o occupato | il controllore sposta a tempo, a ritmo basso; la zona dura ore |
| FIFO veloce a disco fermo (decine di GB/h) | sposta quando è fermo; la zona vive solo finché lavori |
| nessuno spostamento a fermo, sì a occupato | sposta a traino dell'attività |
| X cresce con lo spazio libero | cache dinamica: più spazio libero, più zona veloce |
| X fisso | parte fissa: la zona non dipende dallo spazio libero |
| tutto lento dopo 20 GB di altre letture | è una cache di mappatura, non celle SLC |
