# Sfruttare la zona veloce nel motore? Conti e verdetto

Domanda: conviene far leggere al motore gli esperti più usati da un file tenuto nella zona veloce?
Risposta breve: **no, non come funzione del motore**. Al massimo 6 ms su 100 per passo, nella pratica 1-3,
e per poche ore. Sotto i conti, i costi, e che cosa dovrebbe uscire dalle prove perché la risposta cambi.

Legenda: [C] letto nel codice del repo (file:riga), [M] misurato da te, [D] dedotto.

## 1. Come legge gli esperti il motore oggi

- Modello: Qwen3.6-35B-A3B, contenitore int4 gs64: 40 strati, 256 esperti per strato, 8 scelti per token.
  Un esperto sono 1.572.864 byte di pesi più 196.608 byte di scale: 1.769.472 byte, cioè i tuoi 1,77 MB [C `c/qwen36.c:1496`, `load_expert_merged`].
  Due `pread` per esperto: pesi e scale sono due tensori separati [C `c/qwen36.c:1523` e `1551`].
- I descrittori "gemelli" dei frammenti sono aperti con F_NOCACHE [C `c/compat.h:59`]: gli esperti non passano
  dalla memoria del sistema. È coerente con quello che vede `disco residenza`.
- Cache in RAM: per ogni strato un LRU di `cap` posti [C `c/qwen36.c:1620`, `expert_get`].
  Carichi a domanda uno alla volta nel thread principale [C `c/qwen36.c:2001`],
  più un solo thread PILOT che precarica lo strato dopo [C `c/qwen36.c:2353`, `pilot_worker`].
  Quindi in volo ci sono al massimo 2 letture, non 4: la tua misura "quattro insieme" è un po' generosa per questo motore [D].
- Contatore d'uso: `m->freq` per (strato, esperto) finché `HOT=n` non blocca i più usati [C `c/qwen36.c:1679`, `pin_hot_experts`].
  Non c'è storia su disco (`.coli_usage`): Qwen3.6 è l'eccezione fra i motori [C `docs/routing-telemetry.md`].
- Specchi: `st.h` sa usare copie parziali del modello, frammento per frammento, confrontando nome e intestazione
  [C `c/st.h:200`, `st_mirror_add`]. Il motore GLM le usa [C `c/colibri.c:9654`]; `qwen36.c` no (zero chiamate a `st_mirror`).
- Contenitore: un file per strato, `model-000NN.safetensors`, circa 0,55 GB l'uno, esperti in ordine di numero [C `c/tools/convert_qwen36.py:432`].

## 2. L'idea: un "file caldo"

1. Si contano gli esperti più usati (il contatore `freq` c'è già, manca solo scriverlo su disco).
2. Si scrive un file nuovo con i pesi e le scale di quegli esperti, grande meno della cache SLC.
3. Il motore, per quegli esperti, legge dal file caldo invece che dal contenitore.
4. Ordine di scrittura: **i più usati per ultimi**. Il disco svuota la cache in ordine di scrittura [M],
   quindi gli ultimi scritti restano veloci più a lungo [D].
5. Si rifà il file a ogni sessione, perché la zona veloce si consuma in ore.

Codice: una tabella (strato, esperto) -> posizione nel file caldo, un descrittore F_NOCACHE in più,
un ramo in `load_expert_merged`, un piccolo programma che costruisce il file. Circa 150 righe più prove [D].

## 3. Quanto si guadagna

Numeri di partenza [M]:
- passo di generazione 100 ms, di cui 43 ms di attesa del disco;
- zona veloce: 14 % di tempo del disco in meno sulle letture da 1,77 MB;
- zona veloce di circa 10 GB subito dopo la scrittura, che cala di 2,5-4 GB/h;
- spazio libero 13 GB.

Formula: guadagno per passo = 43 ms × 0,14 × f, dove f è la parte delle letture a vuoto
(esperti non in RAM) servita dal file caldo.

| f | guadagno per passo | passo | velocità |
|---:|---:|---:|---:|
| 0,3 | 1,8 ms | 98,2 ms | +1,8 % |
| 0,5 | 3,0 ms | 97,0 ms | +3,1 % |
| 0,7 | 4,2 ms | 95,8 ms | +4,4 % |
| 1,0 (impossibile: 22 GB non entrano in 10) | 6,0 ms | 94,0 ms | +6,4 % |

Quanto vale f? Il file caldo copre una parte del contenitore: 6 GB su 22 sono il 27 %.
Gli esperti non sono usati tutti uguale, quindi f sta sopra il 27 %, ma la RAM si tiene già i più caldi
e lascia al disco quelli tiepidi: f realistico 0,3-0,5 [D]. Si misura in dieci minuti con il contatore `freq`.

Il guadagno poi cala nel tempo: a 3 GB/h un file da 6 GB è sparito in 2 ore;
su una sessione di 3 ore il guadagno medio è circa la metà della tabella [D].

## 4. Che cosa costa

- **Spazio libero.** Il file caldo prende 6-8 GB dei tuoi 13. Se la cache è dinamica (un terzo dello spazio libero),
  con 5 GB liberi la cache è 1-2 GB: il file caldo non ci entra nemmeno [D]. È il problema che da solo chiude il discorso su questo Mac.
  La Prova C del piano dice quanto è grande la cache con 13 GB liberi: se è meno di 4 GB, f scende sotto 0,2.
- **Scritture.** 6-8 GB a sessione. A una sessione al giorno sono 2-3 TB l'anno, contro una stima di circa 300 TB
  di durata per un SSD Apple da 256 GB [V, 9to5Mac 2021]: circa 1 % l'anno. Non è un blocco, ma è usura vera.
- **Tempo di avvio.** 6-8 GB da scrivere: 3-4 s a 2 GB/s, 15 s se il disco va come la copia di oggi (0,48 GB/s) [M].
- **Codice e manutenzione.** 150 righe, un formato in più, un file in più da tenere coerente col contenitore.
- **Rischio zero per il disco**: sono file normali, cancellabili. Ma è una scorciatoia che dipende da un comportamento
  non documentato del firmware, che può cambiare con un aggiornamento di macOS [D].

## 5. Le leve che rendono di più, con gli stessi numeri

- **Percentuale di hit in RAM.** I 43 ms sono le letture a vuoto. Se oggi gli hit sono h, alzarli di 10 punti
  taglia le letture a vuoto di 10/(100-h) %. Con h = 70 %: un terzo in meno, cioè 14 ms per passo.
  Con h = 50 %: un quinto, 8,6 ms. In entrambi i casi più dei 6 ms massimi del file caldo, e dura tutta la sessione.
  Leve: `--cap` più alto se la RAM lo permette, `HOT=n` per bloccare i più usati, `PILOT`/`WIDE` per precaricare meglio.
- **Più letture in volo.** Il motore tiene al massimo 2 letture in volo [C]. Su NVMe la banda sale con le letture
  parallele. Si misura in 5 minuti con `disco`: letture da 1,77 MB sulla zona lenta a 1, 2, 4, 8 insieme.
  Se 4 in volo danno il 30 % in più di banda, vale più del 14 % della zona veloce e non consuma nulla [D].
- **`CACHE_ROUTE`** (già nel repo, spento di serie): preferisce esperti residenti. Cambia l'instradamento,
  quindi resta un esperimento esplicito, mai acceso di nascosto [C `docs/CACHE_ROUTE.md`].

## 6. Verdetto

- Come funzione del motore: **no**. Troppo poco (1-3 %), per poche ore, con spazio libero che non hai.
- Come esperimento manuale: solo se le prove A, B e C dicono tutte e tre:
  zona veloce di almeno 8 GB con 13 GB liberi, durata di almeno 4 ore, e svuotamento lento anche col motore acceso.
  Allora f può arrivare a 0,6-0,7 e il guadagno a 4 %: un pomeriggio di lavoro per misurarlo davvero.
- Prima di quello, misura le due leve del punto 5: costano zero scritture e possono valere il doppio.
