# Ricerca: che cosa si sa (ogni fatto col suo link)

Legenda: [V] letto nella fonte, con citazione breve in inglese come sta nella pagina; [S] solo dal riassunto di un motore di ricerca,
pagina non aperta; [D] dedotto da noi; [M] misurato da te.
Le fonti sono in inglese; le citazioni sono lasciate così per non tradurre male i numeri.
Come ho lavorato: ho aperto le pagine e copiato le frasi; le fonti principali
(driver Linux, TechInsights, Oakley 2020 e 2025, Tom's Hardware, Samsung, Mac Performance Guide, Chen 2009 e 2011, Park 2026, Yang 2024, Queenie)
le ho riaperte una seconda volta per controllare che le frasi ci fossero. Dove non ci sono riuscito lo dico accanto al fatto.

## 1. Il controllore del disco degli M1 (ANS2)

### 1.1 Quello che sa Asahi Linux

- Il controllore si chiama ANS (o ANS2) ed è dentro il chip M1, non su un bus PCIe:
  "Apple SoCs such as the M1 come with an embedded NVMe controller that is not attached to any PCIe bus" [V]
  Sven Peter, lista del kernel, 26 aprile 2022. https://lkml.indiana.edu/hypermail/linux/kernel/2204.3/04217.html
- Non rispetta lo standard NVMe:
  "it doesn't conform to the NVMe specification and requires a bunch of changes to command submission and IOMMU configuration" [V] stessa fonte.
- Gira un sistema operativo proprietario sul coprocessore:
  "The NVMe controller is running a proprietary RTOS Apple calls RTKit" [V]
  Sven Peter, lettera di copertina delle patch, 21 marzo 2022. https://lore.kernel.org/lkml/20220321165049.35985-1-sven@svenpeter.dev/T/
- È lo stesso hardware dei Mac con T2: "The hardware here is the same hardware that's already used in T2 Macs." [V] stessa fonte.
- Emula NVMe: "ANS2 is a co-processor that emulates an NVMe MMIO interface and uses the tag as an index to an internal data structure." [V]
  Sven Peter citato da Keith Busch, 25 settembre 2021. https://lkml.iu.edu/hypermail/linux/kernel/2109.3/01532.html
- Una sola coda di comandi, profondità 64:
  "The Apple NVMe controller only supports a single admin and a single IO queue which are both limited to 64 entries and share a single interrupt." [V]
  Driver `drivers/nvme/host/apple.c` del kernel Linux. https://raw.githubusercontent.com/torvalds/linux/master/drivers/nvme/host/apple.c
- La coda è in realtà un vettore indicizzato dal tag, e ogni comando deve stare anche nella tabella dell'IOMMU interno (NVMMU):
  "The submission "queue" instead is an array indexed by the command tag on this hardware. Commands must also be present in the NVMMU's tcb array." [V] stessa fonte.
- I tag non superano 0x40: "tags cannot be higher than 0x40 which effectively limits the combined queue depth to 0x40." [V] stessa fonte.
  Nel codice: `apple_nvme_t8103_hw` (M1) ha `.max_queue_depth = 64`, il t8015 (A11) 16 [V].
- Comandi proprietari Apple esistono (opcode 0xD8 sulla coda admin) ma Linux non li usa; passano da un filtro di indirizzi chiamato SART,
  con al massimo 16 intervalli: "It allows to specify up to 16 physical address ranges that are allowed" [V] lettera di copertina sopra.
- Se il firmware del coprocessore si pianta non c'è rimedio: "there's unfortunately no known way to recover if RTKit crashed :(" [V] driver sopra.
- Il firmware ANS sta nella NOR, pesa circa 800 KB compressi, lo carica iBoot1; c'è un secondo firmware, "S5E", descritto come
  controllore della NAND (SSD), che sta in una flash sul chip e viene avviato da ANS [V]
  Tabella dei firmware, documentazione Asahi. https://asahilinux.org/docs/platform/introduction/
- Hector Martin (2021): i moduli NAND "grezzi" sono collegati in PCIe a porte dedicate dietro ANS, e ANS fa (almeno in parte) la traduzione degli indirizzi:
  "Yes, the raw NAND storage modules are connected over PCIe on all M1 machines, to dedicated root ports that are behind ANS." [V]
  "As far as I can tell ANS implements (parts of?) the FTL and data striping and other high-level features" [V]
  https://news.ycombinator.com/item?id=28079038
- Il formato di coda non standard serve a scaricare la cifratura sul controllore:
  "The non-standard NVMe queue format is because they support offloading encryption/decryption to the NVMe controller" [V] stessa pagina.
- Nel rapporto Asahi di ottobre 2021: NVMe "breaks the spec in multiple ways" ed è "managed by an ASC, the "ANS"" [V]
  https://asahilinux.org/2021/10/progress-report-september-2021/
- In m1n1 (il bootloader Asahi) la coda è da 64 e i blocchi sono da 4096 byte:
  `#define NVME_QUEUE_SIZE 64`, "1 block a 4096 bytes" [V] https://raw.githubusercontent.com/AsahiLinux/m1n1/main/src/nvme.c
- Non esiste una pagina wiki "HW:ANS": il driver, le lettere delle patch e i commenti di marcan sono tutto quello che c'è [V, cercato].

Che cosa ne segue [D]:
- Dal lato Mac non c'è nessun comando documentato per sapere dove stanno fisicamente i dati, né statistiche della FTL,
  né la geometria della NAND. Si vede solo l'interfaccia NVMe emulata e lo SMART.
- La politica della cache SLC è nel firmware di ANS (e forse nel S5E). Nessuna fonte la descrive.
- Profondità di coda 64 in tutto: con 2-4 letture in volo siamo lontani dal limite.

### 1.2 Quello che vede macOS senza sudo

- Su un MacBook Air M1 l'albero dei dispositivi mostra `IOClass = AppleANS3NVMeController`, `Physical Interconnect = "Apple Fabric"`,
  `role = "ANS2"`, `Encryption Type = "AES-XTS"`, blocco logico e fisico 4096 byte, NVMe 1.10; smartctl funziona:
  "Fortunately the SSD is still exposed as NVMe storage so SMART queries will work." [V]
  Thomas Kaiser, novembre 2020. https://raw.githubusercontent.com/ThomasKaiser/Knowledge/master/articles/Exploring_Apple_Silicon_on_MacBookAir10.md
- Un M1 da 256 GB si presenta come "APPLE SSD AP0256Q", firmware "1161.80.", 3 namespace, e riporta "Percentage Used" e "Data Units Written" [V]
  https://discussions.apple.com/thread/252532855?page=2
- "Percentage Used" è una stima da byte scritti e cicli ipotizzati, non una misura indipendente [V]
  Oakley, 26 febbraio 2026. https://eclecticlight.co/2026/02/26/how-long-will-my-macs-ssd-last/
- Stima di durata: circa 300 TB scritti per un SSD Apple da 256 GB [V] 9to5Mac, 11 marzo 2021. https://9to5mac.com/2021/03/11/m1-mac-ssd-wear/
- Oakley: "conservative estimates are around 3,000 cycles in recent models."; tieni libero almeno il 10 %, meglio 15-25 % [V] articolo del 26 febbraio 2026 sopra.
- TRIM è sempre attivo sugli SSD interni Apple: "all Apple internal SSDs Trim reliably, without any need for tweaking any settings." [V]
  Oakley, 13 novembre 2025. https://eclecticlight.co/2025/11/13/does-that-ssd-trim-and-why-is-it-important/
- Dal Mac, senza sudo: `system_profiler SPNVMeDataType` e `diskutil info disk0` mostrano modello e TRIM [D].

### 1.3 Smontaggi: quali chip

- iFixit, 19 novembre 2020, Air e Pro M1: "(Western Digital/SanDisk?) SDRGJHI4 – 128GB Flash storage (x2)"; sul Pro anche
  "Kioxia KICM232 VD6303 CHNA1 2029 flash storage". Nessun chip controllore separato nella lista [V]
  https://www.ifixit.com/News/46884/m1-macbook-teardowns-something-old-something-new
- TechInsights, scheda del 31 marzo 2021, Air A2337: "KIOXIA provided 256 GB NAND Flash memory in two 128 GB packages (KICM225 and KICM232)."
  e "Both RAM and NAND are soldered to the PCB." [V] https://www.techinsights.com/products/ddt-2012-810
- Quindi il 256 GB dell'Air M1 ha due pacchetti da 128 GB (le due fonti non concordano sul marchio: SanDisk o Kioxia; i due marchi
  condividono le fabbriche) [V+D].
- Notebookcheck, 2 dicembre 2022: l'Air M1 "is still equipped with the AP0256Q, which uses two memory chips and manages higher transfer rates." [V]
  https://www.notebookcheck.net/Apple-s-MacBook-Air-M1-ist-still-shipped-with-the-faster-256-GB-SSD.672370.0.html
- Il controllore sta nel chip M1. Oakley: le funzioni del controllore sono divise fra la NAND fuori dal chip e i core dentro il chip,
  e il volume dati è sempre cifrato [V] 6 marzo 2024. https://eclecticlight.co/2024/03/06/apple-silicon-memory-and-internal-storage/
- Il 2022 ha mostrato quanto contano i due pacchetti: il Pro M2 da 256 GB con un solo chip NAND leggeva a 1446 MB/s e scriveva a 1463,
  contro 2900 e 2215 dell'M1 [V] 9to5Mac, 26 giugno 2022. https://9to5mac.com/2022/06/26/m2-macbook-pro-slower-ssd/
  Apple (Michelle Del Rio): "These new systems use a new higher density NAND that delivers 256GB storage using a single chip." [V]
  https://9to5mac.com/2022/07/14/m2-macbook-air-slower-ssd-base-model/
- Un Air M1 da 256 GB misurato da Notebookcheck: 2798 MB/s in lettura, 2194 in scrittura [V]
  https://www.notebookcheck.net/M2-MacBook-Pro-13-with-256-GB-storage-has-a-33-50-slower-SSD-than-its-M1-equivalent.631793.0.html
- TLC o no: **nessuna fonte primaria lo dice** per i pacchetti KICM225/KICM232/SDRGJHI4 [V, cercato in tre modi].
  Un forum lo attribuisce a "96-layer 3D TLC BiCS" Kioxia, senza fonte [non verificato]:
  https://forums.appleinsider.com/discussion/233309/need-help-does-the-2023-m2-mac-mini-use-tlc-or-qlc-nand-flash-for-storage
  Deduzione [D]: nel 2020 i pacchetti Kioxia/SanDisk da 128 GB erano BiCS a 3 bit per cella; Oakley scrive che "Most consumer SSDs store three bits per cell"
  (vedi 1.4). Lo prendiamo per vero, ma resta una deduzione.

### 1.4 Howard Oakley (eclecticlight.co): misure e opinioni

- 12 dicembre 2020, Mac mini M1 500 GB: Blackmagic "between 2.7 and 3.5 GB/s"; ATTO a un flusso scrive "just under 4 GB/s";
  a 4 flussi "Write speed has fallen fairly uniformly to around 2.5 GB/s" [V] https://eclecticlight.co/2020/12/12/how-fast-is-the-ssd-inside-an-m1-mac/
  Nei commenti: "I don't think that there's any way to control any caching which might take place." [V]
- 17 dicembre 2020, Stibium: "overall transfer rates of 2.7 (read) and 2.9 (write) GB/s" [V]
  https://eclecticlight.co/2020/12/17/comparing-ssd-performance-between-t2-and-m1-macs-with-stibium/
- 3 marzo 2021, Pro e mini M1: 2,9 GB/s lettura, 2,7 scrittura; la velocità di scrittura del mini è calata dopo circa 48 s,
  attribuito a Time Machine [V] https://eclecticlight.co/2021/03/03/back-to-measuring-the-performance-of-m1-mac-internal-storage/
- 22 aprile 2022, commento di Ric Ford su un mini M1 8 GB / 256 GB: "transfer performance dropped dramatically when using 16GB files" [V]
  https://eclecticlight.co/2022/04/22/what-performance-to-expect-from-ssds/
  È l'unico indizio pubblico di un gradino di cache su un SSD Apple da 256 GB, e cade fra 10 e 16 GB [D].
- 24 aprile 2022, commenti: "Very few Mac users ever come across a situation where the SLC Write cache is full." e
  "I don't know of any benchmark for the Mac which tries to exhaust SLC Write Cache." [V]
  https://eclecticlight.co/2022/04/24/last-week-on-my-mac-opening-pandoras-ssd/
- 17 marzo 2025, "Why SSDs slow down, and how to avoid it": "Most consumer SSDs store three bits per cell"; la cache dinamica è
  "borrowed from main storage cells by writing single bits to them as if they were SLC"; una cache statica "up to 16 GB";
  "In practice, its size ranges from around 2.5% of total SSD capacity to over 10%."; "once the dynamic cache is full, write speed suddenly falls.";
  SSD Thunderbolt 5 da esempio: cache 50 GB, da 5,5 a 1,4 GB/s, "It then takes 4 minutes for the cache to be emptied";
  "Even an Apple silicon Mac's internal SSD can slow alarmingly in the wrong circumstances."; "Keep ample free space on the SSD so the whole of its SLC write cache can be used." [V]
  https://eclecticlight.co/2025/03/17/why-ssds-slow-down-and-how-to-avoid-it/
  Nei commenti: "Internal SSDs in Macs, like all similar SSDs, rely on SLC write cache for their speed." e
  "Cache sizes are heavily dependent on the total capacity of the SSD." [V]
- 5 marzo 2025: regola del 20-25 % libero, "never falls below 50 GB free space", meglio 100 GB [V]
  https://eclecticlight.co/2025/03/05/should-you-pay-a-premium-price-for-a-bigger-internal-ssd/
- Oakley non ha mai pubblicato una misura del gradino SLC su un SSD interno Apple; la sua posizione è che c'è, ma che quasi nessuno lo incontra [D dalle pagine sopra].
- Strumento: Stibium, 400 file da 2 MB a 2 GB in ordine casuale, velocità dalla pendenza tempo/dimensione [V]
  https://eclecticlight.co/2021/03/03/back-to-measuring-the-performance-of-m1-mac-internal-storage/ e https://eclecticlight.co/downloads/

## 2. Le cache SLC dinamiche

### 2.1 Che cos'è

- Celle normali usate a 1 bit: la cache dinamica è "borrowed from main storage cells by writing single bits to them as if they were SLC";
  quasi tutti i dischi hanno anche una cache statica "up to 16 GB"; la dinamica "ranges from around 2.5% of total SSD capacity to over 10%" [V]
  Oakley, 17 marzo 2025. https://eclecticlight.co/2025/03/17/why-ssds-slow-down-and-how-to-avoid-it/
- Un blocco in modo SLC tiene un terzo di un blocco TLC: "An SLC block has only one third the capacity of a TLC block"; due tipi:
  "static SLC cache in which blocks can only be used for in SLC mode; and dynamic SLC cache in which blocks can be used in SLC mode or TLC mode";
  "Most current mobile storage devices use dynamic SLC cache." [V] brevetto Micron US10983829B2 (2021). https://patents.google.com/patent/US10983829B2/en
- Samsung TurboWrite (2013): "a high-performance write buffer area within the drive that simulates high-performance SLC";
  grandezza 3/3/6/9/12 GB per 120 GB/250 GB/500 GB/750 GB/1 TB; "the buffer size is defined in SLC terms, therefore the physical capacity will be 3x greater" [V]
  https://download.semiconductor.samsung.com/resources/white-paper/Samsung_SSD_TurboWrite_Whitepaper_EN.pdf
- Nomi dei produttori: TurboWrite (Samsung), nCache (SanDisk/WD), Dynamic Write Acceleration (Micron) [V]
  Tom's Hardware, recensione SK hynix P41. https://www.tomshardware.com/reviews/sk-hynix-platinum-p41-ssd-review/3

### 2.2 Quanto è grande, e come dipende dallo spazio libero

- Samsung 970 EVO Plus: parte fissa 4/4/6 GB più parte dinamica 9/18/36 GB (250 GB/500 GB/1 TB) [V]
  https://pcper.com/2019/01/samsung-970-evo-plus-nvme-ssd-review-the-new-write-speed-leader/
- La parte dinamica ha bisogno di spazio libero: sul 970 EVO "TurboWrite does not work on any Samsung SSD 970 EVO series drive that has than 18GB of free space" [V, refuso nell'originale]
  https://www.legitreviews.com/samsung-ssd-970-evo-nvme-1tb-ssd-review_204678/6
- Samsung 980 PRO: 4 GB fissi + 45 GB dinamici (250 GB), 6 + 108 (1 TB) [V] https://www.allround-pc.com/artikel/storage/test-samsung-ssd-980pro-1tb
- WD SN850 1 TB: "The total dynamic SLC capacity spans roughly one-third of the available capacity (300GB on our 1TB sample) with a small and quick-to-recover static SLC cache (12GB on our 1TB sample)" [V]
  https://www.tomshardware.com/reviews/wd-black-sn850-m-2-nvme-ssd-review
- WD SN770 1 TB, senza DRAM e tutta dinamica: "absorbed roughly 377GB of data at a rate of 4,850 MBps before it degraded." [V]
  https://www.tomshardware.com/reviews/wd-black-sn770-ssd-review/3
- Regola pratica: "Many SSDs allocate about a third of their free space to be used as dynamic SLC cache."; la statica "is generally tiny, less than 10GB even on large 2TB drives" [V]
  https://www.technipages.com/what-is-slc-caching/
- Intel 660p (QLC) da 500 GB: 76 GB di cache a disco vuoto, 41 GB a metà, 6 GB sopra il 75 % [S]
  https://www.legitreviews.com/intel-ssd-660p-1tb-ssd-review-with-qlc-nand-flash_207025/6
- Micron M600: parte in modo SLC e "As the SSD is filled past 50% capacity, the controller intelligently switches dies from SLC to MLC" [V]
  https://pcper.com/2014/09/micron-m600-ssd-review-digging-into-dynamic-write-acceleration/2/
- Su un Mac: Mac Studio M1 Ultra da 8 TB, 1000 file da 5,65 GiB: "The plummet occurs at file ~420 of 1000, equating to about 33% of the 8TB flash drive.";
  da 7 GB/s a "3.2 to 3.3 GB/sec"; "This looks like 2-tier flash behavior." [V] Mac Performance Guide, 19 giugno 2022.
  https://macperformanceguide.com/AppleMacStudioM1Ultra-SSD.html
  Un terzo di un disco quasi vuoto: è la regola del terzo dello spazio libero, su un SSD Apple [D].
- Mini M1 8 GB / 256 GB: "transfer performance dropped dramatically when using 16GB files" [V] (vedi 1.4).

### 2.3 Quando si svuota, e a che velocità

- Samsung 2013: "during the idle periods, the data is moved from the buffer to primary storage region"; tempo di inattività necessario
  per svuotare tutto: 18,5 s per i 3 GB del modello da 250 GB, 55 s per i 12 GB del 1 TB [V] (PDF Samsung sopra). Cioè circa 10 GB al minuto [D].
- Micron DWA: "Drive firmware may use interface idle time to reduce physical saturation and recover acceleration capacity." [V]
  https://www.kitguru.net/components/ssd-drives/anton-shilov/micron-announces-m600-ssds-ups-performance-reduces-power-consumption/
  La scheda tecnica Micron definiva "Interface Idle Time – Periods of time between commands greater than 50ms." [S, PDF non più in rete]
- WD SN770: "It evacuated its SLC cache at a rate of roughly 30GB per every minute of idle." [V] (Tom's Hardware, sopra)
- Samsung 980 PRO: "While the static 6GB cache recovered immediately, the 'Intelligent' dynamic cache did not recover within a half-hour idle window." [V]
  https://www.tomshardware.com/reviews/samsung-980-pro-m-2-nvme-ssd-review/2
- Seagate FireCuda 120: cache di 24-26 GB, "only 6GB of the cache recovers within a half-hour of idle rounds." [V]
  https://www.tomshardware.com/uk/reviews/seagate-firecuda-120-sata-ssd-review/2
- Sabrent Rocket 4 Plus: recupero "roughly 50GB-100GB per minute." [V] https://www.tomshardware.com/reviews/sabrent-rocket-4-plus-m2-nvme-ssd-review/3
- Samsung 990 PRO: "The drive may engage in some folding in the background which can create an effectively larger cache." [V]
  https://www.tomshardware.com/reviews/samsung-990-pro-ssd-review/2
- Oakley: SSD Thunderbolt 5, cache 50 GB, "It then takes 4 minutes for the cache to be emptied"; e "I'm not sure whether anyone has determined how long is necessary between writes" [V]
  https://eclecticlight.co/2025/03/17/why-ssds-slow-down-and-how-to-avoid-it/ e https://eclecticlight.co/2025/01/30/thunderbolt-5-ssds-are-promising-but-fill-their-cache-quicker/
- Yang 2024 (Crucial P1 500 GB): 10 minuti di pausa bastano a svuotare 20 GB [V] (vedi sezione 3).
- Park 2026: "some tested SSDs retain either the oldest or the newest data in the SLC buffer even after long idle times" [V] (vedi sezione 3).
- Mac Studio: "SSD speed "regenerates" after the first test, so long as there is some time in-between." [V] (Mac Performance Guide, sopra)
- Nessuna fonte misura lo svuotamento DURANTE le letture: tutti misurano a disco fermo o durante le scritture [D, cercato].

Il quadro: i dischi da recensione svuotano da 10 a 100 GB al minuto a disco fermo, ma alcuni (980 PRO, FireCuda) tengono la parte dinamica
per più di mezz'ora. I tuoi 2,5-4 GB all'ora sono fuori scala rispetto ai primi e in linea con i secondi [D].

### 2.4 In che ordine

- "Samsung's TurboWrite, as a point of comparison, was designed to work in a first-in, first-out (FIFO) fashion, with the static cache being filled first." [V]
  Tom's Hardware, recensione P41 (sopra).
- Brevetto Silicon Motion US10740013B2 (2020): il controllore "selects three first SLC-data blocks, which have had valid data for the longest time (the oldest valid data)"
  e li riprogramma in un blocco TLC [V; al secondo tentativo la pagina ha risposto 503, quindi non riaperta]
  https://patents.google.com/patent/US10740013B2/en
- Park 2026: alcuni dischi tengono "the oldest", altri "the newest" [V] (sezione 3).
- Nessuna fonte parla di ordine per posizione (LBA) [D]. Gli schemi "per calore" esistono solo nella ricerca, non nei dischi in vendita [D, da Yang 2024].

### 2.5 Quanto ci mette una cella a essere letta

- SLC planare, scheda Micron MT29F1G08: "Read page: 25µs" [V]
  https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/306/MT29F1G08,16_Rev.E_July2012.pdf
- TLC 3D, scheda Frontgrade UT81NDQ512G8T: lettura di pagina a un piano 74 µs, "SNAP READ" 51 µs, programmazione 1900 µs [V]
  https://www.frontgrade.com/sites/default/files/documents/datasheet-ut81ndq512g8t.pdf
- Parametri di un simulatore 3D ibrido (Yang 2024): "0.02 ms for SLC read; 0.066 ms for TLC read; 0.5 ms for SLC write; 3 ms for TLC write" [V]
  https://arxiv.org/pdf/2409.14360
- Toshiba XL-FLASH (SLC): lettura sotto i 5 µs contro circa 50 µs del TLC 3D [S] https://www.anandtech.com/show/14707
- Rassegna 2025: SLC 20-25 µs, TLC 66-170 µs, QLC 120-200 µs [S] https://arxiv.org/pdf/2507.10573
- Letture casuali più veloci dalla pSLC, su un disco dove si poteva spegnerla: "In random scenarios like this, the pSLC Cache significantly improved read performance." [V]
  https://theoverclockingpage.com/2024/08/23/the-impact-of-slc-cache-in-performance-of-an-nvme-ssd-benchmarks-and-results/?lang=en
- Dati in SLC letti "artificialmente" veloci: Park 2026 [V] (sezione 3).

Il quadro: fra SLC e TLC la cella da sola fa 30-50 µs di differenza. Tu ne vedi 22 su 77-100 totali: la cella è una parte del tempo,
il resto è controllore, cifratura e trasferimento, e il TLC può usare letture rapide ("snap read") per una parte delle pagine [D].

### 2.6 Cache piena e disco quasi pieno

- Samsung 2013: "the buffer will eventually become full. At this point, the transfer will exit TurboWrite and write data directly to the main storage area";
  velocità da 520 a 270 MB/s sul 250 GB [V] (PDF Samsung).
- Samsung 990 PRO: "After the pSLC cache runs out, the 990 Pro maintains around 1.4 GBps in a direct-to-TLC mode." e "you are more likely to hit this state when the drive is fuller." [V]
  https://www.tomshardware.com/reviews/samsung-990-pro-ssd-review/2
- Brevetto Micron: "If there is no free SLC cache, the memory controller can directly write the data to MLC storage" [V] (sopra).
- La statica resta anche a disco pieno: "The static cache is always available, even when the drive is almost full." [V] (technipages, sopra).
- Oakley: "SLC write cache can only use storage that's not already in use"; "If the SSD only has 100 GB free, then that can't all be used for cache" [V] (17 marzo 2025).

Per i tuoi 13 GB liberi [D]: se c'è una parte fissa (3-16 GB nei dischi comuni) resta quella; la parte dinamica sarebbe al massimo
un terzo di 13 GB, circa 4 GB, e con meno di un certo spazio libero i produttori la spengono del tutto (18 GB sul 970 EVO).
La Prova C del piano misura proprio questo.

### 2.7 SSD interni Apple e cache SLC

- Mac Studio M1 Ultra 8 TB: gradino a circa un terzo del disco, da 7 a 3,2 GB/s, ripetibile, uguale sul MacBook Pro M1 Max del 2021 [V] (sopra).
- Oakley: "Internal SSDs in Macs, like all similar SSDs, rely on SLC write cache for their speed." [V] (commento, 17 marzo 2025).
- Un utente esperto sui forum Apple (2021): la velocità cala "once the SSD's write cache is exhausted which can occur in as little as 30 seconds" [V]
  https://discussions.apple.com/thread/252698867
- Air M1 256 GB, Blackmagic (5 GB, quindi dentro la cache): 2190 MB/s in scrittura, 2676 in lettura [V]
  https://www.notebookcheck.net/M1-MacBook-Air-s-SSD-is-twice-as-fast-as-that-of-the-latest-Intel-based-model.504454.0.html
- Nessuno ha pubblicato la grandezza della cache di un Air M1 da 256 GB, né una misura come la tua (latenza di lettura per zona) [D, cercato].

## 3. Ricavare l'interno di un SSD misurando da fuori

Quello che fa il tuo `disco` ha una tradizione: letture piccole e ripetute, latenze, e poi si deduce la struttura.

- **Chen, Koufaty, Zhang, SIGMETRICS 2009**, "Understanding Intrinsic Characteristics and System Implications of Flash Memory based Solid State Drives".
  Tre SSD SATA, letture da 4 KB con I/O diretto, latenza per richiesta con blktrace. Dalle latenze deducono il read-ahead
  ("sequential has latencies of only 75-90µs, which is nearly 65% less than stride and random (200-230µs)"), un buffer di scrittura
  ("A spike of around 1.4-1.5ms appears every 64 requests") e il lavoro di sottofondo: con 10 ms di pausa fra le richieste
  "the percentage of the reads that have latencies lower than 300µs increases from 57% to 74% on SSD-L" [V]
  https://homes.luddy.indiana.edu/fchen25/publications/pdf/sigmetrics09.pdf
  Per noi: la pausa fra le letture (la tua "una alla volta") è il modo giusto per vedere il disco senza code [D].
- **Chen, Lee, Zhang, HPCA 2011**, "Essential Roles of Exploiting Internal Parallelism of Flash Memory based Solid State Drives in High-Speed Data Processing".
  Trattano l'SSD come "black box". Dalle valli di latenza ricavano la grandezza del chunk ("the detected chunk size is 8 sectors (4KB)"),
  dalle cadute di banda con due flussi il numero di canali ("we observe 10 domains, each of which actually corresponds to one channel"),
  e la politica di mappatura: dopo scritture casuali "the pattern (repeatedly appearing dips) disappears", quindi "LBA-based mapping is not used"
  e "the layout is only determined by writes." [V] https://homes.luddy.indiana.edu/fchen25/publications/pdf/hpca11.pdf
  Per noi: la posizione fisica la decide l'ordine di scrittura, non la posizione nel file. È il motivo della Prova D [D].
- **Kim, Seo, Jung, Kim, Huh, IEEE Trans. Computers 2012**, "Parameter-Aware I/O Management for Solid State Disks" (e MASCOTS 2009,
  "A methodology for extracting performance parameters in solid state disks"). Esistenza e dati bibliografici verificati su Crossref
  (DOI 10.1109/TC.2011.76) [V]; il testo non è raggiungibile. Il riassunto, da uno snippet, parla di "read/write unit size, erase unit size,
  NAND type, and read and write buffer sizes" [non verificato]. https://api.crossref.org/works/10.1109/TC.2011.76
- **Bouganim, Jónsson, Bonnet, CIDR 2009**, "uFLIP: Understanding Flash IO Patterns". Nove micro-benchmark su undici dispositivi.
  Lo stato conta: un Samsung nuovo faceva scritture casuali in 1 ms, "After randomly writing the entire 32GB of flash, however, the write
  performance decreased by almost an order of magnitude"; "Writing the whole flash device completely yields a well-defined state." [V]
  https://arxiv.org/pdf/0909.1780
  Per noi: ogni misura va fatta partendo da uno stato noto (cache vuota, spazio libero segnato) [D].
- **He, Kannan, Arpaci-Dusseau, EuroSys 2017**, "The Unwritten Contract of Solid State Drives". Cinque regole (richieste grandi o tante in volo,
  località, scritture allineate e sequenziali, raggruppare per tempo di morte, vita uniforme dei dati). Ma è uno studio con simulatore,
  non misure dal di fuori [V]. Cita l'SLC come cache: "is still often used to cache bursty writes because of its lower latency" [V]
  https://pages.cs.wisc.edu/~jhe/eurosys17-he.pdf
- **Kakaraparthy, Patel, Park, Kroth, PVLDB 2019**, "Optimizing Databases by Learning Hidden Parameters of Solid State Drives".
  "external measurements can be used to study these characteristics, and can sometimes reveal information regarding the internal parameters of the device."
  Ricavano chunk, stripe, pagina e "hot locations" (es. Samsung 960 EVO: chunk 64 KB, "offset group 32KB has 39% lower latency than offset group 0") [V]
  https://www.vldb.org/pvldb/vol13/p519-kakaraparthy.pdf
- **Li, Hao, Li, Lin, Emami, Gunawi, SYSTOR 2022**, "Fantastic SSD Internals and How to Learn and Use Them" (strumento Queenie, 21 modelli di SSD).
  Dieci proprietà dalla distribuzione delle latenze. Il tipo di cella si ricava dai gruppi di latenza di lettura:
  "SLC to represent 1-bit cell type (one category of latencies), MLC for 2-bit (two categories of latencies), and TLC for 3-bit (three categories of latencies)" [V]
  https://github.com/ucare-uchicago/Queenie e https://raw.githubusercontent.com/ucare-uchicago/Queenie/main/pseudo_code.pdf
  Per noi: i tuoi due gruppi (77-80 e 98-101 µs) sono lo stesso tipo di segnale; dentro il gruppo lento, con più letture, potresti vedere
  le tre sotto-latenze delle pagine TLC [D].
- **Park, Cho, Mutlu, Lee, Park, SIGMETRICS 2026**, "Experimental Study on System-Level Performance Impact of Read Disturbance in Modern SSDs".
  Quindici SSD NVMe. Prima di misurare leggono lontano dai dati recenti "to avoid reading data in the SLC buffer. This is because
  we observe that some tested SSDs retain either the oldest or the newest data in the SLC buffer even after long idle times. Reading such data
  yields artificially high performance" [V] https://arxiv.org/pdf/2608.14073
  Per noi: è esattamente il fenomeno che vedi. Alcuni dischi tengono in SLC "i più vecchi", altri "i più nuovi", anche dopo lunghe pause [D].
- **Yang, Cong, Gao, arXiv 2024**, "In-place Switch: Reprogramming based SLC Cache Design for Hybrid 3D SSDs". Su un SSD reale da 500 GB
  (dai riferimenti, un Crucial P1): "a bandwidth cliff is evident when a total 65GB data is written"; cinque flussi da 20 GB con
  "a free period of 10 minutes" fra l'uno e l'altro tengono "around 1090 MB/s throughout the process", quindi
  "the SLC cache is cleared during idle periods, allowing it to be reused for subsequent writes." [V] https://arxiv.org/html/2409.14360v3
  Per noi: 20 GB svuotati in 10 minuti su un disco consumer, cioè 120 GB/h, contro i tuoi 2,5-4 GB/h [D].
- **Juffinger, Rauscher, La Manna, Gruss, NDSS 2025**, "Secret Spilling Drive": dalle latenze deducono che "our set of SSDs with a DRAM cache
  do not use the DRAM to cache recently read data." [V] https://www.ndss-symposium.org/wp-content/uploads/2025-208-paper.pdf
  Per noi: la zona veloce non è una cache di lettura in DRAM [D].
- **Haas, Lee, Bonnet, Leis, PVLDB 2025**, "SSD-iq": amplificazione di scrittura dedotta da fuori, "WAFimplied = bw_seq / bw_measured" [V]
  https://www.vldb.org/pvldb/vol18/p4295-haas.pdf
- **Tom's Hardware**, metodo di recensione: "We use iometer to hammer the SSD with sequential writes for 15 minutes to measure both the size of the
  write cache and performance after the cache is saturated. We also monitor cache recovery via multiple idle rounds." Seagate FireCuda 120:
  cache "roughly 24GB-26GB, only 6GB of the cache recovers within a half-hour of idle rounds." [V]
  https://www.tomshardware.com/uk/reviews/seagate-firecuda-120-sata-ssd-review/2
  Sabrent Rocket 4 Plus 2 TB: 6,8 GB/s fino a circa 710 GB, poi 1135 MB/s; recupero "roughly 50GB-100GB per minute." [V]
  https://www.tomshardware.com/reviews/sabrent-rocket-4-plus-m2-nvme-ssd-review/3
  Per noi: il recupero varia da pochi GB in mezz'ora (FireCuda) a 100 GB al minuto (Rocket): la lentezza del tuo disco non è assurda [D].
- **The Overclocking Page, 23 agosto 2024**: SSD con pSLC disattivabile dallo strumento del produttore; cache dinamica "approximately 691GB";
  "In random scenarios like this, the pSLC Cache significantly improved read performance." [V]
  https://theoverclockingpage.com/2024/08/23/the-impact-of-slc-cache-in-performance-of-an-nvme-ssd-benchmarks-and-results/?lang=en
  Per noi: letture casuali più veloci dalla pSLC, confermato su un disco dove si poteva spegnerla [D].
- **Strumenti**: flashbench (Arnd Bergmann), `-a` fa "small reads across boundaries of various sizes" per trovare pagina e blocco di cancellazione [V]
  https://github.com/bradfa/flashbench ; Queenie e ssdcheck (sopra). Nessuno gira così com'è su macOS senza adattamenti [D].

## 4. Che cosa ne deduciamo per il nostro disco [D]

1. Le tue due famiglie di latenza (77-80 e 98-101 µs) sono il segnale che la letteratura usa per distinguere pagine di tipo diverso (Queenie)
   e per riconoscere dati rimasti nel "SLC buffer" (Park 2026). La spiegazione SLC contro TLC regge.
2. La differenza, circa 22 µs, è piccola rispetto ai 77 µs di base. Le schede tecniche danno 25 µs per una pagina SLC e 50-74 µs per una TLC
   (sezione 2.5): la cella da sola spiega una differenza di 30-50 µs, e il resto del tempo è controllore, cifratura AES-XTS e trasferimento.
   L'ordine di grandezza torna; il valore esatto dipende da come ANS2 legge le pagine TLC (per esempio con letture rapide a una sola soglia).
3. L'ordine di svuotamento "in ordine di scrittura" è coerente con la mappatura per ordine di scrittura (Chen 2011) e con i dischi che
   "tengono i più nuovi" (Park 2026): il tuo tiene i più nuovi e butta i più vecchi.
4. Il ritmo di 2,5-4 GB/h è molto più lento dei dischi consumer da recensione (decine di GB al minuto), ma non è senza precedenti
   (FireCuda: 6 GB in mezz'ora). Apple può aver scelto uno svuotamento lento per consumo e usura; le Prove A e B lo misurano.
5. La grandezza (circa 10 GB con 41 GB liberi) sta fra "2,5 % della capacità" (6 GB) e "10 %" (25 GB) di Oakley, ed è vicina al limite
   di un terzo dello spazio libero. Con 13 GB liberi la cache dinamica potrebbe scendere a 4 GB: la Prova C lo dice.
6. Dal Mac non puoi chiedere nulla al firmware: niente comandi, niente statistiche. Solo misure da fuori, come in questo piano.
