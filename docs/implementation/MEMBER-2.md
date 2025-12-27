# GUIDA COMPLETA MEMBRO 2: INTEGRAZIONE & DEVOPS

**Versione:** 1.0  
**Ruolo:** Integrazione, DevOps, Training e Analisi  
**Stato:** Guida definitiva per l'implementazione

---

## Introduzione

Questa guida è dedicata esclusivamente al **Membro 2** del team, responsabile dell'integrazione dei componenti, della configurazione dell'ambiente di sviluppo, dell'esecuzione del training e dell'analisi finale dei risultati. Il lavoro di questo ruolo è fondamentale perché funge da collante tra tutti i componenti sviluppati dal Membro 1 e trasforma il codice in risultati concreti e misurabili.

Prima di iniziare qualsiasi attività pratica, è essenziale comprendere il contesto teorico del progetto. L'Automatic Domain Randomization (ADR) è una tecnica avanzata che permette di addestrare agenti di Reinforcement Learning in modo che siano robusti alle variazioni ambientali. A differenza della randomizzazione uniforme tradizionale, l'ADR adatta dinamicamente la difficoltà dell'ambiente in base alle performance dell'agente: se l'agente performa bene, l'ambiente diventa più difficile; se performa male, l'ambiente si semplifica. Questo crea un curriculum di apprendimento automatico che porta a policy più robuste.

Il tuo compito principale sarà quello di preparare l'infrastruttura necessaria, integrare i componenti ADR nel loop di training, eseguire gli esperimenti e raccogliere i dati necessari per la relazione finale.

---

## Parte 1: Configurazione dell'Ambiente di Lavoro

### 1.1 Panoramica della Struttura del Progetto

La prima cosa da fare è comprendere e verificare la struttura del progetto. Questa organizzazione non è casuale: ogni cartella ha uno scopo preciso che facilita la separazione delle responsabilità e la manutenibilità del codice.

La cartella principale `project_rl_adr` contiene tutto il necessario per il progetto. Al suo interno troverai la cartella `env` che ospita l'ambiente personalizzato dell'Hopper, sviluppato dal Membro 1. Questa cartella contiene il file `custom_hopper.py` che è il cuore fisico della simulazione, più una sottocartella `assets` con il modello MuJoCo `hopper.xml`.

Parallelamente esiste la cartella `callbacks` che contiene la logica ADR sotto forma di callback per Stable Baselines 3. Il file principale qui è `adr_callback.py`, anch'esso sviluppato dal Membro 1.

La cartella `notebooks` contiene i notebook Jupyter del progetto. In particolare, la sottocartella `notebooks/verification` ospita due notebook di verifica: `verify-member-1.ipynb` e `verify-member-2.ipynb`, che permettono di testare che le implementazioni dei rispettivi membri siano corrette.

A livello root troverai i file principali: `train.py` è lo script di training che dovrai modificare, `test_random_policy.py` serve per il debug, `requirements.txt` elenca le dipendenze, e la cartella `logs` conterrà tutti i dati generati durante il training.

La cartella `docs/implementation` contiene la documentazione del progetto: `IMPLEMENTATION.md` (guida generale), `MEMBER-2.md` (questa guida dettagliata), e `REPORT.md` (report di ricerca).

### 1.2 Creazione della Struttura delle Cartelle

Prima di procedere con qualsiasi altra operazione, è necessario assicurarsi che la struttura delle cartelle esista. Apri il terminale nella directory principale del progetto ed esegui il seguente comando per creare tutte le cartelle necessarie in un colpo solo:

```bash
mkdir -p env/assets callbacks logs
```

Questo comando crea la cartella `env` con la sua sottocartella `assets`, la cartella `callbacks` per le callback SB3, e la cartella `logs` dove Tensorboard salverà i dati di training. L'opzione `-p` garantisce che non vengano generati errori se alcune cartelle esistono già.

Successivamente, è necessario creare i file `__init__.py` che trasformano le cartelle in moduli Python importabili:

```bash
touch env/__init__.py callbacks/__init__.py
```

Questi file possono rimanere vuoti, ma la loro presenza è fondamentale affinché Python riconosca le cartelle come package e permetta gli import.

### 1.3 Installazione delle Dipendenze

Il progetto richiede diverse librerie Python che devono essere installate nella versione corretta per evitare conflitti. Crea un file chiamato `requirements.txt` nella root del progetto con il seguente contenuto:

```text
gymnasium
mujoco
stable-baselines3[extra]>=2.0.0
tensorboard>=2.10.0
matplotlib
scipy
numpy
```

Analizziamo brevemente ogni dipendenza. La libreria `gymnasium` è l'evoluzione moderna di OpenAI Gym e fornisce l'interfaccia standard per gli ambienti di Reinforcement Learning. La libreria `mujoco` è il motore fisico che simula la dinamica del robot Hopper. Stable Baselines 3, installata con l'opzione `[extra]`, include l'implementazione di PPO e altri algoritmi RL insieme a utility aggiuntive come le progress bar. Tensorboard è essenziale per visualizzare i grafici durante e dopo il training. Matplotlib e scipy servono per eventuali analisi aggiuntive e generazione di grafici personalizzati.

Per installare tutte le dipendenze, esegui nel terminale:

```bash
pip install -r requirements.txt
```

L'installazione potrebbe richiedere alcuni minuti, specialmente per MuJoCo che deve compilare alcuni componenti nativi.

### 1.4 Verifica del Modello MuJoCo

Il file `hopper.xml` è il modello 3D del robot Hopper in formato MuJoCo. Questo file definisce la geometria del robot, le sue articolazioni, le masse dei vari componenti e altre proprietà fisiche. Il file dovrebbe essere già presente nel repository nella cartella `env/assets/`.

Se per qualche motivo il file non fosse presente, puoi recuperarlo in due modi. Il primo metodo consiste nel copiarlo direttamente dall'installazione di Gymnasium:

```bash
cp $(python -c "import gymnasium; print(gymnasium.__path__[0])")/envs/mujoco/assets/hopper.xml env/assets/
```

Questo comando usa Python per trovare il percorso di installazione di Gymnasium e poi copia il file nella posizione corretta.

In alternativa, puoi scaricare il file direttamente dal repository ufficiale di MuJoCo:

```bash
curl -o env/assets/hopper.xml https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/hopper/hopper.xml
```

### 1.5 Verifica dell'Ambiente

Prima di procedere oltre, è buona pratica verificare che tutto sia configurato correttamente. Puoi farlo creando un semplice script di test o eseguendo Python interattivamente. Apri un terminale Python e prova a importare le librerie principali:

```python
import gymnasium
import mujoco
import stable_baselines3
print("Tutte le dipendenze sono installate correttamente!")
```

Se non ricevi errori, l'ambiente è pronto per il passo successivo.

---

## Parte 2: Comprensione dei Componenti del Membro 1

Prima di modificare il file di training, è importante comprendere cosa ha sviluppato il Membro 1 e come i suoi componenti si integrano con il tuo lavoro.

### 2.1 L'Ambiente CustomHopper

Il file `env/custom_hopper.py` estende l'ambiente Hopper standard di Gymnasium con funzionalità ADR. Il Membro 1 ha aggiunto tre elementi fondamentali a questa classe.

Il primo elemento è lo stato ADR, un dizionario che tiene traccia dei range di randomizzazione correnti per massa, damping e friction. Inizialmente tutti questi valori sono a zero, il che significa che l'ambiente parte in modo deterministico, senza alcuna randomizzazione.

Il secondo elemento è il metodo `sample_parameters()`, che genera un nuovo set di parametri fisici basandosi sullo stato ADR corrente. Più alto è il valore nel dizionario ADR, più ampia sarà la randomizzazione applicata.

Il terzo elemento è il metodo `update_adr()`, che è il cuore del sistema adattivo. Questo metodo riceve il reward medio recente e le soglie di performance, e decide se espandere o contrarre i range di randomizzazione.

### 2.2 La Callback ADR

Il file `callbacks/adr_callback.py` contiene la classe `ADRCallback` che estende `BaseCallback` di Stable Baselines 3. Questa callback viene eseguita periodicamente durante il training e ha il compito di monitorare le performance dell'agente e comunicare con l'ambiente per aggiornare lo stato ADR.

La callback legge il buffer dei reward episodici tramite `model.ep_info_buffer`, calcola il reward medio, e chiama il metodo `update_adr()` dell'ambiente. Inoltre, logga tutte le metriche rilevanti su Tensorboard per permetterti di visualizzare l'evoluzione del training.

---

## Parte 3: Modifica del File di Training

Questa è la parte centrale del tuo lavoro. Dovrai modificare il file `train.py` esistente per integrare il sistema ADR. Le modifiche sono concentrate in quattro aree principali.

### 3.1 Aggiunta degli Import Necessari

All'inizio del file `train.py`, dopo gli import esistenti, devi aggiungere due nuove importazioni. La prima importa il wrapper Monitor da Stable Baselines 3:

```python
from stable_baselines3.common.monitor import Monitor
```

Il wrapper Monitor è fondamentale perché avvolge l'ambiente e registra automaticamente le statistiche degli episodi come reward, lunghezza e tempo. Senza questo wrapper, la callback ADR non avrebbe accesso ai dati necessari per valutare le performance dell'agente.

La seconda importazione carica la callback ADR sviluppata dal Membro 1:

```python
from callbacks.adr_callback import ADRCallback
```

Questa riga assume che la cartella `callbacks` sia un package Python valido, il che richiede la presenza del file `__init__.py` che hai creato in precedenza.

### 3.2 Wrapping dell'Ambiente con Monitor

Nel punto del codice dove viene creato l'ambiente di training, devi modificare la creazione per includere il wrapper Monitor. La versione originale del codice probabilmente appare così:

```python
env_source = gym.make('CustomHopper-source-v0', udr=False)
```

Questa riga crea semplicemente l'ambiente senza alcun monitoraggio. Devi modificarla in questo modo:

```python
env_source = Monitor(gym.make('CustomHopper-source-v0', udr=False))
```

Nota come l'ambiente viene prima creato con `gym.make()` e poi immediatamente avvolto con `Monitor()`. Questo pattern di wrapping è comune in Stable Baselines 3 e permette di aggiungere funzionalità all'ambiente senza modificare il codice dell'ambiente stesso.

Il parametro `udr=False` indica che non vogliamo usare la Uniform Domain Randomization tradizionale, perché useremo invece l'ADR che è più sofisticata e adattiva.

### 3.3 Creazione dell'Istanza della Callback

Prima della chiamata a `model.learn()`, devi creare un'istanza della callback ADR. Aggiungi questa riga nel punto appropriato:

```python
adr_callback = ADRCallback(check_freq=2048)
```

Il parametro `check_freq` specifica ogni quanti step la callback deve eseguire il suo controllo. Il valore 2048 è stato scelto deliberatamente perché corrisponde esattamente alla dimensione del batch di default di PPO. Questo significa che l'ADR valuta e potenzialmente aggiorna la difficoltà dell'ambiente una volta per ogni update della policy, garantendo una sincronia perfetta tra apprendimento e adattamento ambientale.

Se usassi un valore troppo piccolo, l'ADR reagirebbe troppo velocemente a fluttuazioni casuali nei reward. Se usassi un valore troppo grande, l'ADR sarebbe troppo lenta nell'adattarsi ai progressi dell'agente.

### 3.4 Configurazione del Modello PPO

La creazione del modello PPO deve essere modificata per abilitare il logging su Tensorboard. La versione originale potrebbe apparire così:

```python
model = PPO('MlpPolicy', env_source, verbose=0)
```

Devi modificarla per includere il path dei log:

```python
model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log="./logs/")
```

Il parametro `verbose=1` abilita l'output testuale durante il training, che ti permette di monitorare il progresso anche senza Tensorboard. Il parametro `tensorboard_log` specifica la directory dove salvare i log di Tensorboard. Ogni run di training creerà una sottocartella con timestamp in questa directory.

### 3.5 Esecuzione del Training con Callback

Infine, la chiamata a `learn()` deve essere modificata per includere la callback e possibilmente aumentare il numero di timestep. La versione originale era probabilmente:

```python
model.learn(total_timesteps=200000, progress_bar=True)
```

Devi modificarla così:

```python
model.learn(total_timesteps=5000000, callback=adr_callback, progress_bar=True)
```

Il numero di timestep è stato aumentato a 5.000.000 per dare all'ADR tempo sufficiente di espandere i range di randomizzazione fino a valori elevati (±70%). Il parametro `callback=adr_callback` aggancia la callback che abbiamo creato, facendo sì che venga chiamata ad ogni step del training.

### 3.6 Esempio Completo delle Modifiche

Per chiarezza, ecco come dovrebbe apparire la sezione rilevante del file `train.py` dopo tutte le modifiche:

```python
# Import aggiuntivi (in cima al file)
from stable_baselines3.common.monitor import Monitor
from callbacks.adr_callback import ADRCallback

# ... altro codice esistente ...

# Creazione ambiente con Monitor
env_source = Monitor(gym.make('CustomHopper-source-v0', udr=False))

# Creazione callback ADR
adr_callback = ADRCallback(check_freq=2048)

# Creazione modello con logging Tensorboard
model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log="./logs/")

# Training con callback
model.learn(total_timesteps=300000, callback=adr_callback, progress_bar=True)
```

---

## Parte 4: Esecuzione del Training

Una volta completate tutte le modifiche, sei pronto per lanciare il training. Questa fase richiede attenzione perché dovrai monitorare il processo e verificare che tutto funzioni correttamente.

### 4.1 Lancio del Training

Apri un terminale nella directory principale del progetto e lancia lo script di training:

```bash
python train.py
```

Vedrai apparire una progress bar che mostra l'avanzamento del training. Se il `verbose` è impostato a 1, vedrai anche messaggi periodici dalla callback ADR che indicano quando i bounds vengono espansi o contratti.

Il training con 300000 timestep richiede tipicamente tra 15 e 45 minuti, a seconda dell'hardware disponibile. Durante questo tempo, puoi aprire un secondo terminale per monitorare i grafici con Tensorboard.

### 4.2 Monitoraggio con Tensorboard

In un nuovo terminale, sempre nella directory del progetto, lancia Tensorboard:

```bash
tensorboard --logdir ./logs/
```

Tensorboard avvierà un server web locale e ti mostrerà l'URL a cui accedere, tipicamente `http://localhost:6006`. Apri questo URL nel tuo browser.

Nella tab SCALARS troverai diversi grafici cruciali. Il grafico `adr/mean_reward` mostra l'andamento del reward medio nel tempo. Il grafico `adr/mass_range_delta` mostra come il range di randomizzazione della massa evolve durante il training. Analogamente, `adr/friction_range_delta` mostra l'evoluzione del range di friction.

### 4.3 Interpretazione dei Grafici Durante il Training

Il comportamento atteso è una curva "a gradini" per i grafici dei range ADR. Inizialmente i valori saranno tutti a zero. Man mano che l'agente impara e supera la soglia alta di performance (1200 di reward), vedrai i valori salire di 0.05 alla volta. Questo indica che l'ambiente sta diventando progressivamente più difficile.

Se l'agente fatica troppo e scende sotto la soglia bassa (600 di reward), i valori scenderanno, indicando che l'ambiente si sta semplificando per permettere all'agente di recuperare.

Un segnale molto positivo è vedere i range ADR che salgono costantemente con solo occasionali piccole discese. Questo indica che l'agente sta imparando in modo robusto e sta espandendo continuamente la sua zona di competenza.

### 4.4 Gestione di Problemi Comuni

Se il training si blocca o produce errori, verifica prima di tutto che l'ambiente sia stato registrato correttamente in Gymnasium. Il file `env/__init__.py` dovrebbe contenere la registrazione dell'ambiente custom.

Se la callback non produce alcun log, verifica che il wrapper Monitor sia applicato correttamente all'ambiente. Senza Monitor, il buffer `ep_info_buffer` sarà sempre vuoto e la callback non avrà dati su cui lavorare.

Se i grafici Tensorboard non si aggiornano, prova a ricaricare la pagina del browser o a riavviare Tensorboard. A volte la cache del browser può causare problemi.

---

## Parte 5: Analisi dei Risultati

Una volta completato il training, è il momento di analizzare i risultati e preparare i dati per la relazione finale.

### 5.1 Esportazione dei Grafici

Da Tensorboard puoi esportare i grafici in vari formati. Clicca sull'icona di download in alto a destra di ogni grafico per salvarlo come immagine PNG o come dati CSV.

I grafici essenziali da esportare sono tre: il grafico del reward nel tempo, il grafico dell'evoluzione dei range ADR (massa e/o friction), e il confronto tra performance su source e target environment se disponibile.

### 5.2 Interpretazione dei Risultati Finali

Il valore finale dei range ADR è un indicatore quantitativo della robustezza della policy. Se ad esempio il `mass_range` finale è 0.40, significa che la policy addestrata può gestire un robot la cui massa varia del 40% rispetto al valore nominale in entrambe le direzioni. Questo è un risultato eccezionale che indica un'altissima robustezza.

Confronta i risultati ADR con quelli ottenuti senza ADR o con UDR standard. La differenza nel gap tra performance su source e target environment dovrebbe essere significativamente ridotta con ADR.

### 5.3 Materiale per la Relazione

Per la relazione finale, prepara i seguenti elementi. Includi screenshot dei grafici Tensorboard che mostrano l'evoluzione del training. Documenta il valore finale dei range ADR come misura quantitativa di robustezza. Prepara una tabella comparativa che mostri le performance su source vs target con e senza ADR.

Nella relazione, cita le seguenti fonti accademiche per dare spessore teorico alle scelte tecniche. Il paper di OpenAI "Solving Rubik's Cube with a Robot Hand" del 2019 è fondamentale per spiegare l'algoritmo ADR. Il paper di Tan et al. "Sim-to-Real: Learning Agile Locomotion" del 2018 giustifica la scelta di randomizzare damping e friction. Il paper di Mehta et al. "Active Domain Randomization" del 2020 spiega i pericoli dell'UDR con range fissi.

---

## Parte 6: Checklist Finale

Prima di considerare il lavoro completato, verifica che tutti i seguenti punti siano stati soddisfatti.

L'ambiente Python è correttamente configurato con tutte le dipendenze installate. La struttura delle cartelle è completa con tutti i file `__init__.py` necessari. Il file `train.py` è stato modificato con tutti e quattro i cambiamenti richiesti: import di Monitor, import della callback, wrapping dell'ambiente, creazione della callback e modifica della chiamata a learn().

Il training è stato eseguito con successo per almeno 300000 timestep. I log di Tensorboard sono stati generati nella cartella `logs/`. I grafici essenziali sono stati esportati per la relazione. L'analisi dei risultati è stata completata con interpretazione quantitativa dei range ADR finali.

Se tutti questi punti sono verificati, hai completato con successo il tuo ruolo nel progetto.

---

## Appendice: Troubleshooting

### Il training non parte

Se ricevi un errore all'avvio del training, verifica che l'ambiente CustomHopper sia stato registrato correttamente. Controlla che il file `env/__init__.py` contenga il codice di registrazione e che non ci siano errori di sintassi nei file sviluppati dal Membro 1.

### La callback non logga nulla

Se non vedi output dalla callback ADR, il problema più probabile è che l'ambiente non sia wrappato con Monitor. Verifica che la riga di creazione dell'ambiente includa `Monitor()`.

### Tensorboard non mostra dati

Se Tensorboard è vuoto o non mostra i grafici ADR, verifica che il parametro `tensorboard_log` sia stato specificato nella creazione del modello PPO. Controlla anche che la cartella `logs/` non sia vuota.

### Il training è molto lento

La velocità del training dipende fortemente dall'hardware. Su una CPU moderna, aspettati circa 2000-5000 step al secondo. Se hai una GPU compatibile, puoi provare a passare `device='cuda'` nella creazione del modello PPO, anche se per MLP policies il vantaggio è limitato.

---

**Fine della guida per il Membro 2.**
