# GUIDA COMPLETA "ZERO-TO-HERO": SIM-TO-REAL HOPPER CON AUTOMATIC DOMAIN RANDOMIZATION (ADR)

**Versione:** 4.0 (Final "Deep Research" Edition)  
**Obiettivo:** Unica fonte di verità per l'implementazione, il training e la relazione finale del progetto.  
**Stato:** Definitivo per l'esame.

---

## INDICE

1.  [Il Concetto: Perché ADR?](#1-il-concetto-perché-adr)
2.  [Setup dell'Ambiente di Lavoro](#2-setup-dellambiente-di-lavoro)
3.  [Fase 1: Il Motore Fisico (CustomHopper)](#3-fase-1-il-motore-fisico-customhopper)
4.  [Fase 2: Il Cervello (ADR Callback)](#4-fase-2-il-cervello-adr-callback)
5.  [Fase 3: Il Training Loop (Main Script)](#5-fase-3-il-training-loop-main-script)
6.  [Esecuzione e Monitoraggio](#6-esecuzione-e-monitoraggio)
7.  [Analisi dei Risultati e Scrittura della Relazione](#7-analisi-dei-risultati-e-scrittura-della-relazione)

---

## 1. IL CONCETTO: PERCHÉ ADR?

### 1.1 Il problema: Il Paradosso del Sim-to-Real
Addestrare robot in simulazione (Sim) per farli agire nel mondo reale (Real) è difficile a causa del **Reality Gap**.
*   **Approccio Naive:** Addestrare su un modello fisso. **Risultato:** Fallimento catastrofico nel reale (Overfitting alla simulazione).
*   **Approccio Base (UDR - Uniform Domain Randomization):** Randomizzare parametri (massa, attrito) entro range fissi. 
    *   *Problema A (Range stretti):* Non copre la realtà.
    *   *Problema B (Range ampi):* Genera scenari impossibili ("ghiaccio puro"), l'agente non impara nulla (*Learned Helplessness*).

### 1.2 La soluzione: Automatic Domain Randomization (ADR)
L'ADR trasforma il training in un **curriculum automatico**. Immagina un cerchio di competenza che si allarga:
1.  Si parte da un ambiente deterministico (facile).
2.  Se l'agente è bravo (**Reward alto**), l'ambiente diventa più ostile (aumenta randomizzazione).
3.  Se l'agente fallisce (**Reward basso**), l'ambiente diventa più clemente (diminuisce randomizzazione).

**Obiettivo finale:** Ottenere una policy che sopravvive alla massima entropia (caos) possibile. Questa è la definizione operativa di "Robustezza".

---

## 2. SETUP DELL'AMBIENTE DI LAVORO

Segui questi passaggi per preparare una "Clean Room" per il progetto.

### 2.1 Struttura del File System
Resettiamo la struttura per garantire ordine mentale e pulizia del codice.

```text
/project_rl_adr
│
├── /env
│   ├── __init__.py          # (Vuoto o import base)
│   ├── custom_hopper.py     # IL CUORE FISICO (Modificato Fase 1)
│   └── assets/
│       └── hopper.xml       # Il modello MuJoCo standard
│
├── /callbacks               # NUOVA CARTELLA
│   ├── __init__.py
│   └── adr_callback.py      # IL CERVELLO (Creato Fase 2)
│
├── train.py                 # SCRIPT PRINCIPALE (Modificato Fase 3)
├── test_random_policy.py    # SCRIPT DI DEBUG (Opzionale)
├── requirements.txt         # LE DIPENDENZE
└── logs/                    # Dove Tensorboard scriverà la storia
```

### 2.2 Installazione Dipendenze
Crea un file `requirements.txt` con questo contenuto esatto per evitare conflitti di versione:

```text
gymnasium
mujoco
stable-baselines3[extra]>=2.0.0
tensorboard>=2.10.0
matplotlib
scipy
numpy
```

Esegui nel terminale:
```bash
pip install -r requirements.txt
```

### 2.3 Il File del Modello `hopper.xml`

Il file `env/assets/hopper.xml` è il modello fisico MuJoCo dell'Hopper. Questo file è **già incluso** nel repository. Se per qualche motivo dovesse mancare, puoi recuperarlo da:

**Opzione A:** Copia dal tuo ambiente Gymnasium installato:
```bash
cp $(python -c "import gymnasium; print(gymnasium.__path__[0])")/envs/mujoco/assets/hopper.xml env/assets/
```

**Opzione B:** Scarica dal repository MuJoCo:
```bash
curl -o env/assets/hopper.xml https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/hopper/hopper.xml
```

---

## 3. FASE 1: IL MOTORE FISICO (`custom_hopper.py`)

**Ruolo:** Estendere l'ambiente base per supportare l'ADR. Il file `env/custom_hopper.py` contiene già l'implementazione completa dell'Hopper (step, reward, health check, registrazione gym). Dobbiamo aggiungere:
1. Salvataggio dei valori nominali di **Damping** e **Friction**
2. Stato ADR (`adr_state`)
3. Metodi ADR (`update_adr`, `get_adr_info`)
4. Estensione di `sample_parameters` e `set_parameters`

**Il file base è già presente in `env/custom_hopper.py`.** Qui sotto sono documentate le **sole modifiche da implementare**.

---

### 3.1 Modifiche all'`__init__`

Dopo l'inizializzazione esistente, aggiungere il salvataggio dei valori nominali e lo stato ADR:

```python
# --- ADR EXTENSION: Initialization ---
# Salvataggio valori nominali per Damping e Friction (oltre alla massa già salvata)
self.original_damping = np.copy(self.model.dof_damping)
self.original_friction = np.copy(self.model.geom_friction)

# Stato ADR: delta di randomizzazione (iniziano a 0 = ambiente deterministico)
self.adr_state = {
    "mass_range": 0.0,      
    "damping_range": 0.0,   
    "friction_range": 0.0   
}

# Iperparametri ADR
self.adr_step_size = 0.05     # Step di incremento/decremento (5%)
self.min_friction_floor = 0.3 # Safe-guard: sotto 0.3 è impossibile camminare
```

---

### 3.2 Estensione di `sample_parameters`

Modificare il metodo esistente per restituire un **dizionario** con massa, damping e friction:

```python
def sample_parameters(self):
    """
    Genera un nuovo set di parametri fisici basati sulla difficoltà (adr_state) attuale.
    Estende la logica base randomizzando non solo massa ma anche attrito e damping.
    
    Returns:
        dict: Dizionario con chiavi "masses", "damping", "friction"
    """
    params = {}
    
    # A. MASSA
    m_delta = self.adr_state["mass_range"]
    m_scale = self.np_random.uniform(1.0 - m_delta, 1.0 + m_delta, size=self.original_masses.shape)
    # Nota: original_masses è [1:], quindi dobbiamo gestire l'offset
    params["masses"] = self.original_masses * m_scale

    # B. DAMPING
    d_delta = self.adr_state["damping_range"]
    d_scale = self.np_random.uniform(max(0.1, 1.0 - d_delta), 1.0 + d_delta, size=self.original_damping.shape)
    params["damping"] = self.original_damping * d_scale

    # C. FRICTION
    f_delta = self.adr_state["friction_range"]
    f_scale = self.np_random.uniform(max(self.min_friction_floor, 1.0 - f_delta), 1.0 + f_delta, size=(self.original_friction.shape[0], 1))
    new_friction = self.original_friction.copy()
    new_friction[:, 0] = self.original_friction[:, 0] * f_scale.flatten()
    params["friction"] = new_friction
    
    return params
```

---

### 3.3 Estensione di `set_parameters`

Modificare per accettare il dizionario e applicare tutti i parametri:

```python
def set_parameters(self, params):
    """Applica i parametri fisici al motore MuJoCo."""
    if isinstance(params, dict):
        # Versione ADR
        self.model.body_mass[1:] = params["masses"]
        self.model.dof_damping[:] = params["damping"]
        self.model.geom_friction[:] = params["friction"]
    else:
        # Retrocompatibilità con versione base (lista di masse)
        self.model.body_mass[1:] = params
```

---

### 3.4 Nuovo metodo `update_adr`

Aggiungere questo metodo che sarà chiamato dalla Callback:

```python
def update_adr(self, mean_reward: float, low_th: float, high_th: float) -> Tuple[str, Dict]:
    """
    Regola i 'range' in base alla performance. È il feedback loop dell'ADR.
    
    Args:
        mean_reward: Reward medio recente
        low_th: Soglia inferiore per contrazione
        high_th: Soglia superiore per espansione
        
    Returns:
        tuple: (status_string, adr_state_dict)
    """
    status = "stable"
    
    if mean_reward >= high_th:
        # ESPANSIONE
        self.adr_state["mass_range"] += self.adr_step_size
        self.adr_state["damping_range"] += self.adr_step_size
        self.adr_state["friction_range"] += self.adr_step_size
        status = "expanded"
        
    elif mean_reward < low_th:
        # CONTRAZIONE
        self.adr_state["mass_range"] = max(0.0, self.adr_state["mass_range"] - self.adr_step_size)
        self.adr_state["damping_range"] = max(0.0, self.adr_state["damping_range"] - self.adr_step_size)
        self.adr_state["friction_range"] = max(0.0, self.adr_state["friction_range"] - self.adr_step_size)
        status = "contracted"
    
    return status, self.adr_state

def get_adr_info(self) -> Dict:
    """Ritorna lo stato corrente dell'ADR per il logging."""
    return self.adr_state.copy()
```

---

### 3.5 Modifica di `reset_model`

Modificare per chiamare la pipeline ADR:

```python
def reset_model(self):
    # ... codice esistente per qpos/qvel ...
    
    # Applica ADR (sostituisce la logica UDR)
    params = self.sample_parameters()
    self.set_parameters(params)
    
    # ... resto del metodo ...
```

```

---

## 4. FASE 2: IL CERVELLO (`adr_callback.py`)

**Ruolo:** Monitorare costantemente l'agente e inviare i segnali di Espansione/Contrazione all'ambiente. Usa le API di `Stable Baselines3`.

**Azioni:** Crea il file `callbacks/adr_callback.py` con questo codice.

```python
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class ADRCallback(BaseCallback):
    """
    Monitora il reward medio e adatta la difficoltà dell'ambiente CustomHopper.
    """
    def __init__(self, check_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        
        # SOGLIE DI PERFORMANCE (Performance Thresholds)
        # Hopper-v4 "risolto" è ~3000.
        # Threshold High (2000): Se superato, l'ambiente è "troppo facile" -> Espandi.
        # Threshold Low (1000): Se non raggiunto, l'ambiente è "troppo difficile" -> Contrai.
        self.threshold_high = 2000 
        self.threshold_low = 1000   

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            
            # Recupera il reward buffer dal Monitor wrapper
            ep_info_buffer = self.model.ep_info_buffer
            if len(ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in ep_info_buffer])
                
                # Accedi all'ambiente "nudo" (bypassando i wrapper SB3)
                env_unwrapped = self.training_env.envs[0].unwrapped
                
                if hasattr(env_unwrapped, 'update_adr'):
                    # Chiama l'update nell'environment
                    status, adr_stats = env_unwrapped.update_adr(
                        mean_reward, self.threshold_low, self.threshold_high
                    )
                    
                    # LOGGING CRUCIALE PER IL REPORT
                    # Questi tag appariranno in Tensorboard
                    self.logger.record("adr/mean_reward", mean_reward)
                    self.logger.record("adr/mass_range_delta", adr_stats["mass_range"])
                    self.logger.record("adr/friction_range_delta", adr_stats["friction_range"])
                    
                    if self.verbose > 0 and status != "stable":
                        print(f"[ADR] Step {self.num_timesteps}: {status.upper()} bounds. "
                              f"R={mean_reward:.0f} | MassΔ={adr_stats['mass_range']:.2f}")
                else:
                    pass # Environment non compatibile con ADR
        return True
```

---

## 5. FASE 3: IL TRAINING LOOP (`train.py`)

**Ruolo:** Orchestrare il training con ADR. Il file `train.py` contiene già la logica base del training (setup environment, PPO, evaluate). Dobbiamo aggiungere:
1. Wrapper `Monitor` per permettere alla callback di leggere le statistiche
2. Import e istanza della `ADRCallback`
3. Logging su Tensorboard per visualizzare l'evoluzione ADR

**Il file base è già presente in `train.py`.** Qui sotto sono documentate le **modifiche chiave**.

---

### 5.1 Import aggiuntivi

Aggiungere in cima al file:

```python
from stable_baselines3.common.monitor import Monitor
from callbacks.adr_callback import ADRCallback
```

---

### 5.2 Wrapper Monitor sull'ambiente

Modificare la creazione dell'ambiente di training:

```python
# PRIMA (senza ADR):
env_source = gym.make('CustomHopper-source-v0', udr=False)

# DOPO (con ADR):
env_source = Monitor(gym.make('CustomHopper-source-v0', udr=False))
```

Il `Monitor` wrapper è **essenziale** perché la callback ADR usa `model.ep_info_buffer` per leggere i reward.

---

### 5.3 Istanza della Callback

Prima di `model.learn()`, creare la callback:

```python
# check_freq=2048 allinea l'update ADR con l'update di PPO
adr_callback = ADRCallback(check_freq=2048)
```

---

### 5.4 Training con Callback e Tensorboard

Modificare la chiamata a `learn()`:

```python
# PRIMA:
model = PPO('MlpPolicy', env_source, verbose=0)
model.learn(total_timesteps=200000, progress_bar=True)

# DOPO:
model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=300000, callback=adr_callback, progress_bar=True)
```

Note:
- `tensorboard_log` abilita il salvataggio dei log per Tensorboard
- 300k timesteps danno tempo all'ADR di espandersi
- `callback=adr_callback` aggancia la logica di espansione/contrazione

---

### 5.5 File di Debug: `test_random_policy.py`

Il file `test_random_policy.py` serve per verificare che l'ambiente funzioni correttamente prima del training. Stampare le informazioni ADR per debug:

```python
# Dopo env.reset()
if hasattr(env.unwrapped, 'adr_state'):
    print('ADR State:', env.unwrapped.adr_state)
```


---

## 6. ESECUZIONE E MONITORAGGIO

1.  **Lancia il Training:**
    ```bash
    python train.py
    ```
    
2.  **Monitora l'Evoluzione (Live):**
    In un nuovo terminale, esegui:
    ```bash
    tensorboard --logdir ./logs/
    ```
    Apri il browser (solitamente `http://localhost:6006`).

3.  **Cosa guardare (Il "Segnale di Vita"):**
    *   Vai nella tab **SCALARS**.
    *   Cerca il grafico `adr/mass_range_delta` e `adr/friction_range_delta`.
    *   **Comportamento Atteso:** Dovresti vedere una curva "a gradini" che sale. Questo significa che l'agente sta diventando sempre più forte e l'ambiente sta alzando l'asticella. Se la curva sale e poi scende leggermente, significa che l'ADR sta funzionando correggendo un "eccesso di fiducia".

---

## 7. ANALISI DEI RISULTATI E SCRITTURA DELLA RELAZIONE

Per completare il progetto (Punto 4), usa questi dati nella tua relazione. Cita le fonti teoriche per dare spessore accademico.

### 7.1 Interpreta i Grafici
*   **Curva di Reward vs ADR Range:** Mostra come il reward rimane stabile (o recupera velocemente) anche mentre il `mass_range` aumenta. Questo dimostra **adattamento**.
*   **Il Delta Finale:** Se il `mass_range` finale è, per esempio, `0.40` (40%), significa che la tua policy può gestire un robot che pesa il 40% in più o in meno del previsto. Questo è un risultato quantitativo di robustezza enorme.

### 7.2 Confronto Sim-to-Real
Confronta il grafico `adr_robustness_chart.png` con i risultati che avevi ottenuto prima (senza ADR).
*   **Senza ADR:** Il gap tra Source e Target era probabilmente ampio.
*   **Con ADR:** Il gap dovrebbe essersi ridotto drasticamente (o addirittura il Target potrebbe andare meglio del Source se l'ADR ha generato ambienti più difficili del Target stesso).

### 7.3 Bibliografia Essenziale per la Relazione
Copia/Incolla e rielabora questi riferimenti nel tuo report finale per giustificare le scelte tecniche:

> *   **Su ADR:** OpenAI et al., *"Solving Rubik's Cube with a Robot Hand"*, 2019. (Fondamentale per spiegare l'algoritmo).
> *   **Sulla Latenza:** Sandha et al., *"Sim2Real Transfer... with Stochastic Delays"*, 2021. (Per spiegare perché non abbiamo scelto la latenza: l'ADR è più generale).
> *   **Su Attrito e Damping:** Tan et al., *"Sim-to-Real: Learning Agile Locomotion"*, RSS 2018. (Giustifica perché abbiamo randomizzato `dof_damping` e `geom_friction`).
> *   **Sui pericoli dell'UDR:** Mehta et al., *"Active Domain Randomization"*, 2020. (Per spiegare perché i range fissi sono pericolosi).

---
**Congratulazioni.** Hai trasformato un semplice esercizio di RL in un sistema di training adativo state-of-the-art. Sei pronto per la consegna.