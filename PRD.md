# Consciousness Observatory
## Product Requirements Document

---

## Vision Statement

> *"A personal scientific instrument for mapping the landscape of my own consciousness — discovering which environmental, behavioral, and physiological variables are the highest-leverage attractors toward states of enlightenment, loving awareness, generativity, and joy."*

This is not a meditation app. This is a **dynamical systems research platform** for a single subject: you.

---

## Core Philosophy

### The Dynamical Systems Perspective

Your consciousness isn't a static thing to be measured — it's a **dynamical system** that flows through state space. Some regions of this space are attractors (states you naturally fall into), and various forces (sleep, relationships, activities, thoughts) push you between them.

The observatory aims to:
1. **Map the territory** — What are your natural brain/body states?
2. **Find the currents** — What variables push you toward or away from desired states?
3. **Identify leverage points** — Where small interventions create large shifts
4. **Close the loop** — Real-time awareness and eventually adaptive guidance

### On Labeling: Data-First Discovery

Rather than forcing predefined categories onto your experience, we:
1. Let unsupervised learning discover your brain's natural state clusters
2. Visualize these states and their transitions
3. You provide **meaning** — naming clusters, marking moments of significance
4. Supervised models then learn to predict your meaningful states from raw signals

---

## Data Sources

### Primary Streams

| Source | Data Type | Temporal Resolution | Key Features |
|--------|-----------|---------------------|--------------|
| **Muse EEG** | Brainwaves | ~256 Hz (live stream) | Alpha, beta, theta, gamma power; asymmetry; coherence |
| **Garmin** | Biometrics | Varies (1min - daily) | HRV, resting HR, sleep stages, stress score, body battery, steps, SpO2 |
| **Calendar** | Context | Event-level | Meeting types, social time, work blocks, transitions |
| **Journal** | Text | Entry-level | Semantic content, sentiment, themes, self-reflection |

### Derived Signals (Computed)

| Signal | Derived From | Meaning |
|--------|--------------|---------|
| **Frontal Alpha Asymmetry** | EEG F7/F8 | Approach vs withdrawal motivation |
| **Theta/Beta Ratio** | EEG | Attention regulation, mind-wandering |
| **Alpha Coherence** | EEG bilateral | Interhemispheric integration |
| **HRV Complexity** | Garmin RR intervals | Autonomic flexibility, resilience |
| **Sleep Architecture Score** | Garmin sleep | Quality of restoration |
| **Context Embedding** | Calendar + Journal | Semantic vector of "what's happening in life" |
| **Transition Velocity** | All streams | How quickly states are changing |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CONSCIOUSNESS OBSERVATORY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Muse EEG  │  │   Garmin    │  │  Calendar   │  │   Journal   │        │
│  │  (live BT)  │  │    API      │  │    API      │  │   (text)    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         ▼                ▼                ▼                ▼               │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    INGESTION LAYER                              │       │
│  │  • Stream processing (EEG)    • API polling (Garmin)            │       │
│  │  • Artifact removal           • Time alignment                  │       │
│  │  • Feature extraction         • Missing data handling           │       │
│  └─────────────────────────────────┬───────────────────────────────┘       │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    FEATURE STORE                                │       │
│  │  • Time-indexed multimodal features                             │       │
│  │  • Multiple temporal resolutions (1s, 1min, 1hr, 1day)          │       │
│  │  • SQLite + Parquet for efficient queries                       │       │
│  └─────────────────────────────────┬───────────────────────────────┘       │
│                                    │                                       │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         ▼                          ▼                          ▼            │
│  ┌─────────────┐          ┌─────────────────┐         ┌─────────────┐      │
│  │   STATE     │          │    LEVER        │         │  LABELING   │      │
│  │   MAPPER    │          │    FINDER       │         │  INTERFACE  │      │
│  │             │          │                 │         │             │      │
│  │ • UMAP/PCA  │          │ • Attention     │         │ • Moment    │      │
│  │ • Clustering│          │   weights       │         │   marking   │      │
│  │ • Trajectory│          │ • Granger       │         │ • State     │      │
│  │   analysis  │          │   causality     │         │   naming    │      │
│  │ • Attractor │          │ • SHAP values   │         │ • Rating    │      │
│  │   detection │          │ • Intervention  │         │   scales    │      │
│  │             │          │   simulation    │         │             │      │
│  └──────┬──────┘          └────────┬────────┘         └──────┬──────┘      │
│         │                          │                         │             │
│         └──────────────────────────┼─────────────────────────┘             │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    OBSERVATORY DASHBOARD                        │       │
│  │                                                                 │       │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │       │
│  │  │   State Space   │  │  Lever Rankings │  │    Timeline     │ │       │
│  │  │   Visualizer    │  │   & Influences  │  │    Explorer     │ │       │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘ │       │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │       │
│  │  │  Live Session   │  │   Transition    │  │    Insights     │ │       │
│  │  │    Monitor      │  │     Graphs      │  │    & Reports    │ │       │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘ │       │
│  │                                                                 │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Specifications

### Phase 1: Foundation (MVP) — *"See Yourself"*
**Goal**: Get data flowing and visualize your state space
**Timeline**: 3-4 hours to build core

#### 1.1 Data Ingestion
- [ ] **Muse EEG connector**: Connect via Muse SDK / OSC protocol, stream raw EEG
- [ ] **EEG preprocessing**: Band-pass filter, artifact rejection (blinks, jaw clenches)
- [ ] **EEG feature extraction**: 
  - Band powers (delta, theta, alpha, beta, gamma) per channel
  - Frontal alpha asymmetry
  - Theta/beta ratio
- [ ] **Garmin import**: Manual CSV import initially → API later
- [ ] **Calendar import**: ICS file import or Google Calendar API
- [ ] **Journal import**: Markdown files or simple text input

#### 1.2 State Space Visualization
- [ ] **Dimensionality reduction**: UMAP projection of multimodal features to 2D/3D
- [ ] **Interactive scatter plot**: Each point is a moment in time, colored by:
  - Time of day
  - User labels (once added)
  - Detected cluster
- [ ] **Trajectory visualization**: See how you move through state space over a session
- [ ] **Cluster discovery**: Automatic clustering (HDBSCAN) to find natural states

#### 1.3 Basic Dashboard
- [ ] **Live view**: Real-time EEG features during a session
- [ ] **Historical view**: Browse past sessions on state space
- [ ] **Simple statistics**: Time spent in each cluster, transition frequencies

---

### Phase 2: Understanding — *"Find the Levers"*
**Goal**: Discover what influences your states
**Timeline**: Add 2-3 hours

#### 2.1 Correlation Discovery
- [ ] **Feature importance heatmap**: Which inputs correlate with which clusters?
- [ ] **Temporal analysis**: What happened in the hours *before* you entered a desired state?
- [ ] **Lagged correlations**: Does yesterday's sleep predict today's states?

#### 2.2 Contextual Analysis
- [ ] **Journal embedding**: Use sentence transformers to embed journal entries
- [ ] **Theme extraction**: What topics/themes correlate with states?
- [ ] **Calendar pattern analysis**: Meeting load, social time, transitions

#### 2.3 Lever Ranking
- [ ] **SHAP values**: For any predictive model, show which features matter most
- [ ] **Attention visualization**: If using transformer, show what the model attends to
- [ ] **Controllable vs uncontrollable**: Separate levers you can actually pull

---

### Phase 3: Prediction — *"Know Before You Feel"*
**Goal**: Predict states before/as they happen
**Timeline**: Add 2-3 hours

#### 3.1 State Prediction Model
- [ ] **Multimodal fusion network**: Combine EEG + biometrics + context
- [ ] **Sequence modeling**: LSTM or Transformer over temporal windows
- [ ] **Probabilistic output**: "70% likely to be in focus state in 30 min"

#### 3.2 Transition Prediction
- [ ] **Transition probability matrix**: Given current state, predict next state
- [ ] **Early warning**: Detect when you're drifting from a desired state
- [ ] **Momentum indicators**: Are you stable or in transition?

#### 3.3 Labeling Interface
- [ ] **Quick label**: Button to mark current moment with a state
- [ ] **Retrospective labeling**: Review past sessions, label segments
- [ ] **Continuous scales**: Joy (1-10), Awareness (1-10), etc.
- [ ] **Open tags**: Custom tags for states you discover

---

### Phase 4: Guidance — *"The RL Coach"*
**Goal**: Learn optimal interventions
**Timeline**: Add 3-4 hours

#### 4.1 Contextual Bandit Framework
- [ ] **State representation**: Current EEG state + context + recent trajectory
- [ ] **Action space**: Possible interventions (breathwork, break, music, journaling, etc.)
- [ ] **Reward signal**: Did the intervention move toward desired state?
- [ ] **Thompson sampling**: Balance exploration of new interventions vs exploitation

#### 4.2 Intervention Logging
- [ ] **Log interventions**: Record when you try something
- [ ] **Outcome tracking**: Measure state change after intervention
- [ ] **Context-dependent learning**: "Breathwork works when tired but not when anxious"

#### 4.3 Recommendations
- [ ] **Contextual suggestions**: "Based on your current state and history, consider..."
- [ ] **Confidence levels**: Only suggest when model is confident
- [ ] **Explanation**: Why this recommendation?

---

### Phase 5: Expansion — *"Full Observatory"*
**Goal**: Production-quality personal research platform
**Timeline**: Ongoing

#### 5.1 Advanced Analytics
- [ ] **Granger causality testing**: Does X actually cause Y, or just correlate?
- [ ] **Intervention effect estimation**: Causal inference for your interventions
- [ ] **Counterfactual simulation**: "What if I had slept 8 hours?"

#### 5.2 Long-term Patterns
- [ ] **Weekly/monthly rhythms**: Cyclical patterns in your states
- [ ] **Drift detection**: Is your baseline changing over months?
- [ ] **Goal tracking**: Progress toward cultivating desired states

#### 5.3 Integration Expansion
- [ ] **Oura ring**: Sleep and readiness data
- [ ] **Location data**: Places and state correlations
- [ ] **Weather API**: Environmental factors
- [ ] **Music listening history**: Spotify API
- [ ] **Screen time**: App usage patterns

#### 5.4 Advanced Visualizations
- [ ] **3D state space**: WebGL interactive exploration
- [ ] **Sankey diagrams**: Flow between states over time
- [ ] **Personal periodic table**: Your discovered states as elements

---

## Deep Learning Components

### Model 1: EEG State Encoder
**Purpose**: Learn compressed representation of brain state from raw EEG

```
Architecture:
- Input: (channels=4, timepoints=256) — 1 second of Muse EEG
- 1D CNN layers for temporal feature extraction
- Channel attention for spatial weighting
- Output: 64-dim latent vector

Training: Self-supervised (contrastive learning on temporal neighbors)
```

### Model 2: Multimodal Fusion Network
**Purpose**: Combine all data streams into unified state representation

```
Architecture:
- EEG branch: Pretrained encoder → 64-dim
- Biometric branch: MLP on Garmin features → 32-dim
- Context branch: Calendar + time encodings → 32-dim  
- Text branch: Sentence transformer on journal → 64-dim
- Fusion: Concatenate → MLP → 128-dim unified state
- Prediction head: → State probabilities or regression

Training: Supervised on your labels (once you have them)
```

### Model 3: Temporal Dynamics Model
**Purpose**: Model how states evolve and transition

```
Architecture:
- Input: Sequence of unified states (past N timesteps)
- Transformer encoder with positional encoding
- Output: Next state prediction + transition probabilities

Training: Self-supervised (predict next state)
```

---

## Reinforcement Learning Components

### Contextual Bandit for Intervention Learning

**State space (context)**:
- Current brain state (from encoder)
- Recent trajectory (stable, improving, declining)
- Time of day, day of week
- Recent sleep quality
- Current calendar context

**Action space**:
- Breathwork (various types)
- Meditation
- Physical movement
- Music/soundscape
- Social connection
- Nature exposure
- Journaling
- Nap
- Caffeine/nutrition
- Change of environment

**Reward function**:
- Primary: Change in labeled state (e.g., joy rating)
- Secondary: Movement toward desired cluster in state space
- Penalty: Intervention cost (time, effort)

**Algorithm**: Thompson Sampling with neural network reward model
- Maintains uncertainty over intervention effectiveness
- Naturally balances exploration vs exploitation
- Updates beliefs as you try interventions

---

## Target States: Working Definitions

These are starting points — you'll refine based on what the data reveals.

### Enlightenment
*Working definition*: Moments of clarity, insight, non-dual awareness, ego dissolution
*Possible EEG signatures*: High gamma, increased coherence, altered alpha
*Likely correlates*: Meditation, flow states, certain sleep stages (explore!)

### Loving Awareness  
*Working definition*: Warmth, compassion, connection, heart-centered presence
*Possible EEG signatures*: Left frontal asymmetry, theta, heart-brain coherence
*Likely correlates*: Social connection, gratitude practice, certain music

### Generativity
*Working definition*: Creative flow, ideas emerging, productive output
*Possible EEG signatures*: Alpha (relaxed focus), theta bursts, low beta
*Likely correlates*: Unstructured time, sleep quality, low meeting load

### Joy
*Working definition*: Positive affect, aliveness, delight
*Possible EEG signatures*: Left frontal asymmetry, alpha, low high-beta
*Likely correlates*: Play, nature, connection, accomplishment

---

## Technical Stack

### Recommended Technologies

| Component | Technology | Why |
|-----------|------------|-----|
| Language | Python 3.11+ | Ecosystem, ML libraries |
| EEG Processing | MNE-Python | Industry standard for EEG |
| Deep Learning | PyTorch | Flexibility, research-friendly |
| Data Storage | SQLite + Parquet | Simple, portable, efficient |
| Dashboard | Streamlit or Plotly Dash | Fast to build, interactive |
| Visualization | Plotly, UMAP | Interactive, beautiful |
| Text Embeddings | sentence-transformers | Easy, high quality |
| Causality | DoWhy, CausalML | Rigorous causal inference |

### Data Schema (Simplified)

```sql
-- Core time-series features
CREATE TABLE features (
    timestamp DATETIME PRIMARY KEY,
    -- EEG features
    alpha_power REAL,
    beta_power REAL,
    theta_power REAL,
    gamma_power REAL,
    frontal_asymmetry REAL,
    theta_beta_ratio REAL,
    -- Biometrics
    heart_rate REAL,
    hrv_rmssd REAL,
    stress_level REAL,
    body_battery REAL,
    -- Context
    calendar_embedding BLOB,  -- serialized vector
    -- Derived
    unified_state BLOB  -- from fusion model
);

-- User labels
CREATE TABLE labels (
    timestamp DATETIME PRIMARY KEY,
    joy INTEGER,  -- 1-10
    loving_awareness INTEGER,
    generativity INTEGER,
    enlightenment INTEGER,
    custom_tags TEXT,  -- JSON array
    notes TEXT
);

-- Interventions
CREATE TABLE interventions (
    timestamp DATETIME PRIMARY KEY,
    intervention_type TEXT,
    duration_minutes INTEGER,
    state_before BLOB,
    state_after BLOB,
    subjective_effectiveness INTEGER  -- 1-10
);
```

---

## Success Metrics

### Phase 1 Success
- [ ] Can stream and visualize live EEG during a session
- [ ] Can see yourself as a point moving through 2D state space
- [ ] Automatic clustering identifies 3-7 distinct states

### Phase 2 Success  
- [ ] Can identify top 5 variables correlated with each state
- [ ] Temporal patterns visible (e.g., "focus states cluster after morning")
- [ ] At least one surprising insight discovered

### Phase 3 Success
- [ ] Model predicts current state from inputs with >60% accuracy
- [ ] Early warning works for state transitions
- [ ] Labeling interface makes data collection easy

### Phase 4 Success
- [ ] Bandit learns personalized intervention preferences
- [ ] Recommendations feel relevant and useful
- [ ] Measurable improvement in time spent in desired states

---

## Getting Started: MVP Checklist

1. **Set up Muse streaming** (30 min)
   - Install muselsl or use Muse Direct
   - Verify you can receive live data

2. **Build feature extraction pipeline** (1 hour)
   - Band power computation
   - Basic artifact rejection
   - Save to SQLite

3. **Create state space visualization** (1 hour)
   - UMAP on accumulated features
   - Interactive Plotly scatter
   - Color by time/cluster

4. **Build minimal dashboard** (1 hour)
   - Streamlit app
   - Live view + historical view
   - Basic statistics

5. **First session** (30 min)
   - Record a session with varied states
   - Meditate, work, rest, etc.
   - See yourself in state space

---

## Future Explorations

### Research Questions to Investigate
- What is the signature of your most creative moments?
- Does morning meditation affect afternoon brain states?
- Which people/activities reliably shift you toward joy?
- Can you learn to voluntarily navigate to desired states?
- What's the minimum effective dose for each intervention?

### Wild Ideas for Later
- **Dream integration**: Correlate sleep EEG with dream journals
- **Synchrony tracking**: Record EEG with others, measure coherence
- **Prediction markets**: Bet on your own future states
- **Generative states**: Train a model to generate "target" EEG patterns

---

## Appendix: Muse EEG Channels

| Channel | Location | Associated Functions |
|---------|----------|---------------------|
| TP9 | Left temporal | Language, memory |
| AF7 | Left frontal | Approach motivation, positive affect |
| AF8 | Right frontal | Withdrawal motivation, negative affect |
| TP10 | Right temporal | Spatial, emotional |

**Key derived metrics**:
- **Frontal asymmetry** = log(AF8_alpha) - log(AF7_alpha)
  - Positive = more left activation = approach/positive
  - Negative = more right activation = withdrawal/negative

---

*"The real voyage of discovery consists not in seeking new landscapes, but in having new eyes."* — Marcel Proust

This observatory gives you new eyes for your own mind.
