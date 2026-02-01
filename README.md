# Consciousness Observatory

A personal research platform for mapping the high-dimensional dynamics of mind through multimodal data integration.

## Vision

Discover the **attractors** and **manifold structures** underlying states of:
- Generativity & creative flow
- Grounded confidence
- Clarity & enlightenment  
- Peak cognitive performance
- Loving awareness

By collecting and analyzing data from multiple sources (physiology, behavior, context, and phenomenology), we can begin to see patterns that aren't visible from any single stream alone.

---

## Project Structure

```
consciousness-research/
├── data_pipeline/
│   ├── fetchers.py          # Data fetchers for Garmin & Calendar
│   ├── sync_scheduler.py    # Automated background sync
│   └── requirements.txt     # Python dependencies
├── dashboard/
│   └── ConsciousnessObservatory.jsx  # React dashboard
└── README.md
```

---

## Quick Start

### 1. Set Up the Data Pipeline

```bash
cd data_pipeline
pip install -r requirements.txt
```

### 2. Configure Credentials

Create the credentials directory:
```bash
mkdir -p ~/.consciousness_research/credentials
```

#### Garmin Connect
Set environment variables:
```bash
export GARMIN_EMAIL="your-email@example.com"
export GARMIN_PASSWORD="your-password"
```

Or, the system will prompt for credentials on first run and save tokens for future use.

#### Google Calendar
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable the Google Calendar API
4. Create OAuth 2.0 credentials (Desktop application)
5. Download the JSON and save as:
   ```
   ~/.consciousness_research/credentials/google_client_secrets.json
   ```

### 3. Run Initial Setup

```bash
python fetchers.py setup
```

This will authenticate with both services and save tokens.

### 4. Sync Data

One-time sync:
```bash
python fetchers.py sync --days 30
```

Or run the scheduler for continuous sync:
```bash
python sync_scheduler.py
```

### 5. Export Data for Analysis

```bash
python fetchers.py export --start 2026-01-01 --end 2026-01-22 --output my_data.json
```

---

## Data Architecture

### Unified Database Schema

All data is stored in SQLite (`consciousness_research.db`) with the following tables:

| Table | Purpose |
|-------|---------|
| `health_snapshots` | Raw Garmin data (HRV, sleep, stress, etc.) |
| `calendar_events` | Google Calendar events with metadata |
| `daily_context` | Aggregated daily summaries |
| `eeg_sessions` | Muse EEG recordings (future) |
| `journal_entries` | Phenomenological reports (future) |
| `feature_vectors` | Computed features for ML analysis |
| `insights` | Research notes and findings |

### Health Metrics Captured

From Garmin:
- Heart Rate Variability (RMSSD, SDRR)
- Resting Heart Rate
- Sleep architecture (deep, light, REM, awake)
- Sleep score
- Body Battery (energy level)
- Stress level (0-100)
- Steps & active calories
- Respiration rate
- SpO2

### Calendar Metrics Captured

- Event count and duration
- Attendee count
- Context switches (events with <15min gaps)
- Focus block identification
- Calendar source (for multi-calendar support)

---

## Dashboard

The React dashboard provides a visual interface for:
- Monitoring data source connectivity
- Viewing real-time metrics
- Recording research insights
- Tracking target mental states
- Viewing trends over time

To use the dashboard, copy `ConsciousnessObservatory.jsx` into your React project or render it directly in claude.ai.

---

## Research Roadmap

### Phase 1: Data Collection (Current)
- [x] Garmin health data integration
- [x] Google Calendar integration
- [ ] Muse EEG integration
- [ ] Structured journaling system
- [ ] State labeling protocol

### Phase 2: Feature Engineering
- [ ] Time-aligned multimodal feature vectors
- [ ] Derived complexity metrics
- [ ] Temporal dynamics features (rate of change, recurrence)

### Phase 3: Manifold Learning
- [ ] Dimensionality reduction (PCA, UMAP, diffusion maps)
- [ ] State-space model fitting
- [ ] Attractor identification

### Phase 4: Insight Generation
- [ ] Correlation analysis with phenomenological reports
- [ ] Predictive modeling for target states
- [ ] Intervention experiments

---

## Theoretical Background

This project draws from:

- **Dynamical Systems Theory**: Treating the mind-body system as a trajectory through state space
- **Manifold Learning**: Finding lower-dimensional structures in high-dimensional data
- **Contemplative Science**: Using phenomenological reports as ground truth
- **Personal Informatics**: Self-tracking for self-knowledge

Key hypothesis: Mental states like "generativity" or "loving awareness" are not random points in physiological space—they are **attractors** or **modes** that the system tends toward under certain conditions. By mapping these structures, we can learn to navigate to them more reliably.

---

## Adding New Data Sources

The `BaseFetcher` class provides a template for adding new sources:

```python
class MySourceFetcher(BaseFetcher):
    def authenticate(self) -> bool:
        # Implement authentication
        pass
    
    def fetch(self, start_date, end_date) -> List[Any]:
        # Fetch and store data
        pass
    
    def sync(self) -> int:
        # Sync recent data
        pass
```

---

## Privacy & Security

- All data is stored locally in SQLite
- Credentials are stored in `~/.consciousness_research/credentials/`
- No data is sent to external services (beyond the source APIs)
- Consider encrypting the database for sensitive data

---

## License

Personal research project. Use and adapt freely.
