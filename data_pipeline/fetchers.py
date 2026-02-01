"""
Consciousness Research Data Pipeline
=====================================
Automated fetchers for Garmin health data and Google Calendar.
Stores everything in a unified time-indexed format.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class HealthSnapshot:
    """A point-in-time health measurement from Garmin."""
    timestamp: str
    source: str = "garmin"
    
    # Heart metrics
    heart_rate: Optional[int] = None
    resting_heart_rate: Optional[int] = None
    hrv_rmssd: Optional[float] = None  # Root mean square of successive differences
    hrv_sdrr: Optional[float] = None   # Standard deviation of RR intervals
    
    # Sleep metrics
    sleep_score: Optional[int] = None
    deep_sleep_minutes: Optional[int] = None
    light_sleep_minutes: Optional[int] = None
    rem_sleep_minutes: Optional[int] = None
    awake_minutes: Optional[int] = None
    
    # Activity metrics
    steps: Optional[int] = None
    active_calories: Optional[int] = None
    stress_level: Optional[int] = None  # Garmin's 0-100 stress score
    body_battery: Optional[int] = None  # Garmin's energy metric
    
    # Respiratory
    respiration_rate: Optional[float] = None
    spo2: Optional[float] = None


@dataclass
class CalendarEvent:
    """A calendar event with metadata for context analysis."""
    timestamp: str
    source: str = "google_calendar"
    
    event_id: str = ""
    title: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_minutes: int = 0
    
    # Derived/tagged fields
    category: Optional[str] = None  # meeting, focus_time, social, etc.
    energy_tag: Optional[str] = None  # generative, depleting, neutral
    attendee_count: int = 0
    is_recurring: bool = False
    
    # Calendar metadata
    calendar_name: str = ""
    location: Optional[str] = None
    description_hash: Optional[str] = None  # For change detection, not content


@dataclass
class JournalEntry:
    """A timestamped journal entry parsed from markdown."""
    timestamp: str  # ISO format datetime
    date: str  # YYYY-MM-DD
    time: str  # HH:MM (24h format)
    content: str
    source_file: str

    # Extracted metadata
    mood: Optional[str] = None
    energy: Optional[int] = None  # 1-10
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class DailyContext:
    """Aggregated daily context combining multiple sources."""
    date: str

    # Calendar aggregates
    total_meetings: int = 0
    total_meeting_minutes: int = 0
    focus_blocks: int = 0
    context_switches: int = 0

    # Health aggregates
    avg_hrv: Optional[float] = None
    avg_stress: Optional[float] = None
    sleep_score: Optional[int] = None
    body_battery_morning: Optional[int] = None

    # Manual annotations (from journaling integration later)
    primary_state: Optional[str] = None
    energy_rating: Optional[int] = None
    notes: Optional[str] = None


# ============================================================================
# Database Manager
# ============================================================================

class DataStore:
    """SQLite-based unified data store for all research data."""
    
    def __init__(self, db_path: str = "consciousness_research.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database with required tables."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        self.conn.executescript("""
            -- Raw health snapshots from Garmin
            CREATE TABLE IF NOT EXISTS health_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT DEFAULT 'garmin',
                data JSON NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, source)
            );
            
            -- Calendar events
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT DEFAULT 'google_calendar',
                data JSON NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, source)
            );
            
            -- Daily aggregated context
            CREATE TABLE IF NOT EXISTS daily_context (
                date TEXT PRIMARY KEY,
                data JSON NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            -- EEG sessions (for later)
            CREATE TABLE IF NOT EXISTS eeg_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                file_path TEXT,
                metadata JSON,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Journal entries (for later)
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                entry_type TEXT DEFAULT 'freeform',
                content TEXT,
                structured_data JSON,
                state_labels JSON,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Computed features for ML
            CREATE TABLE IF NOT EXISTS feature_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                window_type TEXT DEFAULT '30min',
                features JSON NOT NULL,
                labels JSON,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(window_start, window_end, window_type)
            );
            
            -- Research insights and annotations
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                insight_type TEXT,
                title TEXT,
                content TEXT,
                related_data JSON,
                tags JSON
            );
            
            CREATE INDEX IF NOT EXISTS idx_health_timestamp ON health_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_calendar_timestamp ON calendar_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_features_window ON feature_vectors(window_start, window_end);
            CREATE INDEX IF NOT EXISTS idx_eeg_start_time ON eeg_sessions(start_time);
        """)
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def store_health_snapshot(self, snapshot: HealthSnapshot):
        """Store a health data snapshot."""
        self.conn.execute(
            "INSERT OR REPLACE INTO health_snapshots (timestamp, source, data) VALUES (?, ?, ?)",
            (snapshot.timestamp, snapshot.source, json.dumps(asdict(snapshot)))
        )
        self.conn.commit()
    
    def store_calendar_event(self, event: CalendarEvent):
        """Store a calendar event."""
        self.conn.execute(
            "INSERT OR REPLACE INTO calendar_events (event_id, timestamp, source, data) VALUES (?, ?, ?, ?)",
            (event.event_id, event.timestamp, event.source, json.dumps(asdict(event)))
        )
        self.conn.commit()
    
    def store_daily_context(self, context: DailyContext):
        """Store daily aggregated context."""
        self.conn.execute(
            "INSERT OR REPLACE INTO daily_context (date, data, updated_at) VALUES (?, ?, ?)",
            (context.date, json.dumps(asdict(context)), datetime.now().isoformat())
        )
        self.conn.commit()
    
    def get_health_data(self, start_date: str, end_date: str) -> List[Dict]:
        """Retrieve health snapshots for a date range."""
        cursor = self.conn.execute(
            "SELECT data FROM health_snapshots WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            (start_date, end_date)
        )
        return [json.loads(row['data']) for row in cursor.fetchall()]
    
    def get_calendar_events(self, start_date: str, end_date: str) -> List[Dict]:
        """Retrieve calendar events for a date range."""
        cursor = self.conn.execute(
            "SELECT data FROM calendar_events WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            (start_date, end_date)
        )
        return [json.loads(row['data']) for row in cursor.fetchall()]
    
    def get_daily_context(self, date: str) -> Optional[Dict]:
        """Retrieve daily context for a specific date."""
        cursor = self.conn.execute(
            "SELECT data FROM daily_context WHERE date = ?", (date,)
        )
        row = cursor.fetchone()
        return json.loads(row['data']) if row else None

    def store_journal_entry(self, entry: 'JournalEntry'):
        """Store a journal entry."""
        self.conn.execute(
            """INSERT OR REPLACE INTO journal_entries
               (timestamp, entry_type, content, structured_data, state_labels)
               VALUES (?, ?, ?, ?, ?)""",
            (entry.timestamp, 'freeform', entry.content,
             json.dumps(asdict(entry)),
             json.dumps(entry.tags))
        )
        self.conn.commit()

    def get_journal_entries(self, start_date: str, end_date: str) -> List[Dict]:
        """Retrieve journal entries for a date range."""
        cursor = self.conn.execute(
            """SELECT structured_data FROM journal_entries
               WHERE timestamp >= ? AND timestamp < ?
               ORDER BY timestamp""",
            (start_date, end_date)
        )
        return [json.loads(row['structured_data']) for row in cursor.fetchall()]

    def clear_journal_entries_for_date(self, date: str):
        """Clear all journal entries for a specific date (for re-import)."""
        next_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        self.conn.execute(
            "DELETE FROM journal_entries WHERE timestamp >= ? AND timestamp < ?",
            (date, next_date)
        )
        self.conn.commit()
    
    def store_insight(self, insight_type: str, title: str, content: str, 
                      related_data: Dict = None, tags: List[str] = None):
        """Store a research insight."""
        self.conn.execute(
            "INSERT INTO insights (insight_type, title, content, related_data, tags) VALUES (?, ?, ?, ?, ?)",
            (insight_type, title, content, json.dumps(related_data or {}), json.dumps(tags or []))
        )
        self.conn.commit()
    
    def export_for_analysis(self, start_date: str, end_date: str) -> Dict:
        """Export all data for a date range in a format ready for analysis."""
        return {
            "health": self.get_health_data(start_date, end_date),
            "calendar": self.get_calendar_events(start_date, end_date),
            "date_range": {"start": start_date, "end": end_date},
            "exported_at": datetime.now().isoformat()
        }


# ============================================================================
# Base Fetcher
# ============================================================================

class BaseFetcher(ABC):
    """Abstract base class for data fetchers."""
    
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.credentials_path = Path.home() / ".consciousness_research" / "credentials"
        self.credentials_path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the data source."""
        pass
    
    @abstractmethod
    def fetch(self, start_date: datetime, end_date: datetime) -> List[Any]:
        """Fetch data for the given date range."""
        pass
    
    @abstractmethod
    def sync(self) -> int:
        """Sync recent data. Returns count of new records."""
        pass


# ============================================================================
# Garmin Fetcher
# ============================================================================

class GarminFetcher(BaseFetcher):
    """
    Fetches health data from Garmin Connect.
    
    Uses the garminconnect library for API access.
    Install: pip install garminconnect
    """
    
    def __init__(self, data_store: DataStore):
        super().__init__(data_store)
        self.client = None
        self.token_path = self.credentials_path / "garmin_tokens"  # Directory for garth tokens

    def authenticate(self, email: str = None, password: str = None) -> bool:
        """
        Authenticate with Garmin Connect.

        First tries to use saved tokens, falls back to email/password.
        Credentials can also be set via environment variables:
        - GARMIN_EMAIL
        - GARMIN_PASSWORD

        Note: If MFA is enabled, run the interactive login first:
            import garth
            garth.login(email, password)  # Will prompt for MFA
            garth.client.dump('~/.consciousness_research/credentials/garmin_tokens')
        """
        try:
            from garminconnect import Garmin
            import garth
        except ImportError:
            logger.error("garminconnect not installed. Run: pip install garminconnect")
            return False

        # Try to load saved garth tokens first
        if self.token_path.exists():
            try:
                garth.client.load(str(self.token_path))
                self.client = Garmin()
                self.client.garth = garth.client
                self.client.display_name = garth.client.profile.get('displayName', 'User')
                logger.info("Authenticated with Garmin using saved tokens")
                return True
            except Exception as e:
                logger.warning(f"Token auth failed: {e}")

        # Fall back to email/password (won't work with MFA without interaction)
        email = email or os.environ.get("GARMIN_EMAIL")
        password = password or os.environ.get("GARMIN_PASSWORD")

        if not email or not password:
            logger.error("Garmin credentials required. Set GARMIN_EMAIL and GARMIN_PASSWORD")
            return False

        try:
            garth.login(email, password)
            garth.client.dump(str(self.token_path))

            self.client = Garmin()
            self.client.garth = garth.client
            self.client.display_name = garth.client.profile.get('displayName', 'User')

            logger.info("Authenticated with Garmin and saved tokens")
            return True
        except Exception as e:
            logger.error(f"Garmin authentication failed: {e}")
            return False
    
    def fetch(self, start_date: datetime, end_date: datetime) -> List[HealthSnapshot]:
        """Fetch health data for the given date range."""
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        snapshots = []
        current = start_date
        
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            
            try:
                snapshot = self._fetch_day(date_str)
                if snapshot:
                    snapshots.append(snapshot)
                    self.data_store.store_health_snapshot(snapshot)
                    logger.info(f"Fetched Garmin data for {date_str}")
            except Exception as e:
                logger.warning(f"Failed to fetch Garmin data for {date_str}: {e}")
            
            current += timedelta(days=1)
        
        return snapshots
    
    def _fetch_day(self, date_str: str) -> Optional[HealthSnapshot]:
        """Fetch all available health data for a single day."""
        snapshot = HealthSnapshot(timestamp=f"{date_str}T00:00:00")
        
        try:
            # Heart rate data
            hr_data = self.client.get_heart_rates(date_str)
            if hr_data:
                snapshot.resting_heart_rate = hr_data.get('restingHeartRate')
        except Exception as e:
            logger.debug(f"No heart rate data: {e}")
        
        try:
            # HRV data
            hrv_data = self.client.get_hrv_data(date_str)
            if hrv_data and hrv_data.get('hrvSummary'):
                summary = hrv_data['hrvSummary']
                snapshot.hrv_rmssd = summary.get('lastNightAvg')
                snapshot.hrv_sdrr = summary.get('weeklyAvg')  # Use as proxy
        except Exception as e:
            logger.debug(f"No HRV data: {e}")
        
        try:
            # Sleep data
            sleep_data = self.client.get_sleep_data(date_str)
            if sleep_data:
                # Sleep scores are nested in dailySleepDTO
                daily_dto = sleep_data.get('dailySleepDTO', {})
                sleep_scores = daily_dto.get('sleepScores', {})
                overall = sleep_scores.get('overall', {})
                snapshot.sleep_score = overall.get('value') if isinstance(overall, dict) else overall

                # Sleep durations are also in dailySleepDTO
                snapshot.deep_sleep_minutes = daily_dto.get('deepSleepSeconds', 0) // 60 if daily_dto.get('deepSleepSeconds') else None
                snapshot.light_sleep_minutes = daily_dto.get('lightSleepSeconds', 0) // 60 if daily_dto.get('lightSleepSeconds') else None
                snapshot.rem_sleep_minutes = daily_dto.get('remSleepSeconds', 0) // 60 if daily_dto.get('remSleepSeconds') else None
                snapshot.awake_minutes = daily_dto.get('awakeSleepSeconds', 0) // 60 if daily_dto.get('awakeSleepSeconds') else None
        except Exception as e:
            logger.debug(f"No sleep data: {e}")
        
        try:
            # Daily stats
            stats = self.client.get_stats(date_str)
            if stats:
                snapshot.steps = stats.get('totalSteps')
                snapshot.active_calories = stats.get('activeKilocalories')
                snapshot.stress_level = stats.get('averageStressLevel')
        except Exception as e:
            logger.debug(f"No daily stats: {e}")
        
        try:
            # Body battery
            bb_data = self.client.get_body_battery(date_str)
            if bb_data and len(bb_data) > 0:
                # Get morning body battery (first reading)
                snapshot.body_battery = bb_data[0].get('chargedValue')
        except Exception as e:
            logger.debug(f"No body battery data: {e}")
        
        try:
            # Respiration
            resp_data = self.client.get_respiration_data(date_str)
            if resp_data:
                snapshot.respiration_rate = resp_data.get('avgBreathingRate')
        except Exception as e:
            logger.debug(f"No respiration data: {e}")
        
        try:
            # SpO2
            spo2_data = self.client.get_spo2_data(date_str)
            if spo2_data:
                snapshot.spo2 = spo2_data.get('averageSpO2')
        except Exception as e:
            logger.debug(f"No SpO2 data: {e}")
        
        return snapshot
    
    def sync(self, days_back: int = 7) -> int:
        """Sync the last N days of data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        snapshots = self.fetch(start_date, end_date)
        return len(snapshots)


# ============================================================================
# Google Calendar Fetcher
# ============================================================================

class GoogleCalendarFetcher(BaseFetcher):
    """
    Fetches events from Google Calendar.
    
    Uses the Google Calendar API with OAuth2.
    Requires setting up credentials in Google Cloud Console.
    """
    
    SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
    
    def __init__(self, data_store: DataStore):
        super().__init__(data_store)
        self.service = None
        self.token_path = self.credentials_path / "google_token.json"
        self.client_secrets_path = self.credentials_path / "google_client_secrets.json"
    
    def authenticate(self) -> bool:
        """
        Authenticate with Google Calendar API.
        
        Requires google_client_secrets.json in ~/.consciousness_research/credentials/
        Download from Google Cloud Console > APIs & Services > Credentials
        """
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            logger.error(
                "Google API libraries not installed. Run:\n"
                "pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
            )
            return False
        
        creds = None
        
        # Load existing token
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), self.SCOPES)
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"Token refresh failed: {e}")
                    creds = None
            
            if not creds:
                if not self.client_secrets_path.exists():
                    logger.error(
                        f"Client secrets not found at {self.client_secrets_path}\n"
                        "Download from Google Cloud Console > APIs & Services > Credentials"
                    )
                    return False
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.client_secrets_path), self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save token
            with open(self.token_path, 'w') as f:
                f.write(creds.to_json())
        
        self.service = build('calendar', 'v3', credentials=creds)
        logger.info("Authenticated with Google Calendar")
        return True
    
    def fetch(self, start_date: datetime, end_date: datetime, 
              calendar_id: str = 'primary') -> List[CalendarEvent]:
        """Fetch calendar events for the given date range."""
        if not self.service:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        events = []
        
        # Format times for API
        time_min = start_date.isoformat() + 'Z'
        time_max = end_date.isoformat() + 'Z'
        
        try:
            # Fetch events (paginated)
            page_token = None
            while True:
                result = self.service.events().list(
                    calendarId=calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy='startTime',
                    pageToken=page_token
                ).execute()
                
                for item in result.get('items', []):
                    event = self._parse_event(item, calendar_id)
                    if event:
                        events.append(event)
                        self.data_store.store_calendar_event(event)
                
                page_token = result.get('nextPageToken')
                if not page_token:
                    break
            
            logger.info(f"Fetched {len(events)} calendar events")
            
        except Exception as e:
            logger.error(f"Failed to fetch calendar events: {e}")
        
        return events
    
    def _parse_event(self, item: Dict, calendar_name: str) -> Optional[CalendarEvent]:
        """Parse a Google Calendar event into our format."""
        try:
            # Get start time (handle all-day events)
            start = item.get('start', {})
            start_time = start.get('dateTime') or start.get('date')
            
            end = item.get('end', {})
            end_time = end.get('dateTime') or end.get('date')
            
            # Calculate duration
            try:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                duration = int((end_dt - start_dt).total_seconds() / 60)
            except:
                duration = 0
            
            # Count attendees
            attendees = item.get('attendees', [])
            attendee_count = len([a for a in attendees if a.get('responseStatus') != 'declined'])
            
            return CalendarEvent(
                timestamp=start_time,
                event_id=item['id'],
                title=item.get('summary', 'Untitled'),
                start_time=start_time,
                end_time=end_time,
                duration_minutes=duration,
                attendee_count=attendee_count,
                is_recurring=bool(item.get('recurringEventId')),
                calendar_name=calendar_name,
                location=item.get('location'),
                description_hash=str(hash(item.get('description', '')))[:16]
            )
        except Exception as e:
            logger.warning(f"Failed to parse event: {e}")
            return None
    
    def fetch_all_calendars(self, start_date: datetime, end_date: datetime) -> List[CalendarEvent]:
        """Fetch events from all accessible calendars."""
        if not self.service:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        all_events = []
        
        # Get list of calendars
        calendars = self.service.calendarList().list().execute()
        
        for cal in calendars.get('items', []):
            cal_id = cal['id']
            cal_name = cal.get('summary', cal_id)
            
            logger.info(f"Fetching from calendar: {cal_name}")
            events = self.fetch(start_date, end_date, calendar_id=cal_id)
            
            # Update calendar name in events
            for event in events:
                event.calendar_name = cal_name
            
            all_events.extend(events)
        
        return all_events
    
    def sync(self, days_back: int = 30, days_forward: int = 7) -> int:
        """Sync calendar events for the specified range."""
        now = datetime.now()
        start_date = now - timedelta(days=days_back)
        end_date = now + timedelta(days=days_forward)
        events = self.fetch_all_calendars(start_date, end_date)
        return len(events)


# ============================================================================
# Journal Fetcher
# ============================================================================

import re

class JournalFetcher(BaseFetcher):
    """
    Parses journal entries from markdown files in the journal/ directory.

    Supports formats like:
    - "8:00 - content"
    - "9am - content"
    - "12:30pm - content"
    - "14:15 content"
    """

    # Time patterns to match various formats
    TIME_PATTERNS = [
        r'^(\d{1,2}):(\d{2})\s*(?:am|pm)?\s*[-–—]?\s*(.+)',  # 8:00 - content, 12:30pm - content
        r'^(\d{1,2})\s*(am|pm)\s*[-–—]?\s*(.+)',  # 9am - content
        r'^(\d{1,2}):(\d{2})\s+(.+)',  # 14:15 content (no dash)
    ]

    def __init__(self, data_store: DataStore, journal_path: str = None):
        super().__init__(data_store)
        # Default to journal/ in the repo root
        if journal_path:
            self.journal_path = Path(journal_path)
        else:
            self.journal_path = Path(__file__).parent.parent / "journal"

    def authenticate(self) -> bool:
        """Journal doesn't need authentication, just verify path exists."""
        if not self.journal_path.exists():
            self.journal_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created journal directory: {self.journal_path}")
        return True

    def fetch(self, start_date: datetime, end_date: datetime) -> List[JournalEntry]:
        """Parse journal files in the date range."""
        entries = []

        for file_path in self.journal_path.glob("*.md"):
            file_date = self._parse_filename_date(file_path.name)
            if file_date is None:
                continue

            # Check if file is in date range
            if start_date.date() <= file_date <= end_date.date():
                file_entries = self._parse_journal_file(file_path, file_date)
                entries.extend(file_entries)

        return sorted(entries, key=lambda e: e.timestamp)

    def sync(self, days_back: int = 30) -> int:
        """Sync journal entries from markdown files."""
        now = datetime.now()
        start_date = now - timedelta(days=days_back)
        end_date = now + timedelta(days=1)

        entries = self.fetch(start_date, end_date)

        # Group by date and re-import
        dates_processed = set()
        for entry in entries:
            if entry.date not in dates_processed:
                # Clear existing entries for this date before re-importing
                self.data_store.clear_journal_entries_for_date(entry.date)
                dates_processed.add(entry.date)

            self.data_store.store_journal_entry(entry)

        logger.info(f"Synced {len(entries)} journal entries from {len(dates_processed)} files")
        return len(entries)

    def _parse_filename_date(self, filename: str) -> Optional[datetime.date]:
        """Parse date from filename like '012626.md' or '2026-01-26.md'."""
        name = filename.replace('.md', '')

        # Try MMDDYY format (012626)
        if len(name) == 6 and name.isdigit():
            try:
                month = int(name[0:2])
                day = int(name[2:4])
                year = int('20' + name[4:6])
                return datetime(year, month, day).date()
            except ValueError:
                pass

        # Try YYYY-MM-DD format
        try:
            return datetime.strptime(name, "%Y-%m-%d").date()
        except ValueError:
            pass

        # Try MM-DD-YYYY format
        try:
            return datetime.strptime(name, "%m-%d-%Y").date()
        except ValueError:
            pass

        logger.debug(f"Could not parse date from filename: {filename}")
        return None

    def _parse_journal_file(self, file_path: Path, file_date: datetime.date) -> List[JournalEntry]:
        """Parse a single journal markdown file into entries."""
        entries = []
        current_time = None
        current_content = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip()

            # Skip empty lines at start
            if not line and not current_time:
                continue

            # Skip date header line
            if self._is_date_line(line):
                continue

            # Try to parse as a new time entry
            parsed = self._parse_time_line(line)
            if parsed:
                # Save previous entry if exists
                if current_time and current_content:
                    entry = self._create_entry(
                        file_date, current_time, '\n'.join(current_content).strip(),
                        str(file_path)
                    )
                    if entry:
                        entries.append(entry)

                current_time, content_start = parsed
                current_content = [content_start] if content_start else []
            elif current_time:
                # Continue current entry
                current_content.append(line)

        # Don't forget the last entry
        if current_time and current_content:
            entry = self._create_entry(
                file_date, current_time, '\n'.join(current_content).strip(),
                str(file_path)
            )
            if entry:
                entries.append(entry)

        return entries

    def _is_date_line(self, line: str) -> bool:
        """Check if line is a date header."""
        # Match patterns like "01-26-2026" or "2026-01-26" or "January 26, 2026"
        date_patterns = [
            r'^\d{2}-\d{2}-\d{4}$',
            r'^\d{4}-\d{2}-\d{2}$',
            r'^[A-Za-z]+ \d{1,2},? \d{4}$',
            r'^# ',  # Markdown header
        ]
        return any(re.match(p, line.strip()) for p in date_patterns)

    def _parse_time_line(self, line: str) -> Optional[tuple]:
        """
        Try to parse a line as a time entry.
        Returns (time_str, content) or None.
        """
        line = line.strip()
        if not line:
            return None

        # Pattern 1: "8:00 - content" or "12:30pm - content"
        match = re.match(r'^(\d{1,2}):(\d{2})\s*(am|pm)?\s*[-–—]?\s*(.*)$', line, re.IGNORECASE)
        if match:
            hour, minute, ampm, content = match.groups()
            hour = int(hour)
            minute = int(minute)
            if ampm:
                if ampm.lower() == 'pm' and hour != 12:
                    hour += 12
                elif ampm.lower() == 'am' and hour == 12:
                    hour = 0
            time_str = f"{hour:02d}:{minute:02d}"
            return (time_str, content.strip())

        # Pattern 2: "9am - content"
        match = re.match(r'^(\d{1,2})\s*(am|pm)\s*[-–—]?\s*(.*)$', line, re.IGNORECASE)
        if match:
            hour, ampm, content = match.groups()
            hour = int(hour)
            if ampm.lower() == 'pm' and hour != 12:
                hour += 12
            elif ampm.lower() == 'am' and hour == 12:
                hour = 0
            time_str = f"{hour:02d}:00"
            return (time_str, content.strip())

        return None

    def _create_entry(self, file_date: datetime.date, time_str: str,
                      content: str, source_file: str) -> Optional[JournalEntry]:
        """Create a JournalEntry from parsed data."""
        if not content:
            return None

        # Create full timestamp
        timestamp = f"{file_date.isoformat()}T{time_str}:00"

        # Extract mood/energy from content (simple heuristics)
        mood = self._extract_mood(content)
        energy = self._extract_energy(content)
        tags = self._extract_tags(content)

        return JournalEntry(
            timestamp=timestamp,
            date=file_date.isoformat(),
            time=time_str,
            content=content,
            source_file=source_file,
            mood=mood,
            energy=energy,
            tags=tags
        )

    def _extract_mood(self, content: str) -> Optional[str]:
        """Extract mood from content using keyword matching."""
        content_lower = content.lower()

        positive_words = ['happy', 'good', 'great', 'excited', 'inspired', 'flow', 'focused', 'energized', 'motivated']
        negative_words = ['tired', 'sad', 'anxious', 'stressed', 'unmotivated', 'lost', 'exhausted', 'frustrated']
        neutral_words = ['okay', 'fine', 'neutral', 'calm']

        for word in positive_words:
            if word in content_lower:
                return 'positive'
        for word in negative_words:
            if word in content_lower:
                return 'negative'
        for word in neutral_words:
            if word in content_lower:
                return 'neutral'

        return None

    def _extract_energy(self, content: str) -> Optional[int]:
        """Extract energy level if explicitly mentioned (1-10 scale)."""
        # Look for patterns like "energy: 7" or "energy 8/10"
        match = re.search(r'energy[:\s]+(\d+)(?:/10)?', content, re.IGNORECASE)
        if match:
            energy = int(match.group(1))
            return max(1, min(10, energy))
        return None

    def _extract_tags(self, content: str) -> List[str]:
        """Extract hashtags or tags from content."""
        # Find #hashtags
        tags = re.findall(r'#(\w+)', content)

        # Also extract keywords for common activities/states
        content_lower = content.lower()
        activity_keywords = {
            'meditation': ['meditat', 'mindful'],
            'exercise': ['workout', 'exercise', 'gym', 'run', 'walk'],
            'work': ['meeting', 'work', 'coding', 'project'],
            'social': ['friend', 'family', 'social', 'call', 'chat'],
            'creative': ['creative', 'writing', 'art', 'music'],
            'learning': ['learn', 'read', 'study', 'research'],
            'rest': ['rest', 'relax', 'nap', 'sleep'],
        }

        for tag, keywords in activity_keywords.items():
            if any(kw in content_lower for kw in keywords):
                if tag not in tags:
                    tags.append(tag)

        return tags


# ============================================================================
# Sync Orchestrator
# ============================================================================

class SyncOrchestrator:
    """Coordinates syncing across all data sources."""

    def __init__(self, db_path: str = "consciousness_research.db"):
        self.data_store = DataStore(db_path)
        self.garmin = GarminFetcher(self.data_store)
        self.calendar = GoogleCalendarFetcher(self.data_store)
        self.journal = JournalFetcher(self.data_store)
        self._eeg = None  # Lazy initialization

    def _get_eeg_fetcher(self):
        """Lazy initialization of EEG fetcher."""
        if self._eeg is None:
            try:
                from eeg_fetcher import MuseEEGFetcher
                self._eeg = MuseEEGFetcher(self.data_store)
            except ImportError:
                logger.warning("EEG fetcher not available (missing dependencies)")
        return self._eeg
    
    def setup(self, garmin_email: str = None, garmin_password: str = None) -> Dict[str, bool]:
        """Setup authentication for all sources."""
        results = {}

        results['garmin'] = self.garmin.authenticate(garmin_email, garmin_password)
        results['calendar'] = self.calendar.authenticate()

        # EEG setup is optional (device may not be present)
        eeg = self._get_eeg_fetcher()
        if eeg:
            try:
                results['eeg'] = eeg.authenticate()
            except Exception as e:
                logger.warning(f"EEG setup skipped: {e}")
                results['eeg'] = False
        else:
            results['eeg'] = False

        return results
    
    def sync_all(self, days_back: int = 7) -> Dict[str, int]:
        """Sync all data sources."""
        results = {}

        try:
            results['garmin'] = self.garmin.sync(days_back)
        except Exception as e:
            logger.error(f"Garmin sync failed: {e}")
            results['garmin'] = 0

        try:
            results['calendar'] = self.calendar.sync(days_back)
        except Exception as e:
            logger.error(f"Calendar sync failed: {e}")
            results['calendar'] = 0

        # EEG sync validates file integrity
        eeg = self._get_eeg_fetcher()
        if eeg:
            try:
                results['eeg'] = eeg.sync()
            except Exception as e:
                logger.warning(f"EEG sync skipped: {e}")
                results['eeg'] = 0
        else:
            results['eeg'] = 0

        # Journal sync parses markdown files
        try:
            results['journal'] = self.journal.sync(days_back)
        except Exception as e:
            logger.error(f"Journal sync failed: {e}")
            results['journal'] = 0

        return results
    
    def compute_daily_context(self, date: str) -> DailyContext:
        """Compute aggregated context for a specific date."""
        next_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
        health_data = self.data_store.get_health_data(date, next_date)
        calendar_data = self.data_store.get_calendar_events(date, next_date)
        
        context = DailyContext(date=date)
        
        # Aggregate health
        if health_data:
            hrvs = [h.get('hrv_rmssd') for h in health_data if h.get('hrv_rmssd')]
            stresses = [h.get('stress_level') for h in health_data if h.get('stress_level')]
            
            context.avg_hrv = sum(hrvs) / len(hrvs) if hrvs else None
            context.avg_stress = sum(stresses) / len(stresses) if stresses else None
            context.sleep_score = health_data[0].get('sleep_score')
            context.body_battery_morning = health_data[0].get('body_battery')
        
        # Aggregate calendar
        context.total_meetings = len(calendar_data)
        context.total_meeting_minutes = sum(e.get('duration_minutes', 0) for e in calendar_data)
        
        # Estimate context switches (events with gaps < 15 min)
        if len(calendar_data) > 1:
            switches = 0
            for i in range(1, len(calendar_data)):
                try:
                    prev_end = datetime.fromisoformat(calendar_data[i-1]['end_time'].replace('Z', '+00:00'))
                    curr_start = datetime.fromisoformat(calendar_data[i]['start_time'].replace('Z', '+00:00'))
                    gap = (curr_start - prev_end).total_seconds() / 60
                    if 0 < gap < 15:
                        switches += 1
                except:
                    pass
            context.context_switches = switches
        
        self.data_store.store_daily_context(context)
        return context
    
    def export_research_data(self, start_date: str, end_date: str, 
                             output_path: str = "research_export.json"):
        """Export all data for analysis."""
        data = self.data_store.export_for_analysis(start_date, end_date)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported research data to {output_path}")
        return output_path


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Consciousness Research Data Pipeline")
    parser.add_argument("command", choices=["setup", "sync", "export", "status"])
    parser.add_argument("--days", type=int, default=7, help="Days to sync")
    parser.add_argument("--start", type=str, help="Start date for export (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date for export (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="research_export.json")
    
    args = parser.parse_args()
    
    orchestrator = SyncOrchestrator()
    
    if args.command == "setup":
        results = orchestrator.setup()
        print(f"Setup results: {results}")
    
    elif args.command == "sync":
        orchestrator.setup()  # Authenticate before syncing
        results = orchestrator.sync_all(args.days)
        print(f"Sync results: {results}")
    
    elif args.command == "export":
        if not args.start or not args.end:
            print("Export requires --start and --end dates")
        else:
            path = orchestrator.export_research_data(args.start, args.end, args.output)
            print(f"Exported to {path}")
    
    elif args.command == "status":
        print(f"Database: {orchestrator.data_store.db_path}")
        print(f"Credentials path: {orchestrator.garmin.credentials_path}")
