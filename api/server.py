"""
Consciousness Observatory API Server
Serves health and calendar data from SQLite to the React dashboard.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

DB_PATH = Path(__file__).parent.parent / "consciousness_research.db"


@app.route('/')
def index():
    """API root - show available endpoints."""
    return jsonify({
        'name': 'Consciousness Observatory API',
        'endpoints': [
            '/api/stats',
            '/api/health?days=14',
            '/api/calendar?days=14',
            '/api/trends?days=7',
            '/api/daily/<date>'
        ],
        'dashboard': 'http://localhost:5173'
    })


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _get_journal_count(conn):
    """Get total journal entry count."""
    result = conn.execute("SELECT COUNT(*) as c FROM journal_entries").fetchone()
    return result['c'] if result else 0


def _get_latest_journal_sync(conn):
    """Get latest journal entry timestamp."""
    result = conn.execute(
        "SELECT created_at FROM journal_entries ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return result['created_at'] if result else None


def _has_journal_files():
    """Check if journal directory has markdown files."""
    journal_path = Path(__file__).parent.parent / "journal"
    if journal_path.exists():
        md_files = list(journal_path.glob("*.md"))
        return len(md_files) > 0
    return False


@app.route('/api/health')
def get_health_data():
    """Get health snapshots for a date range."""
    days = request.args.get('days', 14, type=int)
    end_date = datetime.now() + timedelta(days=1)  # Include today
    start_date = end_date - timedelta(days=days + 1)

    conn = get_db()
    cursor = conn.execute(
        """SELECT data FROM health_snapshots
           WHERE timestamp >= ? AND timestamp < ?
           ORDER BY timestamp""",
        (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    )

    rows = cursor.fetchall()
    conn.close()

    data = [json.loads(row['data']) for row in rows]
    return jsonify(data)


@app.route('/api/calendar')
def get_calendar_data():
    """Get calendar events for a date range."""
    days = request.args.get('days', 14, type=int)
    end_date = datetime.now() + timedelta(days=7)  # Include future events
    start_date = datetime.now() - timedelta(days=days)

    conn = get_db()
    cursor = conn.execute(
        """SELECT data FROM calendar_events
           WHERE timestamp >= ? AND timestamp <= ?
           ORDER BY timestamp""",
        (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    )

    rows = cursor.fetchall()
    conn.close()

    data = [json.loads(row['data']) for row in rows]
    return jsonify(data)


@app.route('/api/stats')
def get_stats():
    """Get summary statistics."""
    conn = get_db()

    # Count records
    health_count = conn.execute("SELECT COUNT(*) as c FROM health_snapshots").fetchone()['c']
    calendar_count = conn.execute("SELECT COUNT(*) as c FROM calendar_events").fetchone()['c']
    eeg_count = conn.execute("SELECT COUNT(*) as c FROM eeg_sessions").fetchone()['c']

    # Get date range
    health_range = conn.execute(
        "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM health_snapshots"
    ).fetchone()

    # Get latest sync info
    latest_health = conn.execute(
        "SELECT created_at FROM health_snapshots ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    latest_calendar = conn.execute(
        "SELECT created_at FROM calendar_events ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    latest_eeg = conn.execute(
        "SELECT created_at FROM eeg_sessions ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    # Check if EEG fetcher is available and connected
    eeg_data_status = 'pending'
    lsl_status = check_lsl_streams()

    # EEG is connected if: LSL streams available OR fetcher streaming OR has recorded sessions
    if lsl_status.get('available'):
        eeg_data_status = 'connected'
    else:
        fetcher = get_eeg_fetcher()
        if fetcher and fetcher.state.value in ['streaming', 'recording']:
            eeg_data_status = 'connected'
        elif eeg_count > 0:
            eeg_data_status = 'connected'

    # Get journal data before closing connection
    journal_count = _get_journal_count(conn)
    journal_last_sync = _get_latest_journal_sync(conn)

    # Journal is connected if: has synced entries OR has .md files ready to sync
    journal_has_files = _has_journal_files()

    conn.close()

    return jsonify({
        'sources': [
            {
                'id': 'garmin',
                'name': 'Garmin Health',
                'status': 'connected' if health_count > 0 else 'pending',
                'records': health_count,
                'lastSync': latest_health['created_at'] if latest_health else None
            },
            {
                'id': 'calendar',
                'name': 'Google Calendar',
                'status': 'connected' if calendar_count > 0 else 'pending',
                'records': calendar_count,
                'lastSync': latest_calendar['created_at'] if latest_calendar else None
            },
            {
                'id': 'eeg',
                'name': 'Muse EEG',
                'status': eeg_data_status,
                'records': eeg_count,
                'lastSync': latest_eeg['created_at'] if latest_eeg else (
                    datetime.now().isoformat() if eeg_data_status == 'connected' else None
                )
            },
            {
                'id': 'journal',
                'name': 'Journal Entries',
                'status': 'connected' if (journal_count > 0 or journal_has_files) else 'pending',
                'records': journal_count,
                'lastSync': journal_last_sync
            }
        ],
        'dateRange': {
            'start': health_range['min_date'][:10] if health_range['min_date'] else None,
            'end': health_range['max_date'][:10] if health_range['max_date'] else None
        }
    })


@app.route('/api/daily/<date>')
def get_daily_context(date):
    """Get aggregated context for a specific date."""
    conn = get_db()

    # Get health data for the day
    health = conn.execute(
        "SELECT data FROM health_snapshots WHERE timestamp LIKE ?",
        (f"{date}%",)
    ).fetchone()

    # Get calendar events for the day
    next_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    events = conn.execute(
        """SELECT data FROM calendar_events
           WHERE timestamp >= ? AND timestamp < ?""",
        (date, next_date)
    ).fetchall()

    conn.close()

    health_data = json.loads(health['data']) if health else {}
    calendar_data = [json.loads(e['data']) for e in events]

    # Compute context switches (events with < 15 min gaps)
    context_switches = 0
    if len(calendar_data) > 1:
        for i in range(1, len(calendar_data)):
            try:
                prev_end = datetime.fromisoformat(calendar_data[i-1]['end_time'].replace('Z', '+00:00'))
                curr_start = datetime.fromisoformat(calendar_data[i]['start_time'].replace('Z', '+00:00'))
                gap = (curr_start - prev_end).total_seconds() / 60
                if 0 < gap < 15:
                    context_switches += 1
            except:
                pass

    total_meeting_minutes = sum(e.get('duration_minutes', 0) for e in calendar_data)

    return jsonify({
        'date': date,
        'health': health_data,
        'meetings': len(calendar_data),
        'meetingHours': round(total_meeting_minutes / 60, 1),
        'contextSwitches': context_switches,
        'events': calendar_data
    })


@app.route('/api/trends')
def get_trends():
    """Get 7-day trends for key metrics."""
    days = request.args.get('days', 7, type=int)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    conn = get_db()
    cursor = conn.execute(
        """SELECT data FROM health_snapshots
           WHERE timestamp >= ?
           ORDER BY timestamp""",
        (start_date.strftime("%Y-%m-%d"),)
    )

    rows = cursor.fetchall()
    conn.close()

    data = [json.loads(row['data']) for row in rows]

    # Extract metrics
    hrvs = [d.get('hrv_rmssd') for d in data if d.get('hrv_rmssd')]
    stresses = [d.get('stress_level') for d in data if d.get('stress_level')]
    sleeps = [d.get('sleep_score') for d in data if d.get('sleep_score')]

    def calc_trend(values):
        if len(values) < 2:
            return 0
        mid = len(values) // 2
        first_half = sum(values[:mid]) / mid if mid > 0 else 0
        second_half = sum(values[mid:]) / (len(values) - mid) if len(values) > mid else 0
        if first_half == 0:
            return 0
        return round((second_half - first_half) / first_half * 100, 1)

    return jsonify({
        'hrv': {
            'values': hrvs,
            'avg': round(sum(hrvs) / len(hrvs), 1) if hrvs else None,
            'trend': calc_trend(hrvs)
        },
        'stress': {
            'values': stresses,
            'avg': round(sum(stresses) / len(stresses), 1) if stresses else None,
            'trend': calc_trend(stresses)
        },
        'sleep': {
            'values': sleeps,
            'avg': round(sum(sleeps) / len(sleeps), 1) if sleeps else None,
            'trend': calc_trend(sleeps)
        }
    })


# =========================================================================
# EEG Endpoints
# =========================================================================

# Global EEG fetcher instance (lazy initialization)
_eeg_fetcher = None


def get_eeg_fetcher():
    """Get or create EEG fetcher instance."""
    global _eeg_fetcher
    if _eeg_fetcher is None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "data_pipeline"))
            from eeg_fetcher import MuseEEGFetcher
            from fetchers import DataStore
            store = DataStore(str(DB_PATH))
            _eeg_fetcher = MuseEEGFetcher(store)
        except ImportError as e:
            print(f"EEG fetcher not available: {e}")
            return None
    return _eeg_fetcher


def check_lsl_streams():
    """Check if any LSL streams are currently available (OpenMuse running externally)."""
    try:
        from mne_lsl.lsl import resolve_streams
        streams = resolve_streams(timeout=1.0)
        muse_streams = [s for s in streams if 'Muse' in s.name]
        return {
            'available': len(muse_streams) > 0,
            'streams': [{'name': s.name, 'type': s.stype} for s in muse_streams]
        }
    except ImportError:
        return {'available': False, 'error': 'mne_lsl not installed'}
    except Exception as e:
        return {'available': False, 'error': str(e)}


@app.route('/api/eeg/status')
def eeg_status():
    """Get EEG device and recording status."""
    fetcher = get_eeg_fetcher()

    # Also check for external LSL streams (OpenMuse running separately)
    lsl_status = check_lsl_streams()

    if not fetcher:
        # Even without fetcher, we might have external streams
        if lsl_status.get('available'):
            return jsonify({
                'available': True,
                'state': 'streaming',
                'external_streams': True,
                'streams': lsl_status.get('streams', []),
                'device_address': None,
                'current_session': None,
                'signal_quality': {}
            })
        return jsonify({
            'available': False,
            'error': 'EEG module not installed'
        })

    from dataclasses import asdict

    # Determine state - use LSL detection if fetcher is idle
    effective_state = fetcher.state.value
    if effective_state == 'idle' and lsl_status.get('available'):
        effective_state = 'streaming'

    return jsonify({
        'available': True,
        'state': effective_state,
        'device_address': fetcher.device_address,
        'external_streams': lsl_status.get('available', False) and fetcher.state.value == 'idle',
        'streams': lsl_status.get('streams', []),
        'current_session': asdict(fetcher.current_session) if fetcher.current_session else None,
        'signal_quality': fetcher.get_signal_quality() if effective_state in ['streaming', 'recording'] else {}
    })


@app.route('/api/eeg/discover')
def eeg_discover():
    """Discover nearby Muse devices."""
    fetcher = get_eeg_fetcher()
    if not fetcher:
        return jsonify({'devices': [], 'error': 'EEG module not installed'})

    devices = fetcher.discover_devices()
    return jsonify({'devices': devices})


@app.route('/api/eeg/connect', methods=['POST'])
def eeg_connect():
    """Connect to a Muse device."""
    fetcher = get_eeg_fetcher()
    if not fetcher:
        return jsonify({'success': False, 'error': 'EEG module not installed'}), 500

    data = request.get_json() or {}
    address = data.get('address')

    success = fetcher.connect(address)
    return jsonify({
        'success': success,
        'state': fetcher.state.value
    })


@app.route('/api/eeg/disconnect', methods=['POST'])
def eeg_disconnect():
    """Disconnect from Muse device."""
    fetcher = get_eeg_fetcher()
    if not fetcher:
        return jsonify({'success': False, 'error': 'EEG module not installed'}), 500

    fetcher.disconnect()
    return jsonify({'success': True, 'state': fetcher.state.value})


@app.route('/api/eeg/record/start', methods=['POST'])
def eeg_start_recording():
    """Start a new EEG recording session."""
    fetcher = get_eeg_fetcher()
    if not fetcher:
        return jsonify({'success': False, 'error': 'EEG module not installed'}), 500

    data = request.get_json() or {}

    # Check if we have external streams but fetcher is idle - need to connect first
    lsl_status = check_lsl_streams()
    if fetcher.state.value == 'idle' and lsl_status.get('available'):
        # Connect to LSL streams first
        try:
            fetcher.connect_lsl()
        except Exception as e:
            print(f"Failed to connect to LSL: {e}")

    try:
        session_id = fetcher.start_recording(
            labels=data.get('labels', []),
            notes=data.get('notes')
        )
        return jsonify({
            'success': True,
            'session_id': session_id
        })
    except RuntimeError as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/eeg/record/stop', methods=['POST'])
def eeg_stop_recording():
    """Stop current recording."""
    fetcher = get_eeg_fetcher()
    if not fetcher:
        return jsonify({'success': False, 'error': 'EEG module not installed'}), 500

    from dataclasses import asdict

    try:
        session = fetcher.stop_recording()
        return jsonify({
            'success': True,
            'session': asdict(session)
        })
    except RuntimeError as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/eeg/record/marker', methods=['POST'])
def eeg_add_marker():
    """Add a marker to current recording."""
    fetcher = get_eeg_fetcher()
    if not fetcher:
        return jsonify({'success': False, 'error': 'EEG module not installed'}), 500

    data = request.get_json() or {}

    if not data.get('label'):
        return jsonify({'success': False, 'error': 'Label required'}), 400

    fetcher.add_marker(data['label'])
    return jsonify({'success': True})


@app.route('/api/journal/sync', methods=['POST'])
def sync_journal():
    """Sync journal entries from markdown files."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "data_pipeline"))
        from fetchers import JournalFetcher, DataStore

        store = DataStore(str(DB_PATH))
        fetcher = JournalFetcher(store)
        count = fetcher.sync(days_back=30)

        return jsonify({
            'success': True,
            'entries_synced': count
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/journal/entries')
def get_journal_entries():
    """Get journal entries for a date range."""
    days = request.args.get('days', 7, type=int)
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    conn = get_db()
    cursor = conn.execute(
        """SELECT structured_data FROM journal_entries
           WHERE timestamp >= ? AND timestamp < ?
           ORDER BY timestamp DESC""",
        (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    )

    entries = [json.loads(row['structured_data']) for row in cursor.fetchall()]
    conn.close()

    return jsonify(entries)


@app.route('/api/analysis/daily/<date>')
def get_daily_analysis(date):
    """Get comprehensive daily analysis with correlations across all data sources."""
    conn = get_db()

    # Get health data for the day
    health = conn.execute(
        "SELECT data FROM health_snapshots WHERE timestamp LIKE ?",
        (f"{date}%",)
    ).fetchone()

    # Get calendar events for the day
    next_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    events = conn.execute(
        """SELECT data FROM calendar_events
           WHERE timestamp >= ? AND timestamp < ?""",
        (date, next_date)
    ).fetchall()

    # Get EEG sessions for the day
    eeg_sessions_data = conn.execute(
        """SELECT metadata, file_path FROM eeg_sessions
           WHERE start_time >= ? AND start_time < ?
           ORDER BY start_time""",
        (date, next_date)
    ).fetchall()

    # Get journal entries for the day
    journal_entries = conn.execute(
        """SELECT structured_data FROM journal_entries
           WHERE timestamp >= ? AND timestamp < ?
           ORDER BY timestamp""",
        (date, next_date)
    ).fetchall()

    conn.close()

    health_data = json.loads(health['data']) if health else {}
    calendar_data = [json.loads(e['data']) for e in events]
    eeg_sessions_list = [json.loads(s['metadata']) for s in eeg_sessions_data]
    journal_data = [json.loads(j['structured_data']) for j in journal_entries]

    # Compute calendar metrics
    context_switches = 0
    if len(calendar_data) > 1:
        for i in range(1, len(calendar_data)):
            try:
                prev_end = datetime.fromisoformat(calendar_data[i-1]['end_time'].replace('Z', '+00:00'))
                curr_start = datetime.fromisoformat(calendar_data[i]['start_time'].replace('Z', '+00:00'))
                gap = (curr_start - prev_end).total_seconds() / 60
                if 0 < gap < 15:
                    context_switches += 1
            except:
                pass

    total_meeting_minutes = sum(e.get('duration_minutes', 0) for e in calendar_data)

    # Compute EEG summary
    total_eeg_minutes = sum(s.get('duration_seconds', 0) / 60 for s in eeg_sessions_list)

    # Generate insights based on data correlations
    insights = []

    # Sleep quality insight
    sleep_score = health_data.get('sleep_score')
    if sleep_score:
        if sleep_score >= 80:
            insights.append({
                'type': 'positive',
                'category': 'sleep',
                'message': f'Excellent sleep quality ({sleep_score}). Great foundation for cognitive performance.'
            })
        elif sleep_score < 60:
            insights.append({
                'type': 'warning',
                'category': 'sleep',
                'message': f'Sleep score below optimal ({sleep_score}). May impact focus and HRV today.'
            })

    # HRV insight
    hrv = health_data.get('hrv_rmssd')
    if hrv:
        if hrv > 50:
            insights.append({
                'type': 'positive',
                'category': 'hrv',
                'message': f'Strong HRV ({hrv} ms) indicates good recovery and stress resilience.'
            })
        elif hrv < 30:
            insights.append({
                'type': 'warning',
                'category': 'hrv',
                'message': f'Lower HRV ({hrv} ms) - consider lighter cognitive load or meditation today.'
            })

    # Meeting load insight
    if total_meeting_minutes > 240:  # More than 4 hours
        insights.append({
            'type': 'warning',
            'category': 'calendar',
            'message': f'Heavy meeting day ({round(total_meeting_minutes/60, 1)}h). Schedule recovery time.'
        })

    # Context switching insight
    if context_switches > 5:
        insights.append({
            'type': 'warning',
            'category': 'calendar',
            'message': f'High context switching ({context_switches} rapid transitions). May fragment deep work.'
        })

    # EEG recording insight
    if total_eeg_minutes > 0:
        insights.append({
            'type': 'info',
            'category': 'eeg',
            'message': f'Recorded {round(total_eeg_minutes, 1)} minutes of EEG data across {len(eeg_sessions_list)} session(s).'
        })

    # Sleep + HRV correlation
    if sleep_score and hrv:
        if sleep_score >= 75 and hrv >= 40:
            insights.append({
                'type': 'positive',
                'category': 'correlation',
                'message': 'Both sleep and HRV optimal - ideal conditions for peak performance.'
            })
        elif sleep_score < 60 and hrv < 35:
            insights.append({
                'type': 'warning',
                'category': 'correlation',
                'message': 'Both sleep and HRV suboptimal - prioritize recovery today.'
            })

    # Journal insights
    if journal_data:
        # Count mood occurrences
        moods = [j.get('mood') for j in journal_data if j.get('mood')]
        positive_count = moods.count('positive')
        negative_count = moods.count('negative')

        if len(journal_data) > 0:
            insights.append({
                'type': 'info',
                'category': 'journal',
                'message': f'{len(journal_data)} journal entries recorded today.'
            })

        if positive_count > negative_count and positive_count >= 2:
            insights.append({
                'type': 'positive',
                'category': 'journal',
                'message': 'Mostly positive mood noted throughout the day.'
            })
        elif negative_count > positive_count and negative_count >= 2:
            insights.append({
                'type': 'warning',
                'category': 'journal',
                'message': 'Several low-mood entries - check in with yourself.'
            })

        # Extract all tags
        all_tags = []
        for j in journal_data:
            all_tags.extend(j.get('tags', []))
        if 'flow' in all_tags or 'focused' in [j.get('mood') for j in journal_data]:
            insights.append({
                'type': 'positive',
                'category': 'journal',
                'message': 'Flow state noted - great for deep work!'
            })

    return jsonify({
        'date': date,
        'health': {
            'sleep_score': health_data.get('sleep_score'),
            'hrv_rmssd': health_data.get('hrv_rmssd'),
            'stress_level': health_data.get('stress_level'),
            'body_battery': health_data.get('body_battery'),
            'resting_heart_rate': health_data.get('resting_heart_rate'),
            'steps': health_data.get('steps'),
            'deep_sleep_minutes': health_data.get('deep_sleep_minutes'),
            'rem_sleep_minutes': health_data.get('rem_sleep_minutes'),
            'active_minutes': health_data.get('active_minutes')
        },
        'calendar': {
            'meeting_count': len(calendar_data),
            'meeting_hours': round(total_meeting_minutes / 60, 1),
            'context_switches': context_switches,
            'events': calendar_data[:10]  # First 10 events
        },
        'eeg': {
            'session_count': len(eeg_sessions_list),
            'total_minutes': round(total_eeg_minutes, 1),
            'sessions': eeg_sessions_list
        },
        'journal': {
            'entry_count': len(journal_data),
            'entries': journal_data
        },
        'insights': insights,
        'scores': {
            'recovery': _compute_recovery_score(health_data),
            'cognitive_load': _compute_cognitive_load(calendar_data, context_switches),
            'data_richness': _compute_data_richness(health_data, calendar_data, eeg_sessions_list, journal_data)
        }
    })


def _compute_recovery_score(health_data):
    """Compute recovery score from health metrics (0-100)."""
    score = 50  # Base
    if health_data.get('sleep_score'):
        score = health_data['sleep_score']
    if health_data.get('hrv_rmssd'):
        hrv = health_data['hrv_rmssd']
        hrv_bonus = min(20, max(-20, (hrv - 40) / 2))
        score += hrv_bonus
    if health_data.get('body_battery'):
        battery_bonus = (health_data['body_battery'] - 50) / 5
        score += battery_bonus
    return max(0, min(100, round(score)))


def _compute_cognitive_load(calendar_data, context_switches):
    """Compute cognitive load from calendar (0-100, higher = more load)."""
    meeting_minutes = sum(e.get('duration_minutes', 0) for e in calendar_data)
    meeting_score = min(50, meeting_minutes / 10)  # 0-50 based on meeting time
    switch_score = min(30, context_switches * 5)  # 0-30 based on switches
    count_score = min(20, len(calendar_data) * 2)  # 0-20 based on event count
    return round(meeting_score + switch_score + count_score)


def _compute_data_richness(health_data, calendar_data, eeg_sessions, journal_data=None):
    """Compute how much data we have for the day (0-100)."""
    score = 0
    if health_data.get('sleep_score'):
        score += 20
    if health_data.get('hrv_rmssd'):
        score += 20
    if calendar_data:
        score += 20
    if eeg_sessions:
        score += 20
    if journal_data:
        score += 20
    return score


@app.route('/api/eeg/sessions')
def eeg_sessions():
    """Get list of EEG sessions."""
    days = request.args.get('days', 30, type=int)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    conn = get_db()
    cursor = conn.execute(
        """SELECT metadata FROM eeg_sessions
           WHERE start_time >= ? ORDER BY start_time DESC""",
        (start_date.strftime("%Y-%m-%d"),)
    )

    sessions = [json.loads(row['metadata']) for row in cursor.fetchall()]
    conn.close()

    return jsonify(sessions)


@app.route('/api/eeg/sessions/<session_id>')
def eeg_session_detail(session_id):
    """Get details for a specific session."""
    conn = get_db()
    cursor = conn.execute(
        "SELECT metadata, file_path FROM eeg_sessions WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Session not found'}), 404

    session = json.loads(row['metadata'])
    session['file_exists'] = Path(row['file_path']).exists() if row['file_path'] else False

    return jsonify(session)


if __name__ == '__main__':
    print(f"Database: {DB_PATH}")
    print(f"Starting API server on http://localhost:5001")
    app.run(debug=True, port=5001)
