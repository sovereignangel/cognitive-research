"""
WebSocket Server for Real-Time EEG Streaming
=============================================
Streams EEG, PPG, and motion data to dashboard in real-time.
Uses Flask-SocketIO for WebSocket support.
"""

import sys
from pathlib import Path
from threading import Lock
import logging

from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Add data_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data_pipeline"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
eeg_fetcher = None
streaming_clients = set()
stream_lock = Lock()
sample_counter = 0
eeg_buffer = []  # Buffer for band power computation


def get_fetcher():
    """Get or create EEG fetcher instance."""
    global eeg_fetcher
    if eeg_fetcher is None:
        try:
            from eeg_fetcher import MuseEEGFetcher
            from fetchers import DataStore

            db_path = Path(__file__).parent.parent / "consciousness_research.db"
            store = DataStore(str(db_path))
            eeg_fetcher = MuseEEGFetcher(store)
            logger.info("EEG fetcher initialized")
        except ImportError as e:
            logger.error(f"Failed to import EEG fetcher: {e}")
            return None
    return eeg_fetcher


def eeg_sample_callback(sample):
    """Callback that emits EEG samples via WebSocket."""
    global sample_counter, eeg_buffer

    with stream_lock:
        if not streaming_clients:
            return

        sample_counter += 1

        # Store sample in buffer for band power computation
        eeg_buffer.append(sample.eeg_channels)
        if len(eeg_buffer) > 512:  # Keep ~2 seconds at 256 Hz
            eeg_buffer = eeg_buffer[-512:]

        # Decimation: send every 4th sample to reduce bandwidth
        # 256 Hz / 4 = 64 samples/second to dashboard
        if sample_counter % 4 != 0:
            return

    # Emit to all connected clients
    socketio.emit('eeg_sample', {
        'timestamp': sample.timestamp,
        'eeg': sample.eeg_channels,
        'ppg': sample.ppg_channels,
        'motion': sample.motion_channels
    })

    # Compute and emit band powers every 64 samples (~1 second)
    if sample_counter % 64 == 0 and len(eeg_buffer) >= 256:
        try:
            import numpy as np
            from eeg_processing import (
                compute_band_powers,
                compute_engagement_index,
                compute_relaxation_index,
                compute_focus_index,
                compute_meditation_depth
            )

            eeg_array = np.array(eeg_buffer[-256:])

            # Compute band powers (average across channels)
            band_powers = compute_band_powers(eeg_array, sample_rate=256)
            avg_powers = {
                band: float(np.mean(powers))
                for band, powers in band_powers.items()
            }

            # Compute state metrics
            metrics = {
                'engagement': compute_engagement_index(eeg_array, 256),
                'relaxation': compute_relaxation_index(eeg_array, 256),
                'focus': compute_focus_index(eeg_array, 256),
                'meditation': compute_meditation_depth(eeg_array, 256)
            }

            socketio.emit('band_powers', avg_powers)
            socketio.emit('metrics', metrics)

        except Exception as e:
            logger.warning(f"Failed to compute band powers: {e}")


# =========================================================================
# WebSocket Event Handlers
# =========================================================================

@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connection."""
    logger.info(f"Client connected: {request.sid if hasattr(request, 'sid') else 'unknown'}")

    fetcher = get_fetcher()
    emit('status', {
        'connected': True,
        'eeg_available': fetcher is not None,
        'eeg_state': fetcher.state.value if fetcher else 'unavailable'
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    with stream_lock:
        streaming_clients.discard(request.sid if hasattr(request, 'sid') else None)

    logger.info("Client disconnected")


@socketio.on('start_stream')
def handle_start_stream(data=None):
    """Start streaming EEG data to this client."""
    fetcher = get_fetcher()

    if fetcher is None:
        emit('error', {'message': 'EEG module not available'})
        return

    # Auto-connect to LSL streams if available but not connected
    if fetcher.state.value == 'idle':
        try:
            from mne_lsl.lsl import resolve_streams
            streams = resolve_streams(timeout=1.0)
            muse_streams = [s for s in streams if 'Muse' in s.name]
            if muse_streams:
                logger.info("Auto-connecting to LSL streams...")
                fetcher.connect_lsl()
        except Exception as e:
            logger.error(f"Failed to auto-connect to LSL: {e}")

    if fetcher.state.value not in ['streaming', 'recording']:
        emit('error', {'message': f'EEG not streaming (state: {fetcher.state.value})'})
        return

    with stream_lock:
        # Register callback if this is the first client
        if not streaming_clients:
            fetcher.register_callback(eeg_sample_callback)

        streaming_clients.add(request.sid if hasattr(request, 'sid') else 'default')

    logger.info(f"Started streaming to client. Total clients: {len(streaming_clients)}")
    emit('stream_started', {
        'success': True,
        'sample_rate': fetcher.EEG_SAMPLE_RATE // 4,  # Decimated rate
        'channels': fetcher.EEG_CHANNELS
    })


@socketio.on('stop_stream')
def handle_stop_stream():
    """Stop streaming EEG data to this client."""
    fetcher = get_fetcher()

    with stream_lock:
        streaming_clients.discard(request.sid if hasattr(request, 'sid') else 'default')

        # Unregister callback if no more clients
        if not streaming_clients and fetcher:
            fetcher.unregister_callback(eeg_sample_callback)

    logger.info(f"Stopped streaming to client. Remaining clients: {len(streaming_clients)}")
    emit('stream_stopped', {'success': True})


@socketio.on('get_status')
def handle_get_status():
    """Get current EEG status."""
    fetcher = get_fetcher()

    if fetcher is None:
        emit('status', {
            'eeg_available': False,
            'error': 'EEG module not available'
        })
        return

    emit('status', {
        'eeg_available': True,
        'state': fetcher.state.value,
        'device_address': fetcher.device_address,
        'signal_quality': fetcher.get_signal_quality() if fetcher.state.value == 'streaming' else {},
        'is_recording': fetcher.state.value == 'recording',
        'current_session': fetcher.current_session.session_id if fetcher.current_session else None
    })


@socketio.on('get_quality')
def handle_get_quality():
    """Get current signal quality."""
    fetcher = get_fetcher()

    if fetcher and fetcher.state.value in ['streaming', 'recording']:
        emit('signal_quality', fetcher.get_signal_quality())
    else:
        emit('signal_quality', {})


# =========================================================================
# REST Endpoints (for health checks)
# =========================================================================

@app.route('/')
def index():
    """WebSocket server info."""
    return {
        'name': 'Consciousness Observatory WebSocket Server',
        'purpose': 'Real-time EEG streaming',
        'websocket_url': 'ws://localhost:5002',
        'events': {
            'incoming': ['start_stream', 'stop_stream', 'get_status', 'get_quality'],
            'outgoing': ['eeg_sample', 'status', 'signal_quality', 'error']
        }
    }


@app.route('/health')
def health():
    """Health check endpoint."""
    fetcher = get_fetcher()
    return {
        'status': 'ok',
        'eeg_available': fetcher is not None,
        'eeg_state': fetcher.state.value if fetcher else 'unavailable',
        'streaming_clients': len(streaming_clients)
    }


# =========================================================================
# Main
# =========================================================================

# Need to import request for session ID access
from flask import request

if __name__ == '__main__':
    print("=" * 60)
    print("Consciousness Observatory - WebSocket Server")
    print("=" * 60)
    print(f"WebSocket URL: ws://localhost:5002")
    print(f"Health check: http://localhost:5002/health")
    print()
    print("Events:")
    print("  -> start_stream: Begin receiving EEG data")
    print("  -> stop_stream: Stop receiving EEG data")
    print("  -> get_status: Get current EEG state")
    print("  <- eeg_sample: Real-time EEG data")
    print("  <- signal_quality: Channel quality metrics")
    print("=" * 60)

    socketio.run(app, host='0.0.0.0', port=5002, debug=False, allow_unsafe_werkzeug=True)
