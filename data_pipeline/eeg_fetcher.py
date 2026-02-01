"""
Muse S Athena EEG Fetcher
=========================
Real-time EEG streaming and recording from Muse S Athena headband.
Uses OpenMuse for device communication and LSL for data streaming.
"""

import os
import json
import subprocess
import threading
import queue
import time
import re
import uuid
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from enum import Enum

import numpy as np

from fetchers import BaseFetcher, DataStore

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """EEG session lifecycle states."""
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EEGSession:
    """Represents an EEG recording session."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    file_path: Optional[str] = None
    device_address: Optional[str] = None

    # Recording details
    sample_count: int = 0
    duration_seconds: float = 0.0

    # Quality metrics
    signal_quality: Dict[str, float] = field(default_factory=dict)
    dropout_count: int = 0

    # User annotations
    labels: List[Any] = field(default_factory=list)
    notes: Optional[str] = None

    # Linked health data (from Garmin)
    linked_sleep_score: Optional[int] = None
    linked_hrv: Optional[float] = None
    linked_body_battery: Optional[int] = None


@dataclass
class EEGSample:
    """A single multi-modal sample from Muse."""
    timestamp: float
    eeg_channels: List[float]  # TP9, AF7, AF8, TP10
    ppg_channels: Optional[List[float]] = None  # IR, NIR, Red
    motion_channels: Optional[List[float]] = None  # ACC_X/Y/Z, GYRO_X/Y/Z


class MuseEEGFetcher(BaseFetcher):
    """
    Fetches and records EEG data from Muse S Athena headband.

    Uses OpenMuse for device discovery and LSL streaming,
    mne_lsl for stream ingestion.

    Unlike other fetchers, this handles real-time streaming data
    with session-based recording.
    """

    # LSL Stream name patterns created by OpenMuse
    # Note: OpenMuse includes device address in stream name, e.g. "Muse-EEG (ADDRESS)"
    STREAM_PATTERNS = {
        'eeg': 'Muse-EEG',
        'motion': 'Muse-ACCGYRO',
        'optics': 'Muse-OPTICS',
        'battery': 'Muse-BATTERY'
    }

    # Sampling rates
    EEG_SAMPLE_RATE = 256  # Hz
    MOTION_SAMPLE_RATE = 52  # Hz
    OPTICS_SAMPLE_RATE = 64  # Hz

    # Channel names
    EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
    MOTION_CHANNELS = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']
    PPG_CHANNELS = ['IR', 'NIR', 'Red']

    def __init__(self, data_store: DataStore):
        super().__init__(data_store)

        # Device state
        self.device_address: Optional[str] = None
        self.state = SessionState.IDLE

        # Current session
        self.current_session: Optional[EEGSession] = None

        # Streaming components
        self._stream_process: Optional[subprocess.Popen] = None
        self._receiver_threads: Dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()

        # Data buffers (thread-safe queues)
        self._eeg_buffer = queue.Queue(maxsize=10000)  # ~40 seconds at 256 Hz
        self._motion_buffer = queue.Queue(maxsize=2000)
        self._ppg_buffer = queue.Queue(maxsize=2500)

        # Recording buffers (lists for file storage)
        self._recording_eeg: List[np.ndarray] = []
        self._recording_motion: List[np.ndarray] = []
        self._recording_ppg: List[np.ndarray] = []
        self._timestamps_eeg: List[float] = []
        self._timestamps_motion: List[float] = []
        self._timestamps_ppg: List[float] = []

        # Callbacks for real-time data
        self._sample_callbacks: List[Callable[[EEGSample], None]] = []

        # File paths
        self.eeg_data_path = Path.home() / ".consciousness_research" / "eeg_data"
        self.eeg_data_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # BaseFetcher Interface Implementation
    # =========================================================================

    def authenticate(self) -> bool:
        """
        For EEG, 'authentication' means discovering and pairing with device.
        Returns True if a Muse device is found.
        """
        try:
            devices = self.discover_devices()
            if devices:
                self.device_address = devices[0]
                logger.info(f"Found Muse device: {self.device_address}")
                return True
            logger.warning("No Muse devices found")
            return False
        except Exception as e:
            logger.error(f"Device discovery failed: {e}")
            return False

    def fetch(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        For EEG, 'fetch' returns metadata for sessions in the date range.
        Actual EEG data is stored in files, not fetched from external API.
        """
        sessions = self._get_sessions_in_range(start_date, end_date)
        return [asdict(s) for s in sessions]

    def sync(self) -> int:
        """
        For EEG, 'sync' verifies file integrity and updates session metadata.
        Returns count of validated sessions.
        """
        return self._validate_session_files()

    # =========================================================================
    # Device Discovery and Connection
    # =========================================================================

    def discover_devices(self, timeout: int = 10) -> List[str]:
        """
        Discover nearby Muse devices using OpenMuse.
        Returns list of MAC addresses.
        """
        try:
            result = subprocess.run(
                ['OpenMuse', 'find'],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            addresses = self._parse_device_addresses(result.stdout)
            logger.info(f"Discovered {len(addresses)} Muse device(s)")
            return addresses
        except subprocess.TimeoutExpired:
            logger.warning("Device discovery timed out")
            return []
        except FileNotFoundError:
            logger.error("OpenMuse not installed. Run: pip install OpenMuse")
            return []

    def connect(self, address: Optional[str] = None) -> bool:
        """
        Connect to a Muse device and start LSL streaming.
        """
        if self.state not in [SessionState.IDLE, SessionState.ERROR]:
            logger.warning(f"Cannot connect in state: {self.state}")
            return False

        address = address or self.device_address
        if not address:
            logger.error("No device address specified")
            return False

        self.state = SessionState.CONNECTING

        try:
            # Start OpenMuse stream process
            self._stream_process = subprocess.Popen(
                ['OpenMuse', 'stream', '--address', address],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for LSL streams to become available
            if not self._wait_for_streams(timeout=15):
                raise ConnectionError("LSL streams not available")

            # Start receiver threads for all streams
            self._stop_event.clear()
            self._start_receiver_threads()

            self.device_address = address
            self.state = SessionState.STREAMING
            logger.info(f"Connected to Muse: {address}")
            return True

        except Exception as e:
            self.state = SessionState.ERROR
            logger.error(f"Connection failed: {e}")
            self.disconnect()
            return False

    def connect_lsl(self) -> bool:
        """
        Connect to existing LSL streams (when OpenMuse is running externally).
        Use this when LSL streams are already available but device wasn't
        connected through this fetcher.
        """
        if self.state not in [SessionState.IDLE, SessionState.ERROR]:
            logger.warning(f"Cannot connect in state: {self.state}")
            return False

        self.state = SessionState.CONNECTING

        try:
            # Check that streams are available
            if not self._wait_for_streams(timeout=5):
                raise ConnectionError("LSL streams not available")

            # Start receiver threads for all streams
            self._stop_event.clear()
            self._start_receiver_threads()

            self.state = SessionState.STREAMING
            logger.info("Connected to existing LSL streams")
            return True

        except Exception as e:
            self.state = SessionState.ERROR
            logger.error(f"LSL connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from device and clean up resources."""
        self._stop_event.set()

        # Wait for receiver threads
        for name, thread in self._receiver_threads.items():
            if thread.is_alive():
                thread.join(timeout=5)
        self._receiver_threads.clear()

        # Stop OpenMuse process
        if self._stream_process:
            self._stream_process.terminate()
            try:
                self._stream_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._stream_process.kill()
            self._stream_process = None

        self.state = SessionState.IDLE
        logger.info("Disconnected from Muse")

    # =========================================================================
    # Session Management
    # =========================================================================

    def start_recording(self, labels: List[str] = None, notes: str = None) -> str:
        """
        Start recording EEG data to a new session.
        Returns session_id.
        """
        if self.state != SessionState.STREAMING:
            raise RuntimeError(f"Cannot start recording in state: {self.state}")

        session_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        self.current_session = EEGSession(
            session_id=session_id,
            start_time=start_time.isoformat(),
            device_address=self.device_address,
            labels=labels or [],
            notes=notes
        )

        # Link to today's health data
        self._link_health_data(self.current_session)

        # Clear recording buffers
        self._recording_eeg = []
        self._recording_motion = []
        self._recording_ppg = []
        self._timestamps_eeg = []
        self._timestamps_motion = []
        self._timestamps_ppg = []

        # Store session metadata in database
        self._save_session_metadata(self.current_session)

        self.state = SessionState.RECORDING
        logger.info(f"Started recording session: {session_id}")

        return session_id

    def stop_recording(self) -> EEGSession:
        """
        Stop recording and save data to file.
        Returns the completed session.
        """
        if self.state not in [SessionState.RECORDING, SessionState.PAUSED]:
            raise RuntimeError(f"Cannot stop recording in state: {self.state}")

        self.state = SessionState.STOPPING

        # Finalize session
        self.current_session.end_time = datetime.now().isoformat()
        self.current_session.sample_count = len(self._recording_eeg)
        self.current_session.duration_seconds = (
            self.current_session.sample_count / self.EEG_SAMPLE_RATE
        )

        # Save data to file
        file_path = self._save_recording_data()
        self.current_session.file_path = str(file_path)

        # Compute signal quality metrics
        self.current_session.signal_quality = self._compute_signal_quality()

        # Update database
        self._update_session_metadata(self.current_session)

        completed_session = self.current_session
        self.current_session = None
        self.state = SessionState.STREAMING

        logger.info(f"Stopped recording. Duration: {completed_session.duration_seconds:.1f}s")
        return completed_session

    def pause_recording(self):
        """Pause current recording (data continues to stream but not saved)."""
        if self.state == SessionState.RECORDING:
            self.state = SessionState.PAUSED
            logger.info("Recording paused")

    def resume_recording(self):
        """Resume paused recording."""
        if self.state == SessionState.PAUSED:
            self.state = SessionState.RECORDING
            logger.info("Recording resumed")

    def add_marker(self, label: str, timestamp: float = None):
        """
        Add a timestamped marker/label to current recording.
        Useful for marking events during EEG recording.
        """
        if self.current_session:
            marker = {
                'label': label,
                'timestamp': timestamp or time.time(),
                'sample_index': len(self._recording_eeg)
            }
            self.current_session.labels.append(marker)
            logger.debug(f"Added marker: {label}")

    # =========================================================================
    # Real-time Data Access
    # =========================================================================

    def register_callback(self, callback: Callable[[EEGSample], None]):
        """
        Register a callback for real-time sample delivery.
        Callbacks receive EEGSample objects as data arrives.
        """
        self._sample_callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[EEGSample], None]):
        """Remove a previously registered callback."""
        if callback in self._sample_callbacks:
            self._sample_callbacks.remove(callback)

    def get_latest_samples(self, n_samples: int = 256) -> np.ndarray:
        """
        Get the most recent N EEG samples from the buffer.
        Returns array of shape (n_samples, n_channels).
        """
        samples = []
        try:
            while len(samples) < n_samples:
                sample = self._eeg_buffer.get_nowait()
                samples.append(sample)
        except queue.Empty:
            pass

        if samples:
            return np.array(samples)
        return np.zeros((0, len(self.EEG_CHANNELS)))

    def get_signal_quality(self) -> Dict[str, float]:
        """
        Get current signal quality for each EEG channel.
        Returns dict mapping channel name to quality score (0-100).
        """
        # Get recent samples without removing from buffer
        samples = []
        temp_storage = []
        try:
            while len(samples) < 256:
                sample = self._eeg_buffer.get_nowait()
                samples.append(sample)
                temp_storage.append(sample)
        except queue.Empty:
            pass

        # Put samples back
        for s in temp_storage:
            try:
                self._eeg_buffer.put_nowait(s)
            except queue.Full:
                break

        if len(samples) < 10:
            return {ch: 0 for ch in self.EEG_CHANNELS}

        recent = np.array(samples)
        quality = {}

        for i, channel in enumerate(self.EEG_CHANNELS):
            channel_data = recent[:, i]
            std = float(np.std(channel_data))

            # Quality heuristics
            if std < 1:  # Flat signal (bad contact)
                quality[channel] = 0.0
            elif std > 200:  # Too much noise
                quality[channel] = float(max(0, 100 - (std - 200) / 10))
            else:
                quality[channel] = float(min(100, 50 + std / 2))

        return quality

    # =========================================================================
    # Private Methods: Streaming
    # =========================================================================

    def _start_receiver_threads(self):
        """Start background threads for receiving all LSL streams."""
        # EEG stream (primary)
        eeg_thread = threading.Thread(
            target=self._stream_receiver_loop,
            args=('eeg', self.STREAM_PATTERNS['eeg'], self._eeg_buffer,
                  self._recording_eeg, self._timestamps_eeg),
            daemon=True
        )
        eeg_thread.start()
        self._receiver_threads['eeg'] = eeg_thread

        # Motion stream
        motion_thread = threading.Thread(
            target=self._stream_receiver_loop,
            args=('motion', self.STREAM_PATTERNS['motion'], self._motion_buffer,
                  self._recording_motion, self._timestamps_motion),
            daemon=True
        )
        motion_thread.start()
        self._receiver_threads['motion'] = motion_thread

        # PPG/Optics stream
        ppg_thread = threading.Thread(
            target=self._stream_receiver_loop,
            args=('ppg', self.STREAM_PATTERNS['optics'], self._ppg_buffer,
                  self._recording_ppg, self._timestamps_ppg),
            daemon=True
        )
        ppg_thread.start()
        self._receiver_threads['ppg'] = ppg_thread

    def _stream_receiver_loop(self, stream_type: str, stream_pattern: str,
                               buffer: queue.Queue, recording_list: List,
                               timestamps_list: List):
        """Background thread that receives LSL data for a specific stream."""
        try:
            from mne_lsl.lsl import StreamInlet, resolve_streams
        except ImportError:
            logger.error("mne_lsl not installed. Run: pip install mne-lsl")
            return

        try:
            # Resolve stream by pattern (OpenMuse includes address in name)
            streams = resolve_streams(timeout=10)
            target_stream = None
            for s in streams:
                if stream_pattern in s.name:
                    target_stream = s
                    break

            if not target_stream:
                logger.warning(f"{stream_type} stream matching '{stream_pattern}' not found")
                return

            inlet = StreamInlet(target_stream)
            inlet.open_stream()

            logger.info(f"LSL receiver started for {stream_type}: {target_stream.name}")

            while not self._stop_event.is_set():
                # Pull available samples
                samples, timestamps = inlet.pull_chunk(timeout=0.1)

                if samples is not None and len(samples) > 0:
                    for i in range(len(samples)):
                        sample = samples[i]
                        ts = timestamps[i] if i < len(timestamps) else time.time()

                        # Add to real-time buffer (non-blocking)
                        try:
                            buffer.put_nowait(sample)
                        except queue.Full:
                            try:
                                buffer.get_nowait()
                                buffer.put_nowait(sample)
                            except queue.Empty:
                                pass

                        # If recording, add to recording buffer
                        if self.state == SessionState.RECORDING:
                            recording_list.append(sample)
                            timestamps_list.append(ts)

                        # Notify callbacks (only for EEG for now)
                        if stream_type == 'eeg':
                            eeg_sample = EEGSample(
                                timestamp=float(ts),
                                eeg_channels=sample[:4].tolist() if hasattr(sample, 'tolist') else list(sample[:4])
                            )
                            for callback in self._sample_callbacks:
                                try:
                                    callback(eeg_sample)
                                except Exception as e:
                                    logger.warning(f"Callback error: {e}")

            inlet.close_stream()

        except Exception as e:
            logger.error(f"Stream receiver error ({stream_type}): {e}")
            if self.state == SessionState.STREAMING:
                self.state = SessionState.ERROR

    def _wait_for_streams(self, timeout: int = 15) -> bool:
        """Wait for LSL streams to become available."""
        try:
            from mne_lsl.lsl import resolve_streams
        except ImportError:
            logger.error("mne_lsl not installed")
            return False

        start = time.time()
        while time.time() - start < timeout:
            streams = resolve_streams(timeout=2)
            # Find stream matching our pattern (e.g., "Muse-EEG (ADDRESS)")
            for s in streams:
                if self.STREAM_PATTERNS['eeg'] in s.name:
                    logger.info(f"LSL EEG stream found: {s.name}")
                    return True
            time.sleep(0.5)

        logger.warning("Timeout waiting for LSL streams")
        return False

    # =========================================================================
    # Private Methods: Data Storage
    # =========================================================================

    def _save_recording_data(self) -> Path:
        """Save recording buffer to .npz file."""
        session = self.current_session

        # Create filename: YYYYMMDD_HHMMSS_sessionid.npz
        timestamp = datetime.fromisoformat(session.start_time)
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{session.session_id}.npz"
        file_path = self.eeg_data_path / filename

        # Convert buffers to numpy arrays
        save_dict = {
            'session_id': session.session_id,
            'start_time': session.start_time,
            'device_address': session.device_address or '',
            'eeg_channels': self.EEG_CHANNELS,
            'eeg_sample_rate': self.EEG_SAMPLE_RATE,
        }

        # EEG data
        if self._recording_eeg:
            save_dict['eeg'] = np.array(self._recording_eeg)
            save_dict['eeg_timestamps'] = np.array(self._timestamps_eeg)

        # Motion data
        if self._recording_motion:
            save_dict['motion'] = np.array(self._recording_motion)
            save_dict['motion_timestamps'] = np.array(self._timestamps_motion)
            save_dict['motion_channels'] = self.MOTION_CHANNELS
            save_dict['motion_sample_rate'] = self.MOTION_SAMPLE_RATE

        # PPG/Optics data
        if self._recording_ppg:
            save_dict['ppg'] = np.array(self._recording_ppg)
            save_dict['ppg_timestamps'] = np.array(self._timestamps_ppg)
            save_dict['ppg_channels'] = self.PPG_CHANNELS
            save_dict['ppg_sample_rate'] = self.OPTICS_SAMPLE_RATE

        # Markers
        if session.labels:
            save_dict['markers'] = json.dumps(session.labels)

        # Linked health data
        save_dict['linked_sleep_score'] = session.linked_sleep_score or -1
        save_dict['linked_hrv'] = session.linked_hrv or -1
        save_dict['linked_body_battery'] = session.linked_body_battery or -1

        np.savez_compressed(file_path, **save_dict)

        logger.info(f"Saved EEG data to {file_path}")
        return file_path

    def _link_health_data(self, session: EEGSession):
        """Link session to today's health data from Garmin."""
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        health_data = self.data_store.get_health_data(today, tomorrow)

        if health_data:
            latest = health_data[-1] if health_data else health_data[0]
            session.linked_sleep_score = latest.get('sleep_score')
            session.linked_hrv = latest.get('hrv_rmssd')
            session.linked_body_battery = latest.get('body_battery')
            logger.info(f"Linked health data: sleep={session.linked_sleep_score}, "
                       f"hrv={session.linked_hrv}, battery={session.linked_body_battery}")

    def _save_session_metadata(self, session: EEGSession):
        """Save session metadata to database."""
        self.data_store.conn.execute(
            """INSERT INTO eeg_sessions
               (session_id, start_time, metadata)
               VALUES (?, ?, ?)""",
            (session.session_id, session.start_time, json.dumps(asdict(session)))
        )
        self.data_store.conn.commit()

    def _update_session_metadata(self, session: EEGSession):
        """Update session metadata in database."""
        self.data_store.conn.execute(
            """UPDATE eeg_sessions
               SET end_time = ?, file_path = ?, metadata = ?
               WHERE session_id = ?""",
            (session.end_time, session.file_path,
             json.dumps(asdict(session)), session.session_id)
        )
        self.data_store.conn.commit()

    def _get_sessions_in_range(self, start: datetime, end: datetime) -> List[EEGSession]:
        """Retrieve sessions within date range."""
        cursor = self.data_store.conn.execute(
            """SELECT metadata FROM eeg_sessions
               WHERE start_time >= ? AND start_time <= ?
               ORDER BY start_time""",
            (start.isoformat(), end.isoformat())
        )
        sessions = []
        for row in cursor.fetchall():
            data = json.loads(row['metadata'])
            sessions.append(EEGSession(**data))
        return sessions

    def _validate_session_files(self) -> int:
        """Validate that session files exist and are readable."""
        cursor = self.data_store.conn.execute(
            "SELECT session_id, file_path FROM eeg_sessions WHERE file_path IS NOT NULL"
        )
        valid_count = 0
        for row in cursor.fetchall():
            file_path = Path(row['file_path'])
            if file_path.exists():
                try:
                    np.load(file_path, allow_pickle=True)
                    valid_count += 1
                except Exception as e:
                    logger.warning(f"Invalid file {file_path}: {e}")
        return valid_count

    def _compute_signal_quality(self) -> Dict[str, float]:
        """Compute overall signal quality for recorded session."""
        if not self._recording_eeg:
            return {}

        data = np.array(self._recording_eeg)
        quality = {}

        for i, channel in enumerate(self.EEG_CHANNELS):
            channel_data = data[:, i]
            # Percentage of good samples (not artifacts)
            artifacts = np.abs(channel_data) > 200  # microvolts threshold
            quality[channel] = float(100 * (1 - np.mean(artifacts)))

        return quality

    @staticmethod
    def _parse_device_addresses(output: str) -> List[str]:
        """Parse MAC addresses from OpenMuse find output."""
        pattern = r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}'
        return re.findall(pattern, output)

    # =========================================================================
    # Data Loading Utilities
    # =========================================================================

    @staticmethod
    def load_session_data(file_path: str) -> Dict[str, Any]:
        """
        Load EEG data from a saved session file.

        Returns dict with:
            - 'eeg': numpy array of shape (n_samples, n_channels)
            - 'eeg_timestamps': numpy array of timestamps
            - 'eeg_channels': list of channel names
            - 'eeg_sample_rate': sampling rate in Hz
            - 'session_id': session identifier
            - 'motion': motion data (if available)
            - 'ppg': PPG/fNIRS data (if available)
            - 'linked_sleep_score': sleep score from same day
        """
        data = np.load(file_path, allow_pickle=True)

        result = {
            'session_id': str(data['session_id']),
            'start_time': str(data['start_time']),
            'eeg_sample_rate': int(data['eeg_sample_rate']),
            'eeg_channels': list(data['eeg_channels']),
        }

        # EEG data
        if 'eeg' in data:
            result['eeg'] = data['eeg']
            result['eeg_timestamps'] = data['eeg_timestamps']

        # Motion data
        if 'motion' in data:
            result['motion'] = data['motion']
            result['motion_timestamps'] = data['motion_timestamps']
            result['motion_channels'] = list(data['motion_channels'])

        # PPG data
        if 'ppg' in data:
            result['ppg'] = data['ppg']
            result['ppg_timestamps'] = data['ppg_timestamps']
            result['ppg_channels'] = list(data['ppg_channels'])

        # Linked health data
        linked_sleep = int(data.get('linked_sleep_score', -1))
        result['linked_sleep_score'] = linked_sleep if linked_sleep >= 0 else None

        linked_hrv = float(data.get('linked_hrv', -1))
        result['linked_hrv'] = linked_hrv if linked_hrv >= 0 else None

        linked_battery = int(data.get('linked_body_battery', -1))
        result['linked_body_battery'] = linked_battery if linked_battery >= 0 else None

        # Markers
        if 'markers' in data:
            result['markers'] = json.loads(str(data['markers']))

        return result


# ============================================================================
# CLI Interface for Testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Muse EEG Fetcher CLI")
    parser.add_argument("command", choices=["find", "connect", "record", "status"])
    parser.add_argument("--address", type=str, help="Muse MAC address")
    parser.add_argument("--duration", type=int, default=30, help="Recording duration in seconds")

    args = parser.parse_args()

    store = DataStore()
    fetcher = MuseEEGFetcher(store)

    if args.command == "find":
        devices = fetcher.discover_devices()
        print(f"Found {len(devices)} device(s):")
        for addr in devices:
            print(f"  {addr}")

    elif args.command == "connect":
        if not args.address:
            print("Discovering devices...")
            devices = fetcher.discover_devices()
            if not devices:
                print("No devices found")
                exit(1)
            args.address = devices[0]

        print(f"Connecting to {args.address}...")
        if fetcher.connect(args.address):
            print("Connected! Press Ctrl+C to disconnect.")
            try:
                while True:
                    quality = fetcher.get_signal_quality()
                    print(f"Signal quality: {quality}")
                    time.sleep(2)
            except KeyboardInterrupt:
                pass
            finally:
                fetcher.disconnect()
        else:
            print("Connection failed")

    elif args.command == "record":
        if not args.address:
            devices = fetcher.discover_devices()
            if not devices:
                print("No devices found")
                exit(1)
            args.address = devices[0]

        print(f"Connecting to {args.address}...")
        if fetcher.connect(args.address):
            print(f"Recording for {args.duration} seconds...")
            session_id = fetcher.start_recording()
            time.sleep(args.duration)
            session = fetcher.stop_recording()
            print(f"Session saved: {session.file_path}")
            print(f"Duration: {session.duration_seconds:.1f}s")
            print(f"Samples: {session.sample_count}")
            print(f"Sleep score: {session.linked_sleep_score}")
            fetcher.disconnect()
        else:
            print("Connection failed")

    elif args.command == "status":
        print(f"State: {fetcher.state.value}")
        print(f"Device: {fetcher.device_address or 'None'}")
        print(f"Data path: {fetcher.eeg_data_path}")
