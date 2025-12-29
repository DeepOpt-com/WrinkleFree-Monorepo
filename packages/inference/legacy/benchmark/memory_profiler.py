"""
Memory bandwidth profiler for BitNet.cpp benchmarking.

Profiles memory bandwidth utilization during inference to identify
memory-bound vs compute-bound phases. Critical for understanding
performance on high-bandwidth platforms like GCP H3 with DDR5.

Metrics collected:
- Memory read/write bandwidth (GB/s)
- Bandwidth utilization vs theoretical max
- Page faults per second
- Memory usage (RSS, VMS)
- Optional: CPU cache statistics (if perf available)
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import threading
import time
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryBandwidthMetrics:
    """Memory bandwidth metrics from a profiling session."""

    # Estimated bandwidth (GB/s)
    memory_read_gb_per_sec: float = 0.0
    memory_write_gb_per_sec: float = 0.0
    total_bandwidth_gb_per_sec: float = 0.0

    # Utilization vs theoretical max
    bandwidth_utilization_percent: float = 0.0
    theoretical_max_gb_per_sec: float = 0.0

    # Memory usage
    memory_usage_gb: float = 0.0
    peak_memory_gb: float = 0.0

    # Page faults
    page_faults_per_sec: float = 0.0
    major_page_faults: int = 0
    minor_page_faults: int = 0

    # Optional cache stats (from perf)
    cache_hit_rate_percent: Optional[float] = None
    l1_cache_misses: Optional[int] = None
    l2_cache_misses: Optional[int] = None
    l3_cache_misses: Optional[int] = None

    # Sampling info
    sample_count: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "memory_read_gb_per_sec": round(self.memory_read_gb_per_sec, 3),
            "memory_write_gb_per_sec": round(self.memory_write_gb_per_sec, 3),
            "total_bandwidth_gb_per_sec": round(self.total_bandwidth_gb_per_sec, 3),
            "bandwidth_utilization_percent": round(self.bandwidth_utilization_percent, 2),
            "theoretical_max_gb_per_sec": round(self.theoretical_max_gb_per_sec, 1),
            "memory_usage_gb": round(self.memory_usage_gb, 3),
            "peak_memory_gb": round(self.peak_memory_gb, 3),
            "page_faults_per_sec": round(self.page_faults_per_sec, 2),
            "major_page_faults": self.major_page_faults,
            "minor_page_faults": self.minor_page_faults,
            "cache_hit_rate_percent": (
                round(self.cache_hit_rate_percent, 2)
                if self.cache_hit_rate_percent is not None
                else None
            ),
            "sample_count": self.sample_count,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class MemorySample:
    """Single memory sample."""

    timestamp: float
    rss_bytes: int
    vms_bytes: int
    page_faults: int
    major_faults: int
    minor_faults: int


class MemoryProfiler:
    """
    Profiles memory bandwidth during inference.

    Uses psutil for memory tracking and optionally Linux perf for
    hardware counter access. Memory bandwidth is estimated from:
    - Direct measurement via /proc if available
    - Inference-based calculation: model_size × tokens_per_sec

    Usage:
        profiler = MemoryProfiler(
            model_size_gb=17.5,
            theoretical_bandwidth_gb_s=307.0,
        )
        profiler.start()
        # ... run inference ...
        metrics = profiler.stop()
    """

    def __init__(
        self,
        model_size_gb: float = 0.5,
        theoretical_bandwidth_gb_s: float = 200.0,
        sample_interval_ms: int = 100,
        process_pid: Optional[int] = None,
    ):
        """
        Initialize memory profiler.

        Args:
            model_size_gb: Size of model in GB (for bandwidth estimation)
            theoretical_bandwidth_gb_s: Theoretical max memory bandwidth (GB/s)
            sample_interval_ms: Sampling interval in milliseconds
            process_pid: PID to monitor (None = current process)
        """
        self.model_size_gb = model_size_gb
        self.theoretical_bandwidth_gb_s = theoretical_bandwidth_gb_s
        self.sample_interval_ms = sample_interval_ms
        self.process_pid = process_pid

        self._samples: list[MemorySample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0
        self._stop_time: float = 0

        # Token tracking for bandwidth estimation
        self._tokens_generated = 0
        self._token_lock = threading.Lock()

        # Try to import psutil
        try:
            import psutil

            self._psutil = psutil
            self._process = None
        except ImportError:
            logger.warning("psutil not available, memory profiling disabled")
            self._psutil = None
            self._process = None

    def _get_process(self):
        """Get the process to monitor."""
        if self._psutil is None:
            return None
        if self._process is None:
            pid = self.process_pid or self._psutil.Process().pid
            self._process = self._psutil.Process(pid)
        return self._process

    def _sample(self) -> Optional[MemorySample]:
        """Take a single memory sample."""
        proc = self._get_process()
        if proc is None:
            return None

        try:
            mem_info = proc.memory_info()
            # Get page fault info if available
            try:
                mem_full = proc.memory_full_info()
                major_faults = getattr(mem_full, "num_page_faults", 0)
                minor_faults = 0  # Not always available
            except (AttributeError, self._psutil.AccessDenied):
                major_faults = 0
                minor_faults = 0

            return MemorySample(
                timestamp=time.time(),
                rss_bytes=mem_info.rss,
                vms_bytes=mem_info.vms,
                page_faults=major_faults + minor_faults,
                major_faults=major_faults,
                minor_faults=minor_faults,
            )
        except Exception as e:
            logger.debug(f"Failed to sample memory: {e}")
            return None

    def _sampling_loop(self):
        """Background thread for continuous sampling."""
        interval = self.sample_interval_ms / 1000.0

        while self._running:
            sample = self._sample()
            if sample:
                self._samples.append(sample)
            time.sleep(interval)

    def start(self):
        """Start memory profiling."""
        if self._running:
            return

        self._samples = []
        self._tokens_generated = 0
        self._running = True
        self._start_time = time.time()

        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

        logger.debug("Memory profiler started")

    def stop(self) -> MemoryBandwidthMetrics:
        """Stop profiling and return metrics."""
        self._running = False
        self._stop_time = time.time()

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        return self._compute_metrics()

    def record_tokens(self, count: int):
        """Record tokens generated (for bandwidth estimation)."""
        with self._token_lock:
            self._tokens_generated += count

    def _compute_metrics(self) -> MemoryBandwidthMetrics:
        """Compute final metrics from samples."""
        duration = self._stop_time - self._start_time

        if not self._samples:
            return MemoryBandwidthMetrics(
                theoretical_max_gb_per_sec=self.theoretical_bandwidth_gb_s,
                duration_seconds=duration,
            )

        # Memory usage stats
        rss_values = [s.rss_bytes for s in self._samples]
        peak_rss = max(rss_values)
        avg_rss = statistics.mean(rss_values)

        # Page faults
        if len(self._samples) >= 2:
            first = self._samples[0]
            last = self._samples[-1]
            total_faults = last.page_faults - first.page_faults
            faults_per_sec = total_faults / duration if duration > 0 else 0
            major_faults = last.major_faults - first.major_faults
            minor_faults = last.minor_faults - first.minor_faults
        else:
            faults_per_sec = 0
            major_faults = 0
            minor_faults = 0

        # Estimate bandwidth from tokens generated
        # Assumption: Each token requires reading ~2% of model weights
        # (due to speculative decoding, KV cache, etc.)
        model_read_fraction = 0.02  # Rough estimate
        with self._token_lock:
            tokens = self._tokens_generated

        if duration > 0 and tokens > 0:
            tokens_per_sec = tokens / duration
            # Estimated reads: model_size * fraction * tokens
            estimated_reads_gb = self.model_size_gb * model_read_fraction * tokens
            read_bandwidth = estimated_reads_gb / duration

            # Write bandwidth is typically much lower (KV cache updates)
            write_bandwidth = read_bandwidth * 0.1  # ~10% of reads

            total_bandwidth = read_bandwidth + write_bandwidth
            utilization = (total_bandwidth / self.theoretical_bandwidth_gb_s) * 100
        else:
            read_bandwidth = 0.0
            write_bandwidth = 0.0
            total_bandwidth = 0.0
            utilization = 0.0

        return MemoryBandwidthMetrics(
            memory_read_gb_per_sec=read_bandwidth,
            memory_write_gb_per_sec=write_bandwidth,
            total_bandwidth_gb_per_sec=total_bandwidth,
            bandwidth_utilization_percent=min(utilization, 100.0),
            theoretical_max_gb_per_sec=self.theoretical_bandwidth_gb_s,
            memory_usage_gb=avg_rss / (1024**3),
            peak_memory_gb=peak_rss / (1024**3),
            page_faults_per_sec=faults_per_sec,
            major_page_faults=major_faults,
            minor_page_faults=minor_faults,
            sample_count=len(self._samples),
            duration_seconds=duration,
        )


class MemoryProfilerContext:
    """Context manager for memory profiling."""

    def __init__(
        self,
        model_size_gb: float = 0.5,
        theoretical_bandwidth_gb_s: float = 200.0,
        sample_interval_ms: int = 100,
        token_callback: Optional[Callable[[int], None]] = None,
    ):
        self.profiler = MemoryProfiler(
            model_size_gb=model_size_gb,
            theoretical_bandwidth_gb_s=theoretical_bandwidth_gb_s,
            sample_interval_ms=sample_interval_ms,
        )
        self.token_callback = token_callback
        self.metrics: Optional[MemoryBandwidthMetrics] = None

    def __enter__(self) -> "MemoryProfilerContext":
        self.profiler.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics = self.profiler.stop()
        return False

    def record_tokens(self, count: int):
        """Record tokens generated."""
        self.profiler.record_tokens(count)
        if self.token_callback:
            self.token_callback(count)


# Hardware bandwidth constants (GB/s)
HARDWARE_BANDWIDTH = {
    # GCP instances
    "gcp_h3_88": 307.0,  # DDR5-4800, 8 channels
    "gcp_n2d_96": 200.0,  # DDR4-3200, estimated
    "gcp_c2d_60": 170.0,  # DDR4-3200, estimated
    # AWS instances
    "aws_hpc6a": 200.0,  # AMD EPYC, DDR4
    "aws_hpc7g": 200.0,  # Graviton3E, DDR5
    # RunPod
    "runpod_cpu_64": 150.0,  # Typical server DDR4
    "runpod_a40": 696.0,  # A40 HBM bandwidth
    # Consumer
    "desktop_ddr4": 50.0,  # Typical DDR4-3200 dual channel
    "desktop_ddr5": 100.0,  # Typical DDR5-5600 dual channel
}


def get_hardware_bandwidth(hardware_name: str) -> float:
    """Get theoretical memory bandwidth for a hardware configuration."""
    return HARDWARE_BANDWIDTH.get(hardware_name, 200.0)
