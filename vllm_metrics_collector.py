#!/usr/bin/env python3
"""
vLLM Metrics Collector
Queries Prometheus metrics endpoint every 30s and writes to CSV
"""

import argparse
import csv
import logging
import re
import signal
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class PrometheusParser:
    """Parse Prometheus text format metrics."""

    @staticmethod
    def parse_metrics(text: str) -> List[Dict[str, Any]]:
        """Parse Prometheus format into structured records."""
        metrics = []
        lines = text.strip().split('\n')

        current_metric = None
        current_help = None
        current_type = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                if line.startswith('# HELP'):
                    parts = line.split(' ', 3)
                    if len(parts) >= 4:
                        current_help = parts[3]
                        current_metric = parts[2]
                elif line.startswith('# TYPE'):
                    parts = line.split(' ', 3)
                    if len(parts) >= 4:
                        current_type = parts[3]
                continue

            # Parse metric line: metric_name{labels} value timestamp
            match = re.match(
                r'([a-zA-Z_:][a-zA-Z0-9_:]*)'
                r'(?:\{([^}]*)\})?\s+([^\s]+)(?:\s+(\d+))?',
                line
            )
            if match:
                metric_name = match.group(1)
                labels_str = match.group(2) or ''
                value = match.group(3)

                # Parse labels
                labels = {}
                if labels_str:
                    for label_pair in re.findall(r'(\w+)="([^"]*)"',
                                                  labels_str):
                        labels[label_pair[0]] = label_pair[1]

                # Convert value
                try:
                    if value == '+Inf':
                        value_float = float('inf')
                    elif value == '-Inf':
                        value_float = float('-inf')
                    elif value == 'NaN':
                        value_float = float('nan')
                    else:
                        value_float = float(value)
                except ValueError:
                    value_float = 0.0

                metrics.append({
                    'metric_name': metric_name,
                    'value': value_float,
                    'labels': labels,
                    'type': current_type,
                })

        return metrics


class VLLMMetricsCollector:
    """Collect vLLM metrics and write to CSV."""

    def __init__(
        self,
        metrics_url: str,
        output_file: str,
        collection_interval: int = 30,
        hostname: Optional[str] = None,
        percentiles: Optional[List[float]] = None,
    ):
        self.metrics_url = metrics_url
        self.output_file = output_file
        self.collection_interval = collection_interval
        self.hostname = hostname or socket.gethostname()
        self.percentiles = percentiles or [50.0, 90.0, 99.0]
        self.parser = PrometheusParser()
        self.session = requests.Session()
        self.csv_file = None
        self.csv_writer = None
        self.csv_fieldnames = None
        self.shutdown = False

        # For throughput calculation
        self.prev_counters = {}
        self.prev_collection_time = None

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logging.info(f"\nReceived signal {signum}, shutting down...")
        self.shutdown = True

    def fetch_metrics(self) -> Optional[str]:
        """Fetch metrics from Prometheus endpoint."""
        try:
            response = self.session.get(self.metrics_url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error fetching metrics: {e}")
            return None

    def filter_vllm_metrics(self, metrics: List[Dict]) -> List[Dict]:
        """Filter to only vLLM metrics."""
        filtered = []
        for m in metrics:
            if not m['metric_name'].startswith('vllm:'):
                continue

            # Skip _created metrics (just metadata)
            if '_created' in m['metric_name']:
                continue

            # Skip histogram buckets (we'll compute percentiles)
            if '_bucket' in m['metric_name']:
                continue

            # Skip inf/nan values
            if not isinstance(m['value'], (int, float)):
                continue
            if m['value'] in (float('inf'), float('-inf'), float('nan')):
                continue

            filtered.append(m)

        return filtered

    def group_histogram_metrics(
        self, metrics: List[Dict]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group histogram metrics by base name and labels.
        Returns dict of {base_name+labels: {buckets, sum, count}}
        """
        histograms = defaultdict(lambda: {
            'buckets': [],
            'sum': None,
            'count': None,
            'labels': {}
        })

        for m in metrics:
            metric_name = m['metric_name']
            labels = m.get('labels', {})

            # Create key without the suffix and le label
            if '_bucket' in metric_name:
                base_name = metric_name.replace('_bucket', '')
                label_key = tuple(sorted(
                    (k, v) for k, v in labels.items() if k != 'le'
                ))
                key = (base_name, label_key)

                le_value = labels.get('le')
                if le_value == '+Inf':
                    le_float = float('inf')
                else:
                    try:
                        le_float = float(le_value)
                    except (ValueError, TypeError):
                        continue

                histograms[key]['buckets'].append((le_float, m['value']))
                histograms[key]['labels'] = {
                    k: v for k, v in labels.items() if k != 'le'
                }

            elif '_sum' in metric_name:
                base_name = metric_name.replace('_sum', '')
                label_key = tuple(sorted(labels.items()))
                key = (base_name, label_key)
                histograms[key]['sum'] = m['value']
                histograms[key]['labels'] = labels

            elif '_count' in metric_name:
                base_name = metric_name.replace('_count', '')
                label_key = tuple(sorted(labels.items()))
                key = (base_name, label_key)
                histograms[key]['count'] = m['value']
                histograms[key]['labels'] = labels

        return histograms

    def calculate_percentile(
        self, buckets: List[Tuple[float, float]], percentile: float
    ) -> Optional[float]:
        """
        Calculate percentile from histogram buckets using linear interpolation.
        """
        if not buckets:
            return None

        # Sort buckets by upper bound
        sorted_buckets = sorted(buckets, key=lambda x: x[0])

        # Get total count (last bucket should have total)
        total_count = sorted_buckets[-1][1]
        if total_count == 0:
            return None

        # Calculate target count for this percentile
        target_count = (percentile / 100.0) * total_count

        # Find the bucket containing this percentile
        prev_bound = 0.0
        prev_count = 0.0

        for upper_bound, cumulative_count in sorted_buckets:
            if cumulative_count >= target_count:
                # Linear interpolation within this bucket
                if cumulative_count == prev_count:
                    return upper_bound

                fraction = (target_count - prev_count) / \
                          (cumulative_count - prev_count)
                return prev_bound + fraction * (upper_bound - prev_bound)

            prev_bound = upper_bound
            prev_count = cumulative_count

        # If we get here, return the last bucket's upper bound
        return sorted_buckets[-1][0]

    def compute_histogram_percentiles(
        self, metrics: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Compute percentiles from histogram metrics.
        """
        histograms = self.group_histogram_metrics(metrics)
        percentile_metrics = []

        for (base_name, label_key), hist_data in histograms.items():
            buckets = hist_data['buckets']
            if not buckets:
                continue

            labels = hist_data['labels']

            # Calculate each percentile
            for p in self.percentiles:
                value = self.calculate_percentile(buckets, p)
                if value is not None:
                    percentile_metrics.append({
                        'metric_name': f"{base_name}_p{int(p)}",
                        'value': value,
                        'labels': labels,
                        'type': 'histogram_percentile',
                    })

            # Also include sum and count if available
            if hist_data['sum'] is not None:
                percentile_metrics.append({
                    'metric_name': f"{base_name}_sum",
                    'value': hist_data['sum'],
                    'labels': labels,
                    'type': 'histogram_sum',
                })

            if hist_data['count'] is not None:
                percentile_metrics.append({
                    'metric_name': f"{base_name}_count",
                    'value': hist_data['count'],
                    'labels': labels,
                    'type': 'histogram_count',
                })

        return percentile_metrics

    def compute_throughput_metrics(
        self, metrics: List[Dict], current_time: int
    ) -> List[Dict[str, Any]]:
        """
        Compute throughput from counter deltas.
        Returns list of throughput metrics.
        """
        throughput_metrics = []

        # Skip first collection (need previous values)
        if self.prev_collection_time is None:
            # Save counters for next iteration
            for m in metrics:
                if m['metric_name'] in ['vllm:prompt_tokens_total',
                                       'vllm:generation_tokens_total']:
                    labels = m.get('labels', {})
                    key = (m['metric_name'], tuple(sorted(labels.items())))
                    self.prev_counters[key] = m['value']

            self.prev_collection_time = current_time
            return throughput_metrics

        # Calculate time delta
        time_delta = current_time - self.prev_collection_time
        if time_delta <= 0:
            return throughput_metrics

        # Calculate throughput for token counters
        for m in metrics:
            metric_name = m['metric_name']

            if metric_name == 'vllm:prompt_tokens_total':
                labels = m.get('labels', {})
                key = (metric_name, tuple(sorted(labels.items())))

                if key in self.prev_counters:
                    delta = m['value'] - self.prev_counters[key]
                    throughput = delta / time_delta

                    throughput_metrics.append({
                        'metric_name': 'vllm:avg_prompt_throughput_tokens_per_s',
                        'value': throughput,
                        'labels': labels,
                        'type': 'derived_gauge',
                    })

                self.prev_counters[key] = m['value']

            elif metric_name == 'vllm:generation_tokens_total':
                labels = m.get('labels', {})
                key = (metric_name, tuple(sorted(labels.items())))

                if key in self.prev_counters:
                    delta = m['value'] - self.prev_counters[key]
                    throughput = delta / time_delta

                    throughput_metrics.append({
                        'metric_name': 'vllm:avg_generation_throughput_tokens_per_s',
                        'value': throughput,
                        'labels': labels,
                        'type': 'derived_gauge',
                    })

                self.prev_counters[key] = m['value']

        self.prev_collection_time = current_time
        return throughput_metrics

    def create_csv_row(
        self, metric: Dict[str, Any], collection_time: int
    ) -> Dict[str, Any]:
        """Convert metric to CSV row."""
        row = {
            'time': collection_time,
            'hostname': self.hostname,
            'metric_name': metric['metric_name'],
            'value': metric['value'],
            'metric_type': metric.get('type', 'unknown'),
        }

        # Add labels as separate columns
        for key, value in metric.get('labels', {}).items():
            row[f'label_{key}'] = value

        return row

    def initialize_csv(self, first_batch: List[Dict[str, Any]]):
        """Initialize CSV file with headers based on first batch."""
        if not first_batch:
            return

        # Determine all fieldnames from first batch
        fieldnames_set = set()
        for row in first_batch:
            fieldnames_set.update(row.keys())

        # Order: time, hostname, metric_name, value, metric_type, then labels
        fixed_fields = ['time', 'hostname', 'metric_name', 'value', 'metric_type']
        label_fields = sorted([f for f in fieldnames_set if f.startswith('label_')])
        self.csv_fieldnames = fixed_fields + label_fields

        # Open CSV file and write header
        self.csv_file = open(self.output_file, 'w', newline='')
        self.csv_writer = csv.DictWriter(
            self.csv_file, fieldnames=self.csv_fieldnames
        )
        self.csv_writer.writeheader()
        self.csv_file.flush()

        logging.info(f"Initialized CSV with {len(self.csv_fieldnames)} columns")

    def write_to_csv(self, rows: List[Dict[str, Any]]):
        """Write rows to CSV file."""
        if not self.csv_writer:
            self.initialize_csv(rows)
            if not self.csv_writer:
                return

        for row in rows:
            self.csv_writer.writerow(row)

        self.csv_file.flush()
        logging.info(f"Wrote {len(rows)} rows to CSV")

    def collect_and_write(self):
        """Collect metrics and write to CSV."""
        collection_time = int(time.time())
        timestamp_str = datetime.fromtimestamp(collection_time).isoformat()

        logging.info(
            f"[{timestamp_str}] Collecting metrics from {self.metrics_url}")

        # Fetch metrics
        metrics_text = self.fetch_metrics()
        if not metrics_text:
            logging.warning("No metrics fetched, skipping write")
            return

        # Parse metrics
        all_metrics = self.parser.parse_metrics(metrics_text)

        # Get vLLM metrics only
        vllm_all = [m for m in all_metrics
                    if m['metric_name'].startswith('vllm:')]

        # Filter out histogram buckets and get regular metrics
        vllm_metrics = self.filter_vllm_metrics(vllm_all)

        # Compute histogram percentiles from buckets
        percentile_metrics = self.compute_histogram_percentiles(vllm_all)

        # Compute throughput from counter deltas
        throughput_metrics = self.compute_throughput_metrics(
            vllm_all, collection_time)

        logging.info(
            f"Parsed {len(vllm_metrics)} regular metrics + "
            f"{len(percentile_metrics)} histogram percentiles + "
            f"{len(throughput_metrics)} throughput metrics")

        # Combine regular metrics, percentiles, and throughput
        all_upload_metrics = vllm_metrics + percentile_metrics + throughput_metrics

        if not all_upload_metrics:
            logging.warning("No metrics to write")
            return

        # Convert to CSV rows
        csv_rows = []
        for metric in all_upload_metrics:
            try:
                row = self.create_csv_row(metric, collection_time)
                csv_rows.append(row)
            except Exception as e:
                logging.warning(f"Failed to create row for {metric}: {e}")

        # Write to CSV
        if csv_rows:
            self.write_to_csv(csv_rows)

    def run(self):
        """Run collection loop."""
        self.setup_signal_handlers()

        logging.info("Starting vLLM metrics collector")
        logging.info(f"  Metrics URL: {self.metrics_url}")
        logging.info(f"  Output CSV: {self.output_file}")
        logging.info(f"  Collection Interval: {self.collection_interval}s")
        logging.info(f"  Hostname: {self.hostname}")
        logging.info(f"  Percentiles: {self.percentiles}")
        logging.info("")

        try:
            while not self.shutdown:
                start_time = time.time()

                try:
                    self.collect_and_write()
                except Exception as e:
                    logging.error(f"Collection error: {e}")

                # Sleep for remainder of interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.collection_interval - elapsed)

                # Sleep in small chunks to be responsive to shutdown signal
                while sleep_time > 0 and not self.shutdown:
                    chunk = min(sleep_time, 1.0)
                    time.sleep(chunk)
                    sleep_time -= chunk

        finally:
            if self.csv_file:
                self.csv_file.close()
            logging.info("Metrics collector stopped")
            logging.info(f"CSV file saved: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Collect vLLM Prometheus metrics and write to CSV')
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8081/metrics',
        help='Prometheus metrics endpoint URL (default: http://localhost:8081/metrics)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Collection interval in seconds (default: 30)'
    )
    parser.add_argument(
        '--hostname',
        type=str,
        default=None,
        help='Hostname to tag metrics (default: system hostname)'
    )
    parser.add_argument(
        '--percentiles',
        type=str,
        default='50,90,99',
        help='Comma-separated percentiles to compute from histograms (default: 50,90,99)'
    )

    args = parser.parse_args()

    # Parse percentiles
    percentiles = [float(p.strip()) for p in args.percentiles.split(',')]

    collector = VLLMMetricsCollector(
        metrics_url=args.url,
        output_file=args.output,
        collection_interval=args.interval,
        hostname=args.hostname,
        percentiles=percentiles,
    )

    collector.run()


if __name__ == '__main__':
    main()
