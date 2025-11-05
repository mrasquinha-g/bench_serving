#!/usr/bin/env python3
"""
Generate summary tables from benchmark JSON results
"""

import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np


def parse_json_result(filepath: str) -> Dict:
  """Parse a benchmark JSON result file."""
  with open(filepath, 'r') as f:
    data = json.load(f)
  return data


def extract_config_from_filename(filename: str) -> Tuple[int, int, int]:
  """Extract ISL, OSL, and concurrency from filename.
  Example: isl1024_osl1024_c8.json -> (1024, 1024, 8)
  """
  parts = filename.replace('.json', '').split('_')
  isl = int(parts[0].replace('isl', ''))
  osl = int(parts[1].replace('osl', ''))
  concurrency = int(parts[2].replace('c', ''))
  return isl, osl, concurrency


def calculate_percentile(values: List[float], percentile: float) -> float:
  """Calculate percentile from a list of values."""
  if not values:
    return 0.0
  return np.percentile(values, percentile)


def flatten_nested_list(nested_list: List[List[float]]) -> List[float]:
  """Flatten a nested list of lists into a single list."""
  flattened = []
  for sublist in nested_list:
    if isinstance(sublist, list):
      flattened.extend(sublist)
    else:
      flattened.append(sublist)
  return flattened

def calculate_e2e_latencies(ttfts: List[float],
                            itls: List[List[float]]) -> List[float]:
  """Calculate end-to-end latency for each request.
  E2EL = TTFT + sum(ITLs for that request)
  """
  e2els = []
  for i in range(min(len(ttfts), len(itls))):
    ttft = ttfts[i]
    itl_list = itls[i] if isinstance(itls[i], list) else [itls[i]]
    e2el = ttft + sum(itl_list)
    e2els.append(e2el)
  return e2els


def format_number(value: float, decimal_places: int = 2) -> str:
  """Format number with specified decimal places."""
  return f"{value:.{decimal_places}f}"


def generate_summary_tables(results_dir: str):
  """Generate summary tables from all JSON results."""

  json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

  if not json_files:
    print(f"No JSON files found in {results_dir}")
    return

  results = []

  for json_file in sorted(json_files):
    filepath = os.path.join(results_dir, json_file)
    try:
      print(f"Processing {json_file}...", end=' ', flush=True)
      isl, osl, concurrency = extract_config_from_filename(json_file)
      data = parse_json_result(filepath)

      if not data.get('completed', 0):
        print(f"Warning: {json_file} has no completed requests, skipping")
        continue

      ttft_values = data.get('ttfts', [])
      itl_nested = data.get('itls', [])

      print(f"[ttft:{len(ttft_values)}, itl:{len(itl_nested)}]...",
            end=' ', flush=True)

      e2el_values = calculate_e2e_latencies(ttft_values, itl_nested)

      tpot_p50 = data.get('median_tpot_ms', 0.0)
      tpot_p99 = data.get('p99_tpot_ms', 0.0)
      tpot_mean = data.get('mean_tpot_ms', 0.0)

      tpot_p90 = tpot_p50 + 0.9 * (tpot_p99 - tpot_p50)
      tpot_p95 = tpot_p50 + 0.95 * (tpot_p99 - tpot_p50)

      itl_p50 = data.get('median_itl_ms', 0.0)
      itl_p99 = data.get('p99_itl_ms', 0.0)
      itl_mean = data.get('mean_itl_ms', 0.0)

      itl_p90 = itl_p50 + 0.9 * (itl_p99 - itl_p50)
      itl_p95 = itl_p50 + 0.95 * (itl_p99 - itl_p50)

      result = {
        'isl': isl,
        'osl': osl,
        'concurrency': concurrency,
        'request_throughput': data.get('request_throughput', 0.0),
        'output_throughput': data.get('output_throughput', 0.0),
        'total_token_throughput': data.get('total_token_throughput', 0.0),
        'ttft_p50': calculate_percentile(ttft_values, 50) * 1000,
        'ttft_p90': calculate_percentile(ttft_values, 90) * 1000,
        'ttft_p95': calculate_percentile(ttft_values, 95) * 1000,
        'ttft_p99': calculate_percentile(ttft_values, 99) * 1000,
        'tpot_p50': tpot_p50,
        'tpot_p90': tpot_p90,
        'tpot_p95': tpot_p95,
        'tpot_p99': tpot_p99,
        'itl_p50': itl_p50,
        'itl_p90': itl_p90,
        'itl_p95': itl_p95,
        'itl_p99': itl_p99,
        'e2el_p50': calculate_percentile(e2el_values, 50) * 1000,
        'e2el_p90': calculate_percentile(e2el_values, 90) * 1000,
        'e2el_p95': calculate_percentile(e2el_values, 95) * 1000,
        'e2el_p99': calculate_percentile(e2el_values, 99) * 1000,
      }

      results.append(result)
      print("done")

    except Exception as e:
      print(f"\nError processing {json_file}: {e}")
      import traceback
      traceback.print_exc()
      continue

  if not results:
    print("No valid results to display")
    return

  results.sort(key=lambda x: (x['isl'], x['osl'], x['concurrency']))

  print("\n")
  print("="*80)
  print("COMPLETE RESULTS TABLE")
  print("="*80)
  print(f"{'Input':<8}{'Output':<8}{'Concur-':<9}{'Request':<13}"
        f"{'Output Token':<16}{'P50':<9}{'P99':<9}{'P50':<9}{'P99':<9}"
        f"{'P50':<9}{'P99':<10}{'P50':<11}{'P99':<11}")
  print(f"{'Tokens':<8}{'Tokens':<8}{'rency':<9}{'Throughput':<13}"
        f"{'Throughput':<16}{'TTFT':<9}{'TTFT':<9}{'TPOT':<9}{'TPOT':<9}"
        f"{'ITL':<9}{'ITL':<10}{'E2EL':<11}{'E2EL':<11}")
  print(f"{'':<8}{'':<8}{'':<9}{'(req/s)':<13}{'(tok/s)':<16}"
        f"{'(ms)':<9}{'(ms)':<9}{'(ms)':<9}{'(ms)':<9}{'(ms)':<9}"
        f"{'(ms)':<10}{'(ms)':<11}{'(ms)':<11}")
  print(f"{'------':<8}{'------':<8}{'-------':<9}{'-----------':<13}"
        f"{'--------------':<16}{'-------':<9}{'-------':<9}"
        f"{'-------':<9}{'-------':<9}{'-------':<9}{'--------':<10}"
        f"{'---------':<11}{'---------':<11}")

  for r in results:
    print(f"{r['isl']:<8}{r['osl']:<8}{r['concurrency']:<9}"
          f"{format_number(r['request_throughput']):<13}"
          f"{format_number(r['output_throughput']):<16}"
          f"{format_number(r['ttft_p50']):<9}"
          f"{format_number(r['ttft_p99']):<9}"
          f"{format_number(r['tpot_p50']):<9}"
          f"{format_number(r['tpot_p99']):<9}"
          f"{format_number(r['itl_p50']):<9}"
          f"{format_number(r['itl_p99']):<10}"
          f"{format_number(r['e2el_p50']):<11}"
          f"{format_number(r['e2el_p99']):<11}")

  print("\n")
  print("="*80)
  print("THROUGHPUT COMPARISON")
  print("="*80)
  print(f"{'Configuration':<29}{'Request/s':<13}{'Output Tokens/s':<19}"
        f"{'Total Tokens/s':<17}")
  print(f"{'---------------------------':<29}{'-----------':<13}"
        f"{'-----------------':<19}{'---------------':<17}")

  for r in results:
    config_name = f"{r['isl']}in/{r['osl']}out @ conc={r['concurrency']}"
    print(f"{config_name:<29}{format_number(r['request_throughput']):<13}"
          f"{format_number(r['output_throughput']):<19}"
          f"{format_number(r['total_token_throughput']):<17}")

  print("\n")
  print("="*80)
  print("TIME TO FIRST TOKEN (TTFT) - ALL PERCENTILES")
  print("="*80)
  print(f"{'Configuration':<29}{'P50 (ms)':<12}{'P90 (ms)':<12}"
        f"{'P95 (ms)':<12}{'P99 (ms)':<12}")
  print(f"{'---------------------------':<29}{'----------':<12}"
        f"{'----------':<12}{'----------':<12}{'----------':<12}")

  for r in results:
    config_name = f"{r['isl']}in/{r['osl']}out @ conc={r['concurrency']}"
    print(f"{config_name:<29}{format_number(r['ttft_p50']):<12}"
          f"{format_number(r['ttft_p90']):<12}"
          f"{format_number(r['ttft_p95']):<12}"
          f"{format_number(r['ttft_p99']):<12}")

  print("\n")
  print("="*80)
  print("TIME PER OUTPUT TOKEN (TPOT) - ALL PERCENTILES")
  print("="*80)
  print(f"{'Configuration':<29}{'P50 (ms)':<12}{'P90 (ms)':<12}"
        f"{'P95 (ms)':<12}{'P99 (ms)':<12}")
  print(f"{'---------------------------':<29}{'----------':<12}"
        f"{'----------':<12}{'----------':<12}{'----------':<12}")

  for r in results:
    config_name = f"{r['isl']}in/{r['osl']}out @ conc={r['concurrency']}"
    print(f"{config_name:<29}{format_number(r['tpot_p50']):<12}"
          f"{format_number(r['tpot_p90']):<12}"
          f"{format_number(r['tpot_p95']):<12}"
          f"{format_number(r['tpot_p99']):<12}")

  print("\n")
  print("="*80)
  print("INTER-TOKEN LATENCY (ITL) - ALL PERCENTILES")
  print("="*80)
  print(f"{'Configuration':<29}{'P50 (ms)':<12}{'P90 (ms)':<12}"
        f"{'P95 (ms)':<12}{'P99 (ms)':<12}")
  print(f"{'---------------------------':<29}{'----------':<12}"
        f"{'----------':<12}{'----------':<12}{'----------':<12}")

  for r in results:
    config_name = f"{r['isl']}in/{r['osl']}out @ conc={r['concurrency']}"
    print(f"{config_name:<29}{format_number(r['itl_p50']):<12}"
          f"{format_number(r['itl_p90']):<12}"
          f"{format_number(r['itl_p95']):<12}"
          f"{format_number(r['itl_p99']):<12}")

  print("\n")
  print("="*80)
  print("END-TO-END LATENCY (E2EL) - ALL PERCENTILES")
  print("="*80)
  print(f"{'Configuration':<29}{'P50 (ms)':<13}{'P90 (ms)':<13}"
        f"{'P95 (ms)':<13}{'P99 (ms)':<13}")
  print(f"{'---------------------------':<29}{'-----------':<13}"
        f"{'-----------':<13}{'-----------':<13}{'-----------':<13}")

  for r in results:
    config_name = f"{r['isl']}in/{r['osl']}out @ conc={r['concurrency']}"
    print(f"{config_name:<29}{format_number(r['e2el_p50']):<13}"
          f"{format_number(r['e2el_p90']):<13}"
          f"{format_number(r['e2el_p95']):<13}"
          f"{format_number(r['e2el_p99']):<13}")

def main():
  parser = argparse.ArgumentParser(
    description='Generate summary tables from benchmark results'
  )
  parser.add_argument(
    '--results-dir',
    type=str,
    default='results_set1/run_20251103_153905',
    help='Directory containing JSON result files'
  )

  args = parser.parse_args()

  if not os.path.isdir(args.results_dir):
    print(f"Error: Directory {args.results_dir} does not exist")
    return 1

  generate_summary_tables(args.results_dir)
  return 0


if __name__ == '__main__':
  exit(main())
