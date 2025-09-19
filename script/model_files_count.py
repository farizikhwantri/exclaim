#!/usr/bin/env python3
"""
Script to calculate statistics for model names (before first underscore) in a folder.
Supports various file formats and provides detailed statistics.
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import fnmatch
from typing import Dict, List, Tuple

def extract_model_name(filename: str) -> str:
    """Extract model name from filename (everything before first underscore)."""
    return filename.split('_')[0] if '_' in filename else filename

def get_file_info(filepath: Path) -> Dict:
    """Get additional information about the file."""
    info = {
        'size_bytes': filepath.stat().st_size,
        'size_mb': round(filepath.stat().st_size / (1024 * 1024), 2),
        'extension': filepath.suffix.lower()
    }
    
    # Try to get additional info for JSON files
    if filepath.suffix.lower() == '.json':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                info['json_keys'] = len(data) if isinstance(data, dict) else 0
                info['json_type'] = type(data).__name__
        except (json.JSONDecodeError, Exception):
            info['json_keys'] = 0
            info['json_type'] = 'invalid'
    
    return info

def analyze_files(directory: str, pattern: str = "*", recursive: bool = False) -> Dict:
    """Analyze files in directory and return statistics by model name."""
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    if not directory_path.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a directory")
    
    # Find all files matching the pattern
    if recursive:
        files = list(directory_path.rglob(pattern))
    else:
        files = list(directory_path.glob(pattern))
    
    # Filter only files (not directories)
    files = [f for f in files if f.is_file()]
    
    # Group files by model name
    model_stats = defaultdict(lambda: {
        'count': 0,
        'files': [],
        'total_size_bytes': 0,
        'total_size_mb': 0,
        'extensions': Counter(),
        'file_info': []
    })
    
    total_files = 0
    total_size_bytes = 0
    
    for file_path in files:
        model_name = extract_model_name(file_path.name)
        file_info = get_file_info(file_path)
        
        model_stats[model_name]['count'] += 1
        model_stats[model_name]['files'].append(file_path.name)
        model_stats[model_name]['total_size_bytes'] += file_info['size_bytes']
        model_stats[model_name]['total_size_mb'] += file_info['size_mb']
        model_stats[model_name]['extensions'][file_info['extension']] += 1
        model_stats[model_name]['file_info'].append(file_info)
        
        total_files += 1
        total_size_bytes += file_info['size_bytes']
    
    return {
        'model_stats': dict(model_stats),
        'total_files': total_files,
        'total_size_bytes': total_size_bytes,
        'total_size_mb': round(total_size_bytes / (1024 * 1024), 2),
        'directory': str(directory_path),
        'pattern': pattern,
        'recursive': recursive
    }

def print_detailed_stats(stats: Dict, show_files: bool = False, sort_by: str = 'count'):
    """Print detailed statistics in a formatted way."""
    model_stats = stats['model_stats']
    
    print(f"\nFile Analysis Results")
    print("=" * 60)
    print(f"Directory: {stats['directory']}")
    print(f"Pattern: {stats['pattern']}")
    print(f"Recursive: {stats['recursive']}")
    print(f"Total files found: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']} MB ({stats['total_size_bytes']:,} bytes)")
    print()
    
    if not model_stats:
        print("No files found matching the criteria.")
        return
    
    # Sort models by specified criteria
    if sort_by == 'count':
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    elif sort_by == 'size':
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['total_size_mb'], reverse=True)
    elif sort_by == 'name':
        sorted_models = sorted(model_stats.items(), key=lambda x: x[0])
    else:
        sorted_models = model_stats.items()
    
    print("Model Statistics:")
    print("-" * 60)
    print(f"{'Model Name':<25} {'Files':<8} {'Total Size (MB)':<15} {'Extensions'}")
    print("-" * 60)
    
    for model_name, stats_data in sorted_models:
        extensions_str = ', '.join([f"{ext}({count})" for ext, count in stats_data['extensions'].items()])
        print(f"{model_name:<25} {stats_data['count']:<8} {stats_data['total_size_mb']:<15.2f} {extensions_str}")
    
    print("-" * 60)
    
    # Detailed breakdown for each model
    if show_files:
        print("\nDetailed File Breakdown:")
        print("=" * 60)
        
        for model_name, stats_data in sorted_models:
            print(f"\n{model_name} ({stats_data['count']} files, {stats_data['total_size_mb']:.2f} MB):")
            print("-" * 40)
            
            for i, filename in enumerate(stats_data['files']):
                file_info = stats_data['file_info'][i]
                print(f"  • {filename}")
                print(f"    Size: {file_info['size_mb']} MB ({file_info['size_bytes']:,} bytes)")
                
                if file_info.get('json_keys') is not None:
                    print(f"    JSON: {file_info['json_keys']} keys, type: {file_info['json_type']}")

def export_to_json(stats: Dict, output_file: str):
    """Export statistics to JSON file."""
    # Convert Path objects to strings for JSON serialization
    json_stats = stats.copy()
    for model_name, model_data in json_stats['model_stats'].items():
        # Convert Counter to dict
        model_data['extensions'] = dict(model_data['extensions'])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)
    
    print(f"Statistics exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Calculate statistics for model names (before first underscore) in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Analyze current directory for all files
  %(prog)s /path/to/files              # Analyze specific directory
  %(prog)s -p "*.json"                 # Only analyze JSON files
  %(prog)s -p "*.json" --show-files    # Show detailed file breakdown
  %(prog)s -r                          # Recursive search
  %(prog)s --sort-by size              # Sort by file size
  %(prog)s --export stats.json         # Export results to JSON
        """
    )
    
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Directory to analyze (default: current directory)'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        default='*',
        help='File pattern to match (default: *)'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Search recursively in subdirectories'
    )
    
    parser.add_argument(
        '--show-files',
        action='store_true',
        help='Show detailed file breakdown for each model'
    )
    
    parser.add_argument(
        '--sort-by',
        choices=['count', 'size', 'name'],
        default='count',
        help='Sort models by: count (default), size, or name'
    )
    
    parser.add_argument(
        '--export',
        metavar='FILE',
        help='Export statistics to JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        # Analyze files
        stats = analyze_files(args.directory, args.pattern, args.recursive)
        
        # Print results
        print_detailed_stats(stats, args.show_files, args.sort_by)
        
        # Export if requested
        if args.export:
            export_to_json(stats, args.export)
            
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()