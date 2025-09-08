"""Ground truth comparison tool for analyzing extraction accuracy."""

import sys
import pandas as pd
from pathlib import Path
from simple_image_reader import SimpleImageReader
from typing import List, Dict, Tuple
import re

class GroundTruthComparison:
    """Compare extracted transactions with ground truth data."""
    
    def __init__(self, ground_truth_path: str):
        """Initialize with ground truth CSV path."""
        self.ground_truth_df = pd.read_csv(ground_truth_path)
        self.reader = SimpleImageReader()
    
    def normalize_description(self, description: str) -> str:
        """Normalize description for comparison."""
        # Remove extra spaces, convert to uppercase
        normalized = re.sub(r'\s+', ' ', description.upper().strip())
        # Remove common OCR artifacts
        normalized = normalized.replace('/', ' ').replace('-', ' ')
        return normalized
    
    def normalize_amount(self, amount) -> float:
        """Normalize amount for comparison."""
        if isinstance(amount, str):
            # Remove currency symbols and convert
            amount = amount.replace(',', '.').replace('-', '').replace('+', '')
            amount = re.sub(r'[^\d\.]', '', amount)
            return float(amount) if amount else 0.0
        return abs(float(amount))
    
    def find_best_match(self, extracted_trans: Dict, ground_truth_list: List[Dict]) -> Tuple[Dict, float]:
        """Find the best matching ground truth transaction."""
        best_match = None
        best_score = 0.0
        
        extracted_desc = self.normalize_description(extracted_trans['description'])
        extracted_amount = self.normalize_amount(extracted_trans['amount_value'])
        
        for gt_trans in ground_truth_list:
            gt_desc = self.normalize_description(gt_trans['description'])
            gt_amount = self.normalize_amount(gt_trans['amount'])
            
            # Calculate similarity score
            score = 0.0
            
            # Amount match (most important)
            if abs(extracted_amount - gt_amount) < 0.01:
                score += 0.6
            elif abs(extracted_amount - gt_amount) < 1.0:
                score += 0.3
            
            # Description similarity
            desc_words_extracted = set(extracted_desc.split())
            desc_words_gt = set(gt_desc.split())
            
            if desc_words_extracted and desc_words_gt:
                common_words = desc_words_extracted.intersection(desc_words_gt)
                total_words = desc_words_extracted.union(desc_words_gt)
                desc_similarity = len(common_words) / len(total_words) if total_words else 0
                score += 0.4 * desc_similarity
            
            if score > best_score:
                best_score = score
                best_match = gt_trans
        
        return best_match, best_score
    
    def compare_statement(self, statement_id: int, image_path: Path) -> Dict:
        """Compare extracted transactions with ground truth for a single statement."""
        # Get ground truth for this statement
        gt_transactions = self.ground_truth_df[
            self.ground_truth_df['statement_id'] == statement_id
        ].to_dict('records')
        
        # Extract transactions from image
        result = self.reader.process_single_image(image_path)
        extracted_transactions = result['transactions']
        
        # Match extracted with ground truth
        matched_pairs = []
        used_gt_indices = set()
        
        for ext_trans in extracted_transactions:
            best_match, score = self.find_best_match(ext_trans, gt_transactions)
            
            if best_match and score > 0.5:  # Threshold for considering a match
                gt_index = gt_transactions.index(best_match)
                if gt_index not in used_gt_indices:
                    matched_pairs.append({
                        'extracted': ext_trans,
                        'ground_truth': best_match,
                        'match_score': score,
                        'status': 'MATCHED'
                    })
                    used_gt_indices.add(gt_index)
                else:
                    matched_pairs.append({
                        'extracted': ext_trans,
                        'ground_truth': None,
                        'match_score': 0.0,
                        'status': 'DUPLICATE'
                    })
            else:
                matched_pairs.append({
                    'extracted': ext_trans,
                    'ground_truth': None,
                    'match_score': score,
                    'status': 'NO_MATCH'
                })
        
        # Find missed ground truth transactions
        missed_gt = []
        for i, gt_trans in enumerate(gt_transactions):
            if i not in used_gt_indices:
                missed_gt.append({
                    'ground_truth': gt_trans,
                    'status': 'MISSED'
                })
        
        return {
            'statement_id': statement_id,
            'image_path': str(image_path),
            'ocr_confidence': result['ocr_confidence'],
            'total_gt': len(gt_transactions),
            'total_extracted': len(extracted_transactions),
            'matched_pairs': matched_pairs,
            'missed_transactions': missed_gt,
            'raw_ocr_text': result['raw_ocr_text']
        }
    
    def print_comparison_report(self, comparison: Dict):
        """Print detailed comparison report."""
        print(f"\n{'='*80}")
        print(f"STATEMENT {comparison['statement_id']} COMPARISON REPORT")
        print(f"{'='*80}")
        print(f"Image: {Path(comparison['image_path']).name}")
        print(f"OCR Confidence: {comparison['ocr_confidence']:.1f}%")
        print(f"Ground Truth Transactions: {comparison['total_gt']}")
        print(f"Extracted Transactions: {comparison['total_extracted']}")
        
        # Count matches
        matched = sum(1 for pair in comparison['matched_pairs'] if pair['status'] == 'MATCHED')
        duplicates = sum(1 for pair in comparison['matched_pairs'] if pair['status'] == 'DUPLICATE')
        no_matches = sum(1 for pair in comparison['matched_pairs'] if pair['status'] == 'NO_MATCH')
        missed = len(comparison['missed_transactions'])
        
        print(f"Matched: {matched}, Duplicates: {duplicates}, No Match: {no_matches}, Missed: {missed}")
        print(f"Accuracy: {matched}/{comparison['total_gt']} = {matched/comparison['total_gt']*100:.1f}%")
        
        print(f"\n{'-'*80}")
        print("MATCHED TRANSACTIONS:")
        print(f"{'-'*80}")
        for i, pair in enumerate(comparison['matched_pairs']):
            if pair['status'] == 'MATCHED':
                ext = pair['extracted']
                gt = pair['ground_truth']
                print(f"{i+1}. ✓ MATCH (Score: {pair['match_score']:.2f})")
                print(f"   Extracted: {ext['raw_line']}")
                print(f"   Ground Truth: {gt['date']} | {gt['description']} | {gt['amount']}")
                print()
        
        if duplicates > 0:
            print(f"\n{'-'*80}")
            print("DUPLICATE EXTRACTIONS:")
            print(f"{'-'*80}")
            for i, pair in enumerate(comparison['matched_pairs']):
                if pair['status'] == 'DUPLICATE':
                    ext = pair['extracted']
                    print(f"{i+1}. ⚠ DUPLICATE")
                    print(f"   Extracted: {ext['raw_line']}")
                    print()
        
        if no_matches > 0:
            print(f"\n{'-'*80}")
            print("UNMATCHED EXTRACTIONS:")
            print(f"{'-'*80}")
            for i, pair in enumerate(comparison['matched_pairs']):
                if pair['status'] == 'NO_MATCH':
                    ext = pair['extracted']
                    print(f"{i+1}. ✗ NO MATCH (Score: {pair['match_score']:.2f})")
                    print(f"   Extracted: {ext['raw_line']}")
                    print()
        
        if missed > 0:
            print(f"\n{'-'*80}")
            print("MISSED GROUND TRUTH TRANSACTIONS:")
            print(f"{'-'*80}")
            for i, missed_trans in enumerate(comparison['missed_transactions']):
                gt = missed_trans['ground_truth']
                print(f"{i+1}. ✗ MISSED")
                print(f"   Ground Truth: {gt['date']} | {gt['description']} | {gt['amount']}")
                print()
        
        print(f"\n{'-'*80}")
        print("RAW OCR TEXT:")
        print(f"{'-'*80}")
        lines = comparison['raw_ocr_text'].split('\n')
        for i, line in enumerate(lines, 1):
            if line.strip():
                print(f"{i:2d}: {line}")

    def compare_multiple_statements(self, folder_path: Path, limit: int = None) -> List[Dict]:
        """Compare multiple statements from a folder."""
        # Find all PNG files in the folder
        image_files = list(folder_path.glob("statement_*.png"))
        image_files.sort(key=lambda x: int(re.search(r'statement_(\d+)', x.name).group(1)))
        
        if limit:
            image_files = image_files[:limit]
        
        print(f"Processing {len(image_files)} images from {folder_path}")
        
        all_comparisons = []
        total_matched = 0
        total_gt = 0
        
        for i, image_path in enumerate(image_files, 1):
            # Extract statement ID from filename
            match = re.search(r'statement_(\d+)', image_path.name)
            if not match:
                print(f"Skipping {image_path.name} - could not extract statement ID")
                continue
            
            statement_id = int(match.group(1))
            print(f"\n[{i}/{len(image_files)}] Processing statement {statement_id}...")
            
            try:
                comparison = self.compare_statement(statement_id, image_path)
                all_comparisons.append(comparison)
                
                # Count matches for summary
                matched = sum(1 for pair in comparison['matched_pairs'] if pair['status'] == 'MATCHED')
                total_matched += matched
                total_gt += comparison['total_gt']
                
                print(f"  ✓ {matched}/{comparison['total_gt']} transactions matched ({matched/comparison['total_gt']*100:.1f}%)")
                
            except Exception as e:
                print(f"  ✗ Error processing statement {statement_id}: {e}")
        
        # Print overall summary
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"Total statements processed: {len(all_comparisons)}")
        print(f"Total ground truth transactions: {total_gt}")
        print(f"Total matched transactions: {total_matched}")
        print(f"Overall accuracy: {total_matched}/{total_gt} = {total_matched/total_gt*100:.1f}%")
        
        return all_comparisons
    
    def print_summary_report(self, comparisons: List[Dict]):
        """Print a summary report of all comparisons."""
        print(f"\n{'='*80}")
        print("DETAILED SUMMARY REPORT")
        print(f"{'='*80}")
        
        for comparison in comparisons:
            matched = sum(1 for pair in comparison['matched_pairs'] if pair['status'] == 'MATCHED')
            duplicates = sum(1 for pair in comparison['matched_pairs'] if pair['status'] == 'DUPLICATE')
            no_matches = sum(1 for pair in comparison['matched_pairs'] if pair['status'] == 'NO_MATCH')
            missed = len(comparison['missed_transactions'])
            
            print(f"Statement {comparison['statement_id']:3d}: "
                  f"{matched:2d}/{comparison['total_gt']:2d} matched "
                  f"({matched/comparison['total_gt']*100:5.1f}%) | "
                  f"Dup: {duplicates:2d} | NoMatch: {no_matches:2d} | Missed: {missed:2d} | "
                  f"OCR: {comparison['ocr_confidence']:5.1f}%")

def main():
    """Main function for ground truth comparison."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ground_truth_comparison.py <folder_path> [limit]")
        print("  python ground_truth_comparison.py <statement_id>")
        print("  python ground_truth_comparison.py <image_path>")
        print("")
        print("Examples:")
        print("  python ground_truth_comparison.py \"bank statement generator/bank_statements\" 10")
        print("  python ground_truth_comparison.py 1")
        print("  python ground_truth_comparison.py \"path/to/statement_1.png\"")
        return
    
    ground_truth_path = "bank statement generator/bank_statements/ground_truth.csv"
    comparator = GroundTruthComparison(ground_truth_path)
    
    arg = sys.argv[1]
    
    # Check if it's a folder path
    folder_path = Path(arg)
    if folder_path.is_dir():
        # Folder processing
        limit = None
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            limit = int(sys.argv[2])
        
        comparisons = comparator.compare_multiple_statements(folder_path, limit)
        
        # Ask if user wants detailed reports
        print(f"\nWould you like to see detailed reports for each statement? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes']:
                for comparison in comparisons:
                    comparator.print_comparison_report(comparison)
        except:
            pass  # Handle case where input() might not work in some environments
        
        # Always show summary
        comparator.print_summary_report(comparisons)
        
    elif arg.isdigit():
        # Statement ID provided
        statement_id = int(arg)
        image_path = Path(f"bank statement generator/bank_statements/statement_{statement_id}.png")
        
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return
            
        comparison = comparator.compare_statement(statement_id, image_path)
        comparator.print_comparison_report(comparison)
        
    else:
        # Image path provided
        image_path = Path(arg)
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return
        
        # Extract statement ID from filename
        match = re.search(r'statement_(\d+)', image_path.name)
        if not match:
            print("Could not extract statement ID from filename")
            return
        
        statement_id = int(match.group(1))
        comparison = comparator.compare_statement(statement_id, image_path)
        comparator.print_comparison_report(comparison)

if __name__ == "__main__":
    main()
