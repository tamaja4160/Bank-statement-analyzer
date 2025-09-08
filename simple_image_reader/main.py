"""Main entry point for the simple image reader."""

import sys
from pathlib import Path

from simple_image_reader import SimpleImageReader

def main():
    """Main function to demonstrate the simple image reader."""
    
    # Initialize the reader
    reader = SimpleImageReader()
    
    # Parse arguments
    limit = None
    if len(sys.argv) > 1:
        # Check if last argument is a number (limit)
        if len(sys.argv) > 2 and sys.argv[-1].isdigit():
            limit = int(sys.argv[-1])
            path_arg = sys.argv[1]
        else:
            path_arg = sys.argv[1]
        
        image_path = Path(path_arg)
        
        if image_path.is_file():
            print(f"Processing single image: {image_path}")
            result = reader.process_single_image(image_path)
            
            print(f"\nResults for {result['image_name']}:")
            print(f"OCR Confidence: {result['ocr_confidence']:.1f}%")
            print(f"Transactions found: {result['extracted_count']}")
            
            if result['transactions']:
                print("\nExtracted transactions:")
                for i, trans in enumerate(result['transactions'], 1):
                    print(f"{i}. {trans['raw_line']}")
            
            # Save single result
            if result['transactions']:
                reader.save_results_to_csv(result['transactions'], f"{image_path.stem}_transactions.csv")
                
        elif image_path.is_dir():
            if limit:
                print(f"Processing directory: {image_path} (limit: {limit} images)")
            else:
                print(f"Processing directory: {image_path}")
            results = reader.process_directory(image_path, limit)
            
            # Save all results
            if results['transactions']:
                reader.save_results_to_csv(results['transactions'], "all_extracted_transactions.csv")
                reader.save_results_to_json(results, "extraction_results.json")
        else:
            print(f"Path not found: {image_path}")
    else:
        # Default: process bank statements directory if it exists
        default_dir = Path("../bank statement generator/bank_statements")
        if default_dir.exists():
            print(f"Processing default directory: {default_dir}")
            results = reader.process_directory(default_dir)
            
            # Save results
            if results['transactions']:
                reader.save_results_to_csv(results['transactions'], "extracted_transactions.csv")
                reader.save_results_to_json(results, "extraction_results.json")
        else:
            print("Usage:")
            print("  python main.py <image_file>                    - Process single image")
            print("  python main.py <directory>                     - Process all images in directory")
            print("  python main.py <directory> <limit>             - Process first N images in directory")
            print("  python main.py                                 - Process default bank statements directory")

if __name__ == "__main__":
    main()
