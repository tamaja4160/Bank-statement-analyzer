import pandas as pd
import os
import re

def analyze_parsing_output():
    """Analyze the parsing output from the improved parser"""
    
    print("IMPROVED PARSER RESULTS ANALYSIS")
    print("=" * 50)
    
    # Load ground truth
    ground_truth = pd.read_csv("bank statement generator/bank_statements/ground_truth.csv")
    total_ground_truth = len(ground_truth)
    ground_truth_counts = ground_truth.groupby('statement_id').size()
    
    print(f"Ground Truth: {total_ground_truth} total transactions across 200 statements")
    print(f"Average per statement: {ground_truth_counts.mean():.2f}")
    
    # Since the improved parser failed, let's create a fixed version and run it
    print(f"\nThe improved parser encountered a column name error.")
    print(f"Let me create a quick fixed version to get the results...")
    
    # Create a simple parser to extract transactions and count them
    from improved_bank_statement_parser import ImprovedBankStatementParser
    
    try:
        # Initialize parser
        parser = ImprovedBankStatementParser("bank statement generator/bank_statements/ground_truth.csv")
        
        # Parse just a few statements to test
        test_statements = [1, 2, 3, 4, 5]
        results = {}
        
        print(f"\nTesting improved parser on first 5 statements:")
        print(f"{'Statement':<10} {'Ground Truth':<12} {'Extracted':<10} {'Status'}")
        print("-" * 45)
        
        for stmt_id in test_statements:
            image_path = f"bank statement generator/bank_statements/statement_{stmt_id}.png"
            if os.path.exists(image_path):
                try:
                    # Process the image
                    image_variants = parser.advanced_preprocess_image(image_path)
                    ocr_text = parser.extract_text_with_multiple_configs(image_variants)
                    transactions = parser.enhanced_parse_transactions_from_text(ocr_text, stmt_id)
                    
                    gt_count = len(ground_truth[ground_truth['statement_id'] == stmt_id])
                    extracted_count = len(transactions)
                    
                    status = "✓ Perfect" if extracted_count == gt_count else "~ Partial" if extracted_count > 0 else "✗ Zero"
                    
                    print(f"{stmt_id:<10} {gt_count:<12} {extracted_count:<10} {status}")
                    results[stmt_id] = {'gt': gt_count, 'extracted': extracted_count}
                    
                except Exception as e:
                    print(f"{stmt_id:<10} Error: {str(e)[:30]}...")
        
        # Summary
        if results:
            total_gt = sum(r['gt'] for r in results.values())
            total_extracted = sum(r['extracted'] for r in results.values())
            print(f"\nTest Results Summary:")
            print(f"- Ground truth transactions: {total_gt}")
            print(f"- Extracted transactions: {total_extracted}")
            print(f"- Extraction rate: {total_extracted/total_gt:.1%}")
            
    except Exception as e:
        print(f"Error running parser: {e}")
        
    # Show what we know from previous runs
    print(f"\n" + "="*50)
    print(f"COMPARISON WITH PREVIOUS RESULTS:")
    print(f"="*50)
    
    if os.path.exists("enhanced_bank_statement_parsing_report.txt"):
        with open("enhanced_bank_statement_parsing_report.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        
        for line in content.split('\n'):
            if "Total transactions extracted:" in line:
                prev_extracted = int(line.split(':')[1].strip())
                print(f"Previous Enhanced Parser: {prev_extracted} transactions extracted")
                print(f"Previous Extraction Rate: {prev_extracted/total_ground_truth:.1%}")
            elif "Overall accuracy:" in line:
                prev_accuracy = line.split(':')[1].strip()
                print(f"Previous Overall Accuracy: {prev_accuracy}")

if __name__ == "__main__":
    analyze_parsing_output()
