import pandas as pd

def analyze_ground_truth():
    """Analyze the ground truth data to show transaction counts per statement"""
    
    # Load ground truth data
    df = pd.read_csv("bank statement generator/bank_statements/ground_truth.csv")
    
    print("GROUND TRUTH ANALYSIS")
    print("=" * 50)
    
    # Total transactions
    total_transactions = len(df)
    print(f"Total transactions in ground truth: {total_transactions}")
    
    # Transactions per statement
    statement_counts = df['statement_id'].value_counts().sort_index()
    
    print(f"\nTransactions per statement:")
    print("-" * 30)
    
    for statement_id, count in statement_counts.items():
        print(f"Statement {statement_id:3d}: {count} transactions")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print("-" * 20)
    print(f"Number of statements: {len(statement_counts)}")
    print(f"Average transactions per statement: {statement_counts.mean():.2f}")
    print(f"Min transactions per statement: {statement_counts.min()}")
    print(f"Max transactions per statement: {statement_counts.max()}")
    
    # Show some example transactions
    print(f"\nExample transactions from first few statements:")
    print("-" * 50)
    for stmt_id in [1, 2, 3]:
        stmt_transactions = df[df['statement_id'] == stmt_id]
        print(f"\nStatement {stmt_id} ({len(stmt_transactions)} transactions):")
        for _, trans in stmt_transactions.iterrows():
            print(f"  - {trans['date']}: {trans['amount']:.2f} EUR - {trans['identified_partner']}")

if __name__ == "__main__":
    analyze_ground_truth()
