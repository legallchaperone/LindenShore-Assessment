import sqlite3
import pandas as pd

"""
MEV Sandwich Attack Inspector

This script queries the local SQLite database for potential MEV sandwich attacks
on Uniswap V3. It identifies patterns where a victim transaction is sandwiched
between a front-run and back-run transaction by the same attacker, and calculates
the estimated profit.
"""

conn = sqlite3.connect('blockchain.db')
query = """
    SELECT transaction_hash, block_number, log_index, sender, recipient, amount0, amount1
    FROM uniswap_swaps_v2
    ORDER BY block_number, log_index
"""
df = pd.read_sql_query(query, conn)
df['amount0'] = df['amount0'].apply(float)
df['amount1'] = df['amount1'].apply(float)

grouped = df.groupby('block_number')

print("--- Detailed Sandwich Inspection ---")

for block_num, group in grouped:
    if len(group) < 3: continue
    group = group.sort_values('log_index')
    txs = group.to_dict('records')
    
    for i in range(len(txs) - 2):
        tx1 = txs[i]
        # Check direction (assuming amount0 < 0 is input)
        dir1 = 1 if tx1['amount0'] < 0 else -1
        
        for j in range(i + 1, len(txs) - 1):
            tx2 = txs[j]
            dir2 = 1 if tx2['amount0'] < 0 else -1
            
            if dir1 != dir2: continue # Victim must be same direction
            
            for k in range(j + 1, len(txs)):
                tx3 = txs[k]
                dir3 = 1 if tx3['amount0'] < 0 else -1
                
                if dir1 == dir3: continue # Attacker closes (opposite)
                
                if tx1['recipient'] == tx3['recipient']:
                    # Found one
                    print(f"\nBlock {block_num}:")
                    print(f"  Front: {tx1['transaction_hash'][:10]}... | In: {tx1['amount0']:.4f} | Out: {tx1['amount1']:.4f}")
                    print(f"  Victim: {tx2['transaction_hash'][:10]}... | In: {tx2['amount0']:.4f} | Out: {tx2['amount1']:.4f}")
                    print(f"  Back:  {tx3['transaction_hash'][:10]}... | In: {tx3['amount0']:.4f} | Out: {tx3['amount1']:.4f}")
                    
                    # Calculate Profit (in terms of Token 0)
                    # Front: Input A, Output B
                    # Back: Input B, Output A
                    # Net = Back_Out_A - Front_In_A (absolute)
                    
                    # Example: 
                    # Front: -100 A -> +200 B
                    # Back:  -200 B -> +105 A
                    # Profit: 105 - 100 = 5 A
                    
                    # If dir1 == 1 (amount0 < 0, inputting 0):
                    # Front In = abs(tx1['amount0'])
                    # Back Out = tx3['amount0']
                    # Profit = tx3['amount0'] + tx1['amount0'] (since tx1 is neg)
                    
                    profit0 = tx1['amount0'] + tx3['amount0']
                    profit1 = tx1['amount1'] + tx3['amount1']
                    
                    print(f"  Net Change: Token0: {profit0:.4f}, Token1: {profit1:.4f}")
