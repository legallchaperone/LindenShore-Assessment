import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "blockchain.db"
DECIMALS = {'WETH': 18, 'USDC': 6, 'AAVE': 18, 'DAI': 18, 'WBTC': 8}
PRICES = {'WETH': 3100.0, 'USDC': 1.0, 'AAVE': 90.0, 'DAI': 1.0, 'WBTC': 95000.0}

def get_connection():
    return sqlite3.connect(DB_PATH)

def normalize_amount(amount, token_symbol):
    decimals = DECIMALS.get(token_symbol, 18)
    return float(amount) / (10 ** decimals)

conn = get_connection()
# Query matches the FIXED notebook query
query = """
SELECT transaction_hash, block_number, block_timestamp, log_index, sender, recipient, amount0, amount1, sqrt_price_x96, liquidity
FROM uniswap_swaps_v2
ORDER BY block_number, log_index
"""
print("Fetching data...")
df_swaps = pd.read_sql_query(query, conn)
print(f"Total swaps fetched: {len(df_swaps)}")

if df_swaps.empty:
    print("No swaps found in DB!")
    exit()

df_swaps['amount0'] = df_swaps['amount0'].apply(float)
df_swaps['amount1'] = df_swaps['amount1'].apply(float)
df_swaps['sqrt_price_x96'] = df_swaps['sqrt_price_x96'].apply(float)

# Detect Sandwiches
grouped = df_swaps.groupby('block_number')
sandwiches = []

print("Scanning for sandwiches...")
for block_num, group in grouped:
    if len(group) < 3: continue
    group = group.sort_values('log_index')
    txs = group.to_dict('records')
    
    for i in range(len(txs) - 2):
        tx1 = txs[i]
        dir1 = 1 if tx1['amount0'] < 0 else -1
        for j in range(i + 1, len(txs) - 1):
            tx2 = txs[j]
            dir2 = 1 if tx2['amount0'] < 0 else -1
            if dir1 != dir2: continue
            for k in range(j + 1, len(txs)):
                tx3 = txs[k]
                dir3 = 1 if tx3['amount0'] < 0 else -1
                if dir1 == dir3: continue
                
                if tx1['recipient'] == tx3['recipient']:
                    # Calculate Profit
                    profit0_raw = tx1['amount0'] + tx3['amount0']
                    profit1_raw = tx1['amount1'] + tx3['amount1']
                    profit0 = normalize_amount(profit0_raw, 'WETH')
                    profit1 = normalize_amount(profit1_raw, 'USDC')
                    profit_usd = (profit0 * PRICES['WETH']) + (profit1 * PRICES['USDC'])
                    
                    # if profit_usd > 0:
                    sandwiches.append({
                        'block_number': block_num,
                        'attacker': tx1['recipient'],
                        'profit_usd': profit_usd,
                        'victim_tx_hash': tx2['transaction_hash']
                    })

df_mev = pd.DataFrame(sandwiches)
print(f"Found {len(df_mev)} profitable sandwiches")

if not df_mev.empty:
    print("Sample sandwich:")
    print(df_mev.head(1))
    
    # Merge logic
    df_victim_liquidity = df_swaps[['transaction_hash', 'liquidity']].copy()
    # Check if liquidity is populated
    print(f"Liquidity sample: {df_victim_liquidity['liquidity'].head()}")
    
    df_victim_liquidity['liquidity'] = df_victim_liquidity['liquidity'].apply(float)
    
    df_analysis = df_mev.merge(df_victim_liquidity, left_on='victim_tx_hash', right_on='transaction_hash', how='left')
    print(f"Merged analysis rows: {len(df_analysis)}")
    print("Sample analysis row:")
    print(df_analysis.head(1))
else:
    print("No sandwiches to analyze.")
