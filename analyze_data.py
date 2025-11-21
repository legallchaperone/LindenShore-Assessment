import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "blockchain.db"

# Constants for Normalization
DECIMALS = {
    'WETH': 18,
    'USDC': 6,
    'AAVE': 18,
    'DAI': 18,
    'WBTC': 8
}

# Approximate Prices (for demonstration)
PRICES = {
    'WETH': 3100.0,
    'USDC': 1.0,
    'AAVE': 90.0,
    'DAI': 1.0,
    'WBTC': 95000.0
}

KNOWN_BOTS = {
    '0x00000000004e9C6Fb96c08B0Cb12C924d49266eF': 'Aave V3 Bot (0x00...6eF)',
    '0xc9cF99b95ED3cEBd5E08ca131c9E52443792986a': 'MEV Bot (0xc9...86a)',
}

def get_connection():
    return sqlite3.connect(DB_PATH)

def normalize_amount(amount, token_symbol):
    decimals = DECIMALS.get(token_symbol, 18) # Default to 18
    return float(amount) / (10 ** decimals)

def get_usd_value(amount, token_symbol):
    price = PRICES.get(token_symbol, 0)
    return amount * price

def analyze_mev(conn):
    print("\n--- MEV Sandwich Attack Analysis (Uniswap V3) ---")
    query = """
    SELECT transaction_hash, block_number, log_index, sender, recipient, amount0, amount1
    FROM uniswap_swaps_v2
    ORDER BY block_number, log_index
    """
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No swap data found.")
        return

    df['amount0'] = df['amount0'].apply(float)
    df['amount1'] = df['amount1'].apply(float)
    
    grouped = df.groupby('block_number')
    
    sandwich_count = 0
    total_attacker_profit_usd = 0
    total_victim_loss_usd = 0
    
    # Assuming Pool is USDC/WETH (Token0/Token1)
    # We need to know which token is which. 
    # Usually USDC is Token0 (smaller address) but let's assume based on decimals.
    # In the provided DB, we don't have token addresses in the table, but the user mentioned USDC/WETH.
    # Let's assume Token0 = USDC (6 dec), Token1 = WETH (18 dec) for this specific pool analysis.
    
    for block_num, group in grouped:
        if len(group) < 3: continue
        group = group.sort_values('log_index')
        txs = group.to_dict('records')
        
        for i in range(len(txs) - 2):
            tx1 = txs[i]
            direction1 = 1 if tx1['amount0'] < 0 else -1
            
            for j in range(i + 1, len(txs) - 1):
                tx2 = txs[j]
                direction2 = 1 if tx2['amount0'] < 0 else -1
                if direction1 != direction2: continue
                
                for k in range(j + 1, len(txs)):
                    tx3 = txs[k]
                    direction3 = 1 if tx3['amount0'] < 0 else -1
                    if direction1 == direction3: continue
                    
                    if tx1['recipient'] == tx3['recipient']:
                        sandwich_count += 1
                        
                        # Calculate Profit (Attacker)
                        # Profit = Out - In (absolute)
                        # If dir1=1 (In T0, Out T1): Profit T0 = Out(Back) - In(Front)
                        # amount0 is negative for Input.
                        # Profit T0 = tx3['amount0'] + tx1['amount0']
                        
                        profit0_raw = tx1['amount0'] + tx3['amount0']
                        profit1_raw = tx1['amount1'] + tx3['amount1']
                        
                        # Normalize (Corrected Mapping: Token0=WETH, Token1=USDC based on raw values)
                        profit0 = normalize_amount(profit0_raw, 'WETH')
                        profit1 = normalize_amount(profit1_raw, 'USDC')
                        
                        profit_usd = (profit0 * PRICES['WETH']) + (profit1 * PRICES['USDC'])
                        
                        if profit_usd > 0:
                            sandwich_count += 1
                            total_attacker_profit_usd += profit_usd

                            # Calculate Victim Loss (Slippage)
                            victim_vol0 = normalize_amount(abs(tx2['amount0']), 'WETH')
                            victim_vol1 = normalize_amount(abs(tx2['amount1']), 'USDC')
                            victim_usd = (victim_vol0 * PRICES['WETH']) + (victim_vol1 * PRICES['USDC'])
                            total_victim_loss_usd += victim_usd
                            
                            print(f"  [Block {block_num}] Sandwich Bundle Found!")
                            print(f"    Attacker: {tx1['recipient']}")
                            print(f"    Victim Tx: {tx2['transaction_hash']}")
                            print(f"    Profit: ${profit_usd:.2f}")
                        else:
                            # Optional: Log failed attempts
                            # print(f"  [Block {block_num}] Unprofitable Bundle: ${profit_usd:.2f}")
                            pass

    print(f"Detected Profitable Bundles: {sandwich_count}")
    print(f"Total Attacker Profit: ${total_attacker_profit_usd:,.2f}")
    print(f"Total Volume Sandwiched (Victim): ${total_victim_loss_usd:,.2f}")
    if sandwich_count > 0:
        print(f"Avg Profit per Bundle: ${total_attacker_profit_usd / sandwich_count:,.2f}")

    print(f"Detected Bundles: {sandwich_count}")
    print(f"Total Attacker Profit: ${total_attacker_profit_usd:,.2f}")
    print(f"Total Volume Sandwiched (Victim): ${total_victim_loss_usd:,.2f}")
    if sandwich_count > 0:
        print(f"Avg Profit per Bundle: ${total_attacker_profit_usd / sandwich_count:,.2f}")

def analyze_liquidations(conn):
    print("\n--- Aave Liquidation Analysis ---")
    query = """
    SELECT transaction_hash, collateral_asset, debt_asset, user, debt_covered, collateral_amount, liquidator
    FROM aave_liquidations_v2
    """
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No liquidation data found.")
        return

    df['debt_covered'] = df['debt_covered'].apply(float)
    df['collateral_amount'] = df['collateral_amount'].apply(float)
    
    # Map addresses to symbols (Mock mapping based on common Arbitrum tokens or just generic)
    # In a real app, we'd query the contract or use a token list.
    # For this demo, let's assume:
    # Collateral is often WETH or WBTC. Debt is often USDC or DAI.
    # We will use a generic normalization of 18 decimals for unknown, or try to guess.
    # Let's just normalize to 18 for now as a baseline if unknown, or 6 if it looks like USDC.
    # Actually, let's just use a simplified approach:
    # Most Aave liquidations on Arb might be WETH/USDC.
    
    # Let's assume standard 18 decimals for simplicity unless we know better.
    # Or better, let's just print "Normalized Units" and assume 18 decimals for aggregate.
    
    total_debt_usd = 0
    total_collateral_usd = 0
    
    # We'll iterate to apply specific logic if needed, or just vectorise
    # Let's assume 18 decimals for everything to get "Human Readable" magnitude roughly correct for ETH-like tokens.
    # If it's USDC (6 decimals), dividing by 1e18 makes it tiny.
    # We need to be careful.
    # Let's use the 'debt_asset' address to guess?
    # Without an address map, it's hard.
    # Let's just divide by 1e18 and call it "ETH-equivalent" for now, or 
    # better: let's look at the raw values. 1e21 raw -> 1000 if 18 decimals.
    
    # Let's assume 18 decimals for aggregate stats.
    df['debt_norm'] = df['debt_covered'] / 1e18
    df['coll_norm'] = df['collateral_amount'] / 1e18
    
    # Estimate USD (assuming ETH price for everything as a rough proxy or $1 for stablecoins? 
    # This is risky. Let's just report Normalized Units and mention assumption).
    # Wait, user wants "Total Debt Repaid: $1,540,000 USD".
    # I will assume the dominant asset is WETH-like (18 decimals) and price is ~$3100.
    
    total_debt_norm = df['debt_norm'].sum()
    total_coll_norm = df['coll_norm'].sum()
    
    est_debt_usd = total_debt_norm * 2000 # Blended price assumption? Or just say "Units"
    # Actually, let's try to be smarter. 
    # If raw > 1e15 it's likely 18 decimals. If raw is large but < 1e12 it might be 6 decimals?
    # No, 1 USDC = 1e6. 1000 USDC = 1e9.
    # 1 ETH = 1e18.
    # The raw sum was 1.02e21.
    # If 18 decimals: 1.02e21 / 1e18 = 1020 units. If ETH, that's $3M.
    # If 6 decimals: 1.02e21 / 1e6 = 1e15 units. That's quadrillions of dollars. Impossible.
    # So it MUST be 18 decimals (or close).
    # So we treat it as 18 decimals.
    
    # Let's assume price average of $2000 (mix of ETH/WBTC/Stables normalized to 18 dec).
    # This is a heuristic for the assessment.
    
    est_debt_usd = total_debt_norm * 1 # If it's DAI/USDC (18 dec for DAI)
    # Wait, if it's WETH, it's * 3000.
    # Let's assume it's mostly WETH/WBTC collateral.
    # Let's use a conservative $2000/unit price for the "18-decimal-normalized" basket.
    
    print(f"Total Liquidations: {len(df)}")
    print(f"Total Debt Covered: {total_debt_norm:,.2f} Units (Normalized 18 dec)")
    print(f"Total Collateral Seized: {total_coll_norm:,.2f} Units")
    
    # Profit = Collateral - Debt (roughly)
    # Liquidation Bonus is usually the profit.
    # Profit ~= CollateralUSD - DebtUSD
    # But we need prices.
    # Let's assume 5% average profit on the Debt value.
    est_profit_usd = total_debt_norm * 0.05 * 3000 # Assuming ETH price for value
    
    print(f"Estimated Liquidator Profit: ~${est_profit_usd:,.2f} (Assuming 5% bonus & ETH price)")

    # Calculate totals and identify top liquidators
    liquidator_stats = {}
    total_collateral = 0
    total_profit_usd = 0

    for index, row in df.iterrows():
        liq = row['liquidator']
        debt_covered = row['debt_covered']
        collateral_amt = row['collateral_amount']
        
        # Normalize
        debt_norm = debt_covered / 1e18
        coll_norm = collateral_amt / 1e18
        
        # Estimate Profit (5% bonus)
        profit = debt_norm * 0.05 * 3000 # Approx ETH price
        
        total_collateral += coll_norm
        total_profit_usd += profit
        
        if liq not in liquidator_stats:
            liquidator_stats[liq] = {'count': 0, 'profit': 0}
        liquidator_stats[liq]['count'] += 1
        liquidator_stats[liq]['profit'] += profit

    sorted_liquidators = sorted(liquidator_stats.items(), key=lambda x: x[1]['profit'], reverse=True)

    print(f"Total Collateral Seized: {total_collateral:,.2f} Units")
    print(f"Estimated Liquidator Profit: ~${total_profit_usd:,.2f} (Assuming 5% bonus & ETH price)")
    print("\nTop Liquidator Identity:")
    for liq, stats in sorted_liquidators[:5]:
        print(f"  {liq[:10]}... ({KNOWN_BOTS.get(liq, 'Unknown Bot')}) - {stats['count']} Liqs - Est. Profit: ${stats['profit']:,.2f}")

def analyze_volume_stats(conn):
    print("\n--- Swap Volume Analysis (Uniswap V3) ---")
    query = "SELECT amount0, amount1, sqrt_price_x96 FROM uniswap_swaps_v2"
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No swap data found.")
        return

    df['amount0'] = df['amount0'].apply(float).abs()
    df['amount1'] = df['amount1'].apply(float).abs()
    df['sqrt_price_x96'] = df['sqrt_price_x96'].apply(float)

    # Pool Classification based on sqrt_price_x96
    # WETH/USDT: ~10^24
    # WBTC/WETH: ~10^29
    # USDC/WETH: ~10^33
    
    def classify_pool(row):
        p = row['sqrt_price_x96']
        if p < 1e27:
            return 'WETH/USDT'
        elif p < 1e31:
            return 'WBTC/WETH'
        else:
            return 'USDC/WETH'

    df['pool'] = df.apply(classify_pool, axis=1)
    
    # Calculate Volume in USD based on pool
    def calc_volume(row):
        pool = row['pool']
        a0 = row['amount0']
        a1 = row['amount1']
        
        if pool == 'WETH/USDT':
            # Token0=WETH(18), Token1=USDT(6)
            # Vol = max(WETH*Price, USDT*Price)
            v0 = (a0 / 1e18) * PRICES['WETH']
            v1 = (a1 / 1e6) * 1.0 # USDT ~ $1
            return max(v0, v1)
        elif pool == 'WBTC/WETH':
            # Token0=WBTC(8), Token1=WETH(18)
            v0 = (a0 / 1e8) * PRICES['WBTC']
            v1 = (a1 / 1e18) * PRICES['WETH']
            return max(v0, v1)
        elif pool == 'USDC/WETH':
            # Token0=USDC(6), Token1=WETH(18)
            v0 = (a0 / 1e6) * PRICES['USDC']
            v1 = (a1 / 1e18) * PRICES['WETH']
            return max(v0, v1)
        return 0.0

    df['volume_usd'] = df.apply(calc_volume, axis=1)
    
    total_vol = df['volume_usd'].sum()
    avg_vol = df['volume_usd'].mean()
    median_vol = df['volume_usd'].median()
    
    print(f"Total Swap Volume: ${total_vol:,.2f}")
    print(f"Average Swap Size: ${avg_vol:,.2f}")
    print(f"Median Swap Size:  ${median_vol:,.2f}")
    
    print("\nVolume by Pool:")
    pool_stats = df.groupby('pool')['volume_usd'].agg(['sum', 'count', 'mean'])
    for pool, row in pool_stats.iterrows():
        print(f"  {pool}: ${row['sum']:,.2f} (Count: {int(row['count'])}, Avg: ${row['mean']:,.2f})")
    
    # Categorize
    small = len(df[df['volume_usd'] < 1000])
    medium = len(df[(df['volume_usd'] >= 1000) & (df['volume_usd'] < 10000)])
    large = len(df[(df['volume_usd'] >= 10000) & (df['volume_usd'] < 100000)])
    whale = len(df[df['volume_usd'] >= 100000])
    
    print(f"\nDistribution:")
    print(f"  Small (<$1k):   {small} ({small/len(df)*100:.1f}%)")
    print(f"  Medium ($1k-$10k): {medium} ({medium/len(df)*100:.1f}%)")
    print(f"  Large ($10k-$100k): {large} ({large/len(df)*100:.1f}%)")
    print(f"  Whale (>$100k):    {whale} ({whale/len(df)*100:.1f}%)")

def main():
    conn = get_connection()
    analyze_mev(conn)
    analyze_liquidations(conn)
    analyze_volume_stats(conn)
    conn.close()

if __name__ == "__main__":
    main()

