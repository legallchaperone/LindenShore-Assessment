# Blockchain Data Discovery & Analysis

This repository contains tools and analysis for discovering and analyzing blockchain data, specifically focusing on MEV (Maximal Extractable Value) sandwich attacks and liquidation events on the Arbitrum network.

## Project Overview

The goal of this project is to:
1.  **Fetch Data**: Connect to the Arbitrum blockchain via RPC to collect raw data on Uniswap V3 swaps and Aave liquidations.
2.  **Store Data**: Persist the collected data in a local SQLite database (`blockchain.db`) for efficient querying.
3.  **Analyze MEV**: Detect potential sandwich attacks by analyzing swap patterns and calculating the profitability of these attacks.
4.  **Analyze Liquidations**: Calculate the Return on Investment (ROI) for liquidation events, factoring in dynamic gas costs.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Linden
    ```

2.  **Install dependencies**:
    Ensure you have Python 3.8+ installed. Install the required packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Fetching Data
To populate the database with fresh data from the Arbitrum blockchain, run the `fetch_data.py` script.
*Note: This script connects to a public RPC endpoint. Ensure you have a stable internet connection.*

```bash
python fetch_data.py
```

This will create (or update) `blockchain.db` with the latest blocks, transactions, and logs relevant to our analysis.

### 2. MEV Analysis
To inspect MEV sandwich attacks, run the `inspect_mev.py` script. This script queries the database for potential sandwich patterns and outputs the details of detected attacks, including estimated profits.

```bash
python inspect_mev.py
```

### 3. Visualization & Deep Dive
For a comprehensive visual analysis, including liquidation ROI distribution and correlation analysis, open the Jupyter Notebook:

```bash
jupyter notebook analysis_visualization.ipynb
```

This notebook provides:
*   **Liquidation Analysis**: Visualizations of profit distribution and ROI for liquidators.
*   **MEV Insights**: detailed breakdown of victim losses vs. attacker profits.

## Analysis & Insights

### Why this data is interesting
This dataset reveals the "invisible hand" of the blockchain—MEV bots and liquidators. These actors operate in the "Dark Forest" of the mempool, enforcing market efficiency while extracting value. Understanding their behavior is crucial because:
*   **Market Health**: Liquidations are the immune system of DeFi, preventing bad debt.
*   **User Cost**: MEV extraction represents a hidden tax on users (slippage).
*   **Adversarial Environment**: It highlights the highly competitive, zero-sum nature of on-chain arbitrage.

### What we learned
*   **MEV Mechanics**: Sandwich attacks are highly mechanical. Attackers monitor the mempool for large pending swaps and atomically insert their own transactions to profit from the predictable price movement.
*   **Liquidation Economics**: Liquidation is not guaranteed profit. High gas costs on L1 (or even L2 congestion) can erode margins. Successful liquidators must optimize for gas and speed.
*   **Victim Impact**: Retail users trading on popular pairs (like USDC/WETH) with high slippage tolerance are the primary victims of sandwich attacks.

### Limitations
*   **Data Volume**: Our attempt to analyze the correlation between **Concentrated Liquidity** and **MEV Profitability** was inconclusive. Due to the short data collection window, we only identified a single profitable sandwich attack, which is insufficient for statistical analysis. A longer observation period is required to validate the hypothesis that lower liquidity leads to higher MEV profits.

### Potential Applications
1.  **Slippage Protection**: Wallets can use this data to warn users if their transaction is vulnerable to sandwiching before they sign.
2.  **Protocol Design**: DEXs and lending protocols can adjust parameters (e.g., tick spacing, LTV ratios) to minimize MEV extraction and ensure efficient liquidations.
3.  **MEV-Aware RPCs**: Developing private RPC endpoints that bypass the public mempool to protect users from front-running.

## Database Structure

The `blockchain.db` SQLite database consists of the following main tables:
*   `blocks`: Block headers and metadata.
*   `transactions`: Raw transaction data.
*   `logs`: Event logs (e.g., Swap, LiquidationCall) emitted by smart contracts.
