"""Data ingestion script for Lindenshore assessment.

This script fetches Uniswap V3 swap events and Aave V3 liquidation
calls from Arbitrum One and stores them in SQLite for later analysis.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Sequence

# from dotenv import load_dotenv # Optional if we hardcode or use env
from web3 import Web3
# from web3.middleware import geth_poa_middleware # Removed for Web3.py v7 compatibility

ARBITRUM_RPC_ENV = "ARBITRUM_RPC_URL"
# Fallback URL provided by user
FALLBACK_RPC_URL = "https://arb-mainnet.g.alchemy.com/v2/FBvrCLDYF0rXn-eOa2gVD"

DEFAULT_DB_PATH = Path("blockchain.db") # Changed to current dir for simplicity
DEFAULT_CHUNK_SIZE = 2_000
MIN_CHUNK_SIZE = 250
SLEEP_SECONDS = 0.1

UNISWAP_POOL_ADDRESSES = [
    "0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443",  # USDC/WETH 0.05%
    "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f",  # WBTC/WETH 0.05%
    "0x641C00A822e8b671738d32a431a4Fb6074E5c79d",  # WETH/USDT 0.05%
]
AAVE_POOL_ADDRESS = "0x794a61358D6845594F94dc1DB02A252b5b4814aD"

UNISWAP_POOL_ABI: List[dict] = [
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "sender",
                "type": "address",
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "recipient",
                "type": "address",
            },
            {
                "indexed": False,
                "internalType": "int256",
                "name": "amount0",
                "type": "int256",
            },
            {
                "indexed": False,
                "internalType": "int256",
                "name": "amount1",
                "type": "int256",
            },
            {
                "indexed": False,
                "internalType": "uint160",
                "name": "sqrtPriceX96",
                "type": "uint160",
            },
            {
                "indexed": False,
                "internalType": "uint128",
                "name": "liquidity",
                "type": "uint128",
            },
            {
                "indexed": False,
                "internalType": "int24",
                "name": "tick",
                "type": "int24",
            },
        ],
        "name": "Swap",
        "type": "event",
    }
]

AAVE_POOL_ABI: List[dict] = [
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "collateralAsset",
                "type": "address",
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "debtAsset",
                "type": "address",
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "user",
                "type": "address",
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "debtToCover",
                "type": "uint256",
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "liquidatedCollateralAmount",
                "type": "uint256",
            },
            {
                "indexed": False,
                "internalType": "address",
                "name": "liquidator",
                "type": "address",
            },
            {
                "indexed": False,
                "internalType": "bool",
                "name": "receiveAToken",
                "type": "bool",
            },
        ],
        "name": "LiquidationCall",
        "type": "event",
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Arbitrum logs into SQLite")
    parser.add_argument(
        "--from-block", dest="from_block", type=int, help="Starting block (inclusive)"
    )
    parser.add_argument(
        "--to-block", dest="to_block", type=int, help="Ending block (inclusive)"
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of blocks to query at a time",
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default=str(DEFAULT_DB_PATH),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--skip-uniswap",
        dest="skip_uniswap",
        action="store_true",
        help="Skip fetching Uniswap swap logs",
    )
    parser.add_argument(
        "--skip-aave",
        dest="skip_aave",
        action="store_true",
        help="Skip fetching Aave liquidation logs",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def init_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def connect_web3(rpc_url: str) -> Web3:
    provider = Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30})
    w3 = Web3(provider)
    # w3.middleware_onion.inject(geth_poa_middleware, layer=0) # Removed for v7
    if not w3.is_connected():
        raise ConnectionError("Unable to connect to Arbitrum RPC provider")
    logging.info("Connected to Arbitrum at block %s", w3.eth.block_number)
    return w3


def init_db(path: Path) -> sqlite3.Connection:
    # path.parent.mkdir(parents=True, exist_ok=True) # Assumes path has parent
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS uniswap_swaps_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_hash TEXT NOT NULL,
            block_number INTEGER NOT NULL,
            block_timestamp INTEGER,
            log_index INTEGER NOT NULL,
            tx_from TEXT,
            sender TEXT,
            recipient TEXT,
            amount0 TEXT,
            amount1 TEXT,
            sqrt_price_x96 TEXT,
            liquidity TEXT,
            tick INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(transaction_hash, log_index)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aave_liquidations_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_hash TEXT NOT NULL,
            block_number INTEGER NOT NULL,
            block_timestamp INTEGER,
            log_index INTEGER NOT NULL,
            collateral_asset TEXT,
            debt_asset TEXT,
            user TEXT,
            debt_covered TEXT,
            collateral_amount TEXT,
            liquidator TEXT,
            receive_a_token INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(transaction_hash, log_index)
        )
        """
    )
    return conn


def chunk_ranges(start: int, end: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    current = start
    while current <= end:
        chunk_end = min(current + chunk_size - 1, end)
        yield current, chunk_end
        current = chunk_end + 1


def fetch_logs(
    w3: Web3,
    addresses: List[str] | str,
    topics: List[str] | None,
    start_block: int,
    end_block: int,
    chunk_size: int,
) -> Iterable[dict]:
    current_chunk = min(chunk_size, DEFAULT_CHUNK_SIZE)
    from_block = start_block
    retries = 0

    # Normalize addresses to checksum
    if isinstance(addresses, list):
        checksum_addresses = [w3.to_checksum_address(a) for a in addresses]
    else:
        checksum_addresses = w3.to_checksum_address(addresses)

    while from_block <= end_block:
        to_block = min(from_block + current_chunk - 1, end_block)
        logging.debug("Fetching logs %s-%s", from_block, to_block)

        filter_params = {
            "fromBlock": from_block,
            "toBlock": to_block,
            "address": checksum_addresses,
        }
        if topics:
            filter_params["topics"] = topics

        try:
            logs = w3.eth.get_logs(filter_params)
            for entry in logs:
                yield entry
            from_block = to_block + 1
            retries = 0
            if current_chunk < chunk_size:
                current_chunk = min(chunk_size, current_chunk * 2)
            time.sleep(SLEEP_SECONDS)
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Error fetching logs %s-%s (%s). Reducing chunk size.",
                from_block,
                to_block,
                exc,
            )
            if current_chunk <= MIN_CHUNK_SIZE:
                retries += 1
                if retries > 5:
                    raise
            current_chunk = max(MIN_CHUNK_SIZE, current_chunk // 2)
            time.sleep(SLEEP_SECONDS * 2)


def persist_uniswap(conn: sqlite3.Connection, logs: Sequence[dict]) -> None:
    rows = [
        (
            log["transactionHash"],
            log["blockNumber"],
            log["blockTimestamp"],
            log["logIndex"],
            log["txFrom"],
            log["args"]["sender"],
            log["args"]["recipient"],
            str(log["args"]["amount0"]),
            str(log["args"]["amount1"]),
            str(log["args"]["sqrtPriceX96"]),
            str(log["args"]["liquidity"]),
            log["args"]["tick"],
        )
        for log in logs
    ]
    conn.executemany(
        """
        INSERT OR IGNORE INTO uniswap_swaps_v2 (
            transaction_hash, block_number, block_timestamp, log_index, tx_from, sender, recipient,
            amount0, amount1, sqrt_price_x96, liquidity, tick
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def persist_aave(conn: sqlite3.Connection, logs: Sequence[dict]) -> None:
    rows = [
        (
            log["transactionHash"],
            log["blockNumber"],
            log["blockTimestamp"],
            log["logIndex"],
            log["args"]["collateralAsset"],
            log["args"]["debtAsset"],
            log["args"]["user"],
            str(log["args"]["debtToCover"]),
            str(log["args"]["liquidatedCollateralAmount"]),
            log["args"]["liquidator"],
            int(log["args"]["receiveAToken"]),
        )
        for log in logs
    ]
    conn.executemany(
        """
        INSERT OR IGNORE INTO aave_liquidations_v2 (
            transaction_hash, block_number, block_timestamp, log_index, collateral_asset,
            debt_asset, user, debt_covered, collateral_amount,
            liquidator, receive_a_token
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def _process_uniswap_batch(
    w3: Web3, contract: Any, logs: List[dict], conn: sqlite3.Connection
) -> None:
    if not logs:
        return

    # 1. Identify unique blocks
    block_numbers = set(log["blockNumber"] for log in logs)

    # 2. Fetch blocks and build tx_hash -> sender map
    tx_sender_map = {}
    block_timestamp_map = {}

    logging.info("Batch fetching %d blocks for %d logs...", len(block_numbers), len(logs))

    for block_num in block_numbers:
        try:
            # Fetch full transactions to get 'from' address
            block = w3.eth.get_block(block_num, full_transactions=True)
            block_timestamp_map[block_num] = block["timestamp"]
            for tx in block["transactions"]:
                # tx is a dict (AttributeDict)
                tx_hash = tx["hash"].hex()
                tx_sender_map[tx_hash] = tx["from"]
        except Exception as e:
            logging.warning(f"Failed to fetch block {block_num}: {e}")

    # 3. Process logs
    buffer = []
    for log in logs:
        try:
            decoded = contract.events.Swap().process_log(log)
            decoded = dict(decoded)

            tx_hash = decoded["transactionHash"].hex()
            decoded["transactionHash"] = tx_hash

            block_num = decoded["blockNumber"]
            decoded["blockTimestamp"] = block_timestamp_map.get(block_num)

            # Lookup sender from our local map instead of RPC
            decoded["txFrom"] = tx_sender_map.get(tx_hash)

            buffer.append(decoded)
        except Exception as e:
            logging.error(f"Error processing log: {e}")
            continue

    # 4. Persist
    if buffer:
        persist_uniswap(conn, buffer)
        logging.info("Persisted batch of %s swap logs", len(buffer))


def determine_block_range(
    w3: Web3, start: int | None, end: int | None
) -> tuple[int, int]:
    latest = w3.eth.block_number
    resolved_end = end or latest
    resolved_start = start or (resolved_end - 100) # Default to 100 blocks for demo
    resolved_start = max(0, resolved_start)
    if resolved_start > resolved_end:
        raise ValueError("from-block must be <= to-block")
    return resolved_start, resolved_end


def main() -> None:
    # load_dotenv()
    args = parse_args()
    init_logging(args.log_level)

    if args.chunk_size < MIN_CHUNK_SIZE:
        logging.warning(
            "Chunk size %s too small; bumping to minimum %s",
            args.chunk_size,
            MIN_CHUNK_SIZE,
        )
        args.chunk_size = MIN_CHUNK_SIZE

    rpc_url = os.getenv(ARBITRUM_RPC_ENV) or FALLBACK_RPC_URL
    if not rpc_url:
        logging.error("Missing %s in environment or .env file", ARBITRUM_RPC_ENV)
        sys.exit(1)

    w3 = connect_web3(rpc_url)
    start_block, end_block = determine_block_range(w3, args.from_block, args.to_block)
    logging.info("Fetching logs from blocks %s-%s", start_block, end_block)

    conn = init_db(Path(args.db_path))

    # Cache for block timestamps to avoid repeated RPC calls
    block_timestamp_cache = {}

    if not args.skip_uniswap:
        # Create a dummy contract to use for decoding logs
        dummy_swap_contract = w3.eth.contract(abi=UNISWAP_POOL_ABI)
        swap_topic = (
            "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
        )

        buffer = []
        # Collect logs first to batch process blocks
        temp_logs = []
        for log in fetch_logs(
            w3,
            UNISWAP_POOL_ADDRESSES,
            [swap_topic],
            start_block,
            end_block,
            args.chunk_size,
        ):
            temp_logs.append(log)

            # Process in batches to keep memory usage reasonable
            if len(temp_logs) >= 2000:
                _process_uniswap_batch(w3, dummy_swap_contract, temp_logs, conn)
                temp_logs = []

        # Process remaining
        if temp_logs:
            _process_uniswap_batch(w3, dummy_swap_contract, temp_logs, conn)

    if not args.skip_aave:
        # Aave pool is a single address
        dummy_aave_contract = w3.eth.contract(abi=AAVE_POOL_ABI)
        liq_topic = "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286"

        buffer = []
        for log in fetch_logs(
            w3, AAVE_POOL_ADDRESS, [liq_topic], start_block, end_block, args.chunk_size
        ):
            try:
                decoded = dummy_aave_contract.events.LiquidationCall().process_log(log)
                decoded = dict(decoded)

                # Enrich data
                tx_hash = decoded["transactionHash"].hex()
                decoded["transactionHash"] = tx_hash

                block_number = decoded["blockNumber"]
                if block_number not in block_timestamp_cache:
                    block = w3.eth.get_block(block_number)
                    block_timestamp_cache[block_number] = block["timestamp"]
                decoded["blockTimestamp"] = block_timestamp_cache[block_number]

                buffer.append(decoded)
            except Exception:
                continue

            if len(buffer) >= 200:
                persist_aave(conn, buffer)
                logging.info("Persisted %s liquidation logs", len(buffer))
                buffer.clear()
        if buffer:
            persist_aave(conn, buffer)
            logging.info("Persisted final %s liquidation logs", len(buffer))

    logging.info("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
