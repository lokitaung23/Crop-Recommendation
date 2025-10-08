"""
evaluate_rl.py — Supabase-only RL uplift evaluation

This script pulls data *directly* from the Supabase `predictions` table.
It does not read SQLite or Excel.

Usage:
  python evaluate_rl.py
  # (optional) filter by farm number:
  python evaluate_rl.py --farm GOMBE-001
  # (optional) limit rows (most recent N):
  python evaluate_rl.py --limit 200

Requires env vars in .env or OS env:
  SUPABASE_URL, SUPABASE_ANON_KEY
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from supabase import create_client

TABLE = "predictions"

def fetch_supabase_df(limit: int | None = None, farm: str | None = None) -> pd.DataFrame:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY. Set them in .env or OS env.")

    sb = create_client(url, key)

    # Build the query
    q = sb.table(TABLE).select("*")
    if farm:
        q = q.eq("FarmNumber", farm)
    # Order by timestamp if present
    try:
        q = q.order("Timestamp")
    except Exception:
        pass
    if limit is not None and limit > 0:
        q = q.limit(limit)

    res = q.execute()
    return pd.DataFrame(res.data or [])

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # Map Feedback -> Reward if Reward missing
    if "Reward" not in df.columns and "Feedback" in df.columns:
        df["Reward"] = df["Feedback"].map({"Good": 1.0, "Not good": 0.0})
    # Ensure required columns exist
    required = ["Reward", "LoggingPropensity", "BaseProbChosen"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for IPS evaluation: {missing}")
    # Clean/convert
    df = df.dropna(subset=required).copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Limit number of rows (most recent).")
    ap.add_argument("--farm", type=str, default=None, help="Filter by FarmNumber.")
    args = ap.parse_args()

    try:
        df = fetch_supabase_df(limit=args.limit, farm=args.farm)
    except Exception as e:
        print(f"❌ Supabase fetch failed: {e}")
        sys.exit(1)

    if df.empty:
        print("No data returned from Supabase. Add rows to the `predictions` table and retry.")
        sys.exit(2)

    try:
        df = prepare_df(df)
    except Exception as e:
        print(f"Data not ready for IPS evaluation: {e}")
        sys.exit(3)

    rl_value = float(df["Reward"].mean())

    # Inverse Propensity Scoring baseline using logging probabilities recorded by the app
    w = np.clip(df["BaseProbChosen"] / np.clip(df["LoggingPropensity"], 1e-8, None), 0, 50)
    ips_base_value = float(np.mean(df["Reward"] * w))

    src = f"Supabase table `{TABLE}`"
    if args.farm:
        src += f" (farm={args.farm})"
    if args.limit:
        src += f" (limit={args.limit})"

    print(f"Loaded from: {src}")
    print(f"Observed RL avg reward:   {rl_value:.3f}")
    print(f"IPS-estimated Base value: {ips_base_value:.3f}")
    print(f"Estimated uplift:         {rl_value - ips_base_value:+.3f}")

    if "RL_Chosen_Crop" in df.columns:
        print("\nPer-crop avg reward (RL chosen):")
        print(df.groupby("RL_Chosen_Crop")["Reward"].mean().sort_values(ascending=False))

if __name__ == "__main__":
    main()
