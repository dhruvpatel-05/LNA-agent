"""
src/run.py — CLI entry point for the LNA pipeline.

Usage
-----
  python src/run.py build   [--tickers AAPL AMD ...] [--date-from YYYY-MM-DD]
                            [--date-to YYYY-MM-DD] [--limit N] [--no-llm]
  python src/run.py agent   [--tickers AAPL AMD ...] [--limit N] [--dry-run]
  python src/run.py analyze [--tickers AAPL AMD ...]
                            [--signals ofi ofi_x_llm ...] [--horizons 1 5 15]
                            [--returns raw|excess|both] [--with-agent]
                            [--out PATH]
  python src/run.py all     [same flags as above combined]

Stages
------
  build   : Load raw LOB, compute OFI/returns; score headlines (LM + LLM);
            assemble per-event panels. Saves to data/clean10_2025/{ofi,sentiment,panel}/.
  agent   : Run the LLM trading agent over the assembled panels. Resumable.
            Saves to data/clean10_2025/agent/{ticker}.jsonl.
  analyze : Compute IC table, regression table, PnL table, and headline comparison.
            Saves to results/tables/ (CSV) and prints summary.
  all     : build → agent → analyze in sequence.

Does NOT import from legacy/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── ensure project root is on path ────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import HORIZONS, RESULTS_DIR, load_tickers


# ── Stage: build ──────────────────────────────────────────────────────────────

def stage_build(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.config import NEWS_DIR, OFI_DIR, SENTIMENT_DIR, PANEL_DIR
    from src.data.sentiment import ensure_scored, save_sentiment
    from src.data.panel import assemble_ticker, save_panel

    tickers        = args.tickers or load_tickers()
    no_llm         = getattr(args, "no_llm",         False)
    sentiment_only = getattr(args, "sentiment_only", False)
    force          = getattr(args, "force",          False)

    for ticker in tickers:
        print(f"\n── {ticker} ── build ──────────────────────────────────────")

        # 1. Score headlines
        news_path = NEWS_DIR / f"{ticker}.csv"
        if not news_path.exists():
            print(f"  [build] SKIP {ticker}: no news file — run scripts/fetch_alpaca_news.py")
            continue

        sent_path = SENTIMENT_DIR / f"{ticker}.csv"
        if sent_path.exists() and not force:
            print(f"  [build] sentiment cached — skipping scoring for {ticker}")
        else:
            headlines = pd.read_csv(news_path)
            headlines["timestamp"] = pd.to_datetime(headlines["timestamp"])
            print(f"  [build] scoring {len(headlines)} headlines for {ticker}")
            scored = ensure_scored(ticker, headlines, llm=not no_llm)
            save_sentiment(ticker, scored)

        if sentiment_only:
            continue

        # 2. Join scored headlines with OFI bars → per-event panel
        if not (OFI_DIR / f"{ticker}.csv").exists():
            print(f"  [build] SKIP panel for {ticker}: no OFI file")
            continue

        panel_path = PANEL_DIR / f"{ticker}.csv"
        if panel_path.exists() and not force:
            print(f"  [build] panel cached — skipping assembly for {ticker}")
            continue

        print(f"  [build] assembling panel for {ticker}")
        panel = assemble_ticker(ticker)
        save_panel(ticker, panel)


# ── Stage: agent ──────────────────────────────────────────────────────────────

def stage_agent(args: argparse.Namespace) -> None:
    from src.data.panel import load_panel
    from src.signals.agent import score_ticker

    tickers = args.tickers or load_tickers()
    limit   = getattr(args, "limit",   None)
    dry_run = getattr(args, "dry_run", False)

    if dry_run:
        print("  [agent] DRY RUN — no API calls will be made.")

    for ticker in tickers:
        print(f"\n── {ticker} ── agent ──────────────────────────────────────")
        panel = load_panel(ticker)
        score_ticker(ticker, panel, limit=limit, dry_run=dry_run)


# ── Stage: analyze ────────────────────────────────────────────────────────────

def stage_analyze(args: argparse.Namespace) -> None:
    import pandas as pd

    from src.config import HORIZONS as DEFAULT_HORIZONS
    from src.data.panel import load_panel, load_pool
    from src.signals.rules import add_rule_signals, RULE_SIGNALS
    from src.signals.agent import load_agent_scores
    from src.data.market import excess_returns
    from src.analysis.ic import ic_table, confirmed_mask
    from src.analysis.regression import run_conditional_regressions
    from src.analysis.pnl import pnl_table
    from src.analysis.compare import build_comparison_table

    tickers      = args.tickers or load_tickers()
    horizons     = getattr(args, "horizons",    None) or DEFAULT_HORIZONS
    ret_mode     = getattr(args, "returns",     "raw")
    with_agent   = getattr(args, "with_agent",  False)
    sig_filter   = getattr(args, "signals",     None)
    out_dir      = Path(getattr(args, "out",    None) or (RESULTS_DIR / "tables"))
    out_dir.mkdir(parents=True, exist_ok=True)

    return_types = (
        ("raw",)                if ret_mode == "raw"
        else ("excess",)        if ret_mode == "excess"
        else ("raw", "excess")
    )

    # Load + enrich panels
    panels: dict[str, pd.DataFrame] = {}
    pool_frames = []

    for ticker in tickers:
        try:
            panel = load_panel(ticker)
        except FileNotFoundError as e:
            print(f"  [analyze] SKIP {ticker}: {e}")
            continue

        panel = add_rule_signals(panel)

        if with_agent:
            agent_df = load_agent_scores(ticker)
            if len(agent_df) > 0 and "event_id" in agent_df.columns:
                agent_cols = [c for c in agent_df.columns if c != "event_id"]
                panel = panel.merge(agent_df[["event_id"] + agent_cols],
                                    on="event_id", how="left")

        pool_frames.append(panel)
        panels[ticker] = panel

    if not panels:
        print("  [analyze] No panels loaded — run 'build' stage first.")
        return

    pool = pd.concat(pool_frames, ignore_index=True)

    # Add excess returns (EW proxy if no SPY)
    if "excess" in return_types or ret_mode == "both":
        pool = excess_returns(pool, pool=pool, horizons=horizons)
        for ticker in list(panels.keys()):
            panels[ticker] = excess_returns(panels[ticker], pool=pool, horizons=horizons)

    # Signals to evaluate
    all_sigs  = (["agent_signal"] if with_agent else []) + RULE_SIGNALS
    sigs      = [s for s in all_sigs if (sig_filter is None or s in sig_filter)]
    tgt_raw   = [f"ret_{h}m" for h in horizons]
    tgt_exc   = [f"ret_{h}m_excess" for h in horizons]
    tgts      = tgt_raw + (tgt_exc if "excess" in return_types else [])

    # 1. IC table
    print("\n[analyze] computing IC table ...")
    ic_df = ic_table(pool, signals=sigs, targets=tgts)
    ic_df.to_csv(out_dir / "ic_table.csv", index=False)
    print(ic_df.to_string(index=False))

    # 2. IC for confirmed events subset
    conf = confirmed_mask(pool)
    if conf.sum() >= 10:
        ic_conf = ic_table(pool[conf], signals=sigs, targets=tgts)
        ic_conf.to_csv(out_dir / "ic_confirmed.csv", index=False)
        print(f"\n[analyze] confirmed events IC ({conf.sum()} rows):")
        print(ic_conf.to_string(index=False))

    # 3. Conditional OFI regression (pooled, llm_score)
    print("\n[analyze] running conditional OFI regressions ...")
    if "llm_score" in pool.columns:
        reg_df = run_conditional_regressions(pool, horizons=horizons)
        reg_df.to_csv(out_dir / "regression_table.csv", index=False)
        print(reg_df[["horizon", "n", "b_pos", "b_neg", "wald_p"]].to_string(index=False))

    # 4. PnL table
    print("\n[analyze] computing PnL table ...")
    pnl_df = pnl_table(pool, signals=sigs, targets=tgts)
    pnl_df.to_csv(out_dir / "pnl_table.csv", index=False)
    print(pnl_df.to_string(index=False))

    # 5. Comparison table (per-ticker)
    print("\n[analyze] building comparison table ...")
    cmp_df = build_comparison_table(
        panels, horizons=horizons, return_types=return_types, signals=sigs
    )
    cmp_df.to_csv(out_dir / "comparison_table.csv", index=False)
    print(cmp_df.to_string(index=False))

    print(f"\n[analyze] tables saved to {out_dir}")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "python src/run.py",
        description = "LNA pipeline: build | agent | analyze | all",
    )
    sub = p.add_subparsers(dest="stage", required=True)

    # shared args factory
    def _add_common(sp):
        sp.add_argument("--tickers",   nargs="+", metavar="TICKER",
                        help="Tickers to process (default: all from tickers.txt)")

    def _add_build_args(sp):
        sp.add_argument("--no-llm", action="store_true",
                        help="Skip LLM sentiment scoring (LM lexicon only)")
        sp.add_argument("--sentiment-only", action="store_true",
                        help="Score headlines only; skip panel assembly (use when OFI not yet available)")
        sp.add_argument("--force", action="store_true",
                        help="Re-score and re-assemble even if cached files exist")

    def _add_agent_args(sp):
        sp.add_argument("--limit",   type=int, metavar="N",
                        help="Score at most N new events per ticker (resume skips are free)")
        sp.add_argument("--dry-run", action="store_true",
                        help="Print what would be scored; make no API calls")

    def _add_analysis_args(sp):
        sp.add_argument("--horizons",   nargs="+", type=int, metavar="H",
                        default=HORIZONS, help="Return horizons in minutes")
        sp.add_argument("--returns",    choices=["raw", "excess", "both"], default="raw",
                        help="Return type for IC / PnL evaluation")
        sp.add_argument("--signals",    nargs="+", metavar="SIG",
                        help="Signals to evaluate (default: all rule signals + agent if --with-agent)")
        sp.add_argument("--with-agent", action="store_true",
                        help="Include agent_signal in analysis (requires agent stage)")
        sp.add_argument("--out",        metavar="PATH",
                        help="Output directory for tables (default: results/tables)")

    # build
    sp_build = sub.add_parser("build", help="Score headlines, assemble per-event panels")
    _add_common(sp_build); _add_build_args(sp_build)

    # agent
    sp_agent = sub.add_parser("agent", help="Run LLM trading agent over assembled panels")
    _add_common(sp_agent); _add_agent_args(sp_agent)

    # analyze
    sp_analyze = sub.add_parser("analyze", help="IC, regression, PnL, comparison tables")
    _add_common(sp_analyze); _add_analysis_args(sp_analyze)

    # all — build → agent → analyze
    sp_all = sub.add_parser("all", help="build → agent → analyze")
    _add_common(sp_all)
    _add_build_args(sp_all)
    sp_all.add_argument("--limit",   type=int, metavar="N",
                        help="Agent: score at most N new events per ticker")
    sp_all.add_argument("--dry-run", action="store_true",
                        help="Agent dry run — no API calls")
    _add_analysis_args(sp_all)

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.stage == "build":
        stage_build(args)
    elif args.stage == "agent":
        stage_agent(args)
    elif args.stage == "analyze":
        stage_analyze(args)
    elif args.stage == "all":
        stage_build(args)
        stage_agent(args)
        stage_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
