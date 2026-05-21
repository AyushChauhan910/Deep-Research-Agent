"""Run the eval dataset through the agent and dump a results JSON + summary."""
import json
import os
import sys
import time
from pathlib import Path

# allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.agent import DeepResearchAgent
from src.llm import LLM
from src.memory import create_session, init_db
from src.search import TavilySearch

from eval.metrics import (citation_density, expresses_uncertainty, flags_conflict,
                          keyword_coverage, llm_judge, n_unique_citations,
                          source_diversity)


def run(dataset_path: str = "eval/dataset.json"):
    init_db()
    dataset = json.load(open(dataset_path))

    llm = LLM()
    judge = LLM(model=os.environ.get("JUDGE_MODEL", os.environ.get("LLM_MODEL")))
    search = TavilySearch()
    agent = DeepResearchAgent(llm, search)

    results = []
    for item in dataset:
        print(f"\n[{item['id']}]  {item['question'][:90]}")
        sid = create_session(title=f"eval-{item['id']}")
        try:
            out = agent.run(item["question"], sid)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append({**item, "error": str(e)})
            continue

        ans, cits = out["answer"], out["citations"]

        m = {
            "citation_density": citation_density(ans),
            "unique_citations": n_unique_citations(ans),
            "source_diversity": source_diversity(cits),
            "n_pages_fetched": out["n_pages_fetched"],
            "n_sources_cited": out["n_sources"],
            "elapsed_s": out["elapsed"],
            "expresses_uncertainty": expresses_uncertainty(ans),
            "flags_conflict": flags_conflict(ans),
            "self_critique": out["critique"],
        }
        if "expected_keywords" in item:
            m["keyword_coverage"] = keyword_coverage(ans, item["expected_keywords"])
        if item.get("should_express_uncertainty"):
            m["uncertainty_correct"] = m["expresses_uncertainty"]
        if item.get("type") == "potentially_conflicting":
            m["conflict_flagged"] = m["flags_conflict"]

        try:
            judged = llm_judge(judge, item["question"], ans, cits)
            m.update({f"judge_{k}": v for k, v in judged.items()})
        except Exception as e:
            print(f"  judge error: {e}")

        print(f"  ✓ {out['elapsed']:.1f}s · {len(cits)} sources · "
              f"cite_density={m['citation_density']}")

        results.append({**item, "answer": ans, "citations": cits, "metrics": m})

    # aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE")
    print("=" * 70)

    num_keys = ["citation_density", "unique_citations", "source_diversity",
                "elapsed_s", "keyword_coverage",
                "judge_grounded", "judge_relevant", "judge_well_cited",
                "judge_clear", "judge_uncertainty_calibration"]
    for k in num_keys:
        vals = [r["metrics"].get(k) for r in results
                if "metrics" in r and isinstance(r["metrics"].get(k), (int, float))]
        if vals:
            print(f"  {k:34s}  mean={sum(vals)/len(vals):.2f}  n={len(vals)}")

    # special-case checks
    unc = [r for r in results if r.get("should_express_uncertainty")]
    if unc:
        correct = sum(1 for r in unc if r["metrics"].get("uncertainty_correct"))
        print(f"  uncertainty_correct                 {correct}/{len(unc)}")
    conf = [r for r in results if r.get("type") == "potentially_conflicting"]
    if conf:
        flagged = sum(1 for r in conf if r["metrics"].get("conflict_flagged"))
        print(f"  conflict_flagged                    {flagged}/{len(conf)}")

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"results_{int(time.time())}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\n📁 {out_file}")


if __name__ == "__main__":
    run()