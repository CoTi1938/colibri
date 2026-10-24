"""DeepSeek V4.1's prompt tail (v41_prefill_tail): past the first layer with compress_ratio
1 the layers read the long context from that layer's caches and keep only a window of their
own, so they run on a prompt's last D * (W - 1) + 1 rows and give the same bits.

The tiny fixture (tools/make_dsv41_tiny.py) has that layout: six layers, compress ratios
0 2 2 1 1 0, layer 3 the last cache owner and the candidate source, a window of 8, so the
exact tail is 2 * 7 + 1 = 15 rows. A prompt of 200 rows takes it. V41_TRACE=1 prints the
logits of every forward (their sum and first values); the run with the tail must print the
lines of the run without it (V41_PREFILL_TAIL=0), the prompt's forward and every decode step
after it, and a tail shorter than the exact one (DeepSeek's kind, V41_PREFILL_TAIL=4 here)
must not, which says the tail ran.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("deepseek_v41.exe" if sys.platform == "win32" else "deepseek_v41")
FIXTURE = Path(os.environ.get("DSV41_TINY", HERE / "dsv41_tiny"))
PROMPT = bytes((7 * i + 3) % 200 + 32 for i in range(200))   # one id per byte, no runs


def run(tail, dspark="0"):
    env = {"SNAP": str(FIXTURE), "SERVE": "1", "CTX": "256", "OMP_NUM_THREADS": "2",
           "V41_DSPARK": dspark, "V41_TRACE": "1", "COLI_VULKAN": "0"}
    if tail is not None:
        env["V41_PREFILL_TAIL"] = tail
    engine = ServeEngine(ENGINE, ["8"], env, reuse=False)
    try:
        text = engine.ask("t", PROMPT, 24)
    finally:
        err = engine.close()
        engine.p.stdout.close(); engine.p.stderr.close()
    logits = [l for l in err.splitlines() if l.startswith("T logits")]
    return text, logits, err


@unittest.skipUnless(ENGINE.exists() and (FIXTURE / "model.safetensors").exists(),
                     "built V4.1 engine and tiny fixture required")
class PrefillTailTest(unittest.TestCase):
    def test_the_exact_tail_gives_every_row_s_bits(self):
        text_all, logits_all, err_all = run("0")
        text, logits, err = run(None)
        self.assertNotIn("[v41] prefill: the layers after", err_all)
        self.assertIn("[v41] prefill: the layers after layer 3 run on a prompt's last 15 rows", err)
        self.assertGreater(len(logits), 20)            # the prompt's forward and the decode steps
        self.assertEqual(logits, logits_all)
        self.assertEqual(text, text_all)

    def test_a_shorter_tail_is_approximate(self):
        _, logits_all, _ = run("0")
        _, logits, err = run("4")
        self.assertIn("last 4 rows (V41_PREFILL_TAIL", err)
        self.assertNotEqual(logits, logits_all)

    def test_dspark_seeded_from_the_tail_drafts_the_same(self):
        # the stages' window is within the tail: the same drafts, so the same forwards
        text_all, logits_all, _ = run("0", dspark="1")
        text, logits, _ = run(None, dspark="1")
        self.assertEqual(logits, logits_all)
        self.assertEqual(text, text_all)


if __name__ == "__main__":
    unittest.main()
