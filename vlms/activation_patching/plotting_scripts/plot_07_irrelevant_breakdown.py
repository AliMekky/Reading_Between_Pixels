#!/usr/bin/env python3
import argparse
from behavior_breakdown_plot import make_plot

p = argparse.ArgumentParser()
p.add_argument("--summary", required=True)
p.add_argument("--comparisons", required=True)
p.add_argument("--counts", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
make_plot("irrelevant_word", a.summary, a.comparisons, a.counts, a.output)
