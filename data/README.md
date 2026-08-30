# Datasets

The datasets used in the paper are **not yet included in this release**; they will be
added here. This directory documents the layout the code expects.

```
data/
├── gsm_symbolic/                 # GSM-Symbolic + constructed variants
│   ├── test_gsm_symbolic.csv     #   clean narrated problems
│   ├── test_gsm_p1.csv           #   +1 required operation
│   ├── test_gsm_noop.csv         #   NoOp distractor clause inserted
│   ├── test_gsm_filler.csv       #   length/digit-matched filler control
│   ├── test_gsm_filler_df.csv    #   digit-free filler control
│   └── test_padded_symbolic.csv  #   length-matched to GSM-P1
├── svamp/                        # SVAMP + operand-resampled variants
└── phantomwiki/                  # generated universes + multi-hop questions
```

All GSM-family datasets derive from the public GSM8K and GSM-Symbolic releases; the
construction pipelines are in `data_scripts/` and are specified in the paper's appendix.
SVAMP and PhantomWiki datasets are built by the scripts in `data_scripts/svamp/` and
`data_scripts/phantomwiki/` from their respective public sources.
