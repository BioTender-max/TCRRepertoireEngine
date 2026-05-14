# TCRRepertoireEngine

Simulate and analyze T-cell receptor repertoires including VDJ recombination, CDR3 diversity, and clonal expansion dynamics.

## Features

- VDJ recombination simulation with realistic gene usage frequencies
- CDR3 length distribution and amino acid composition analysis
- Shannon entropy and D50 diversity metrics computation
- Clonal expansion modeling across 50 donors
- Immune diversity visualization and repertoire overlap analysis

## Results

50 donors × 10,000 clonotypes; Mean CDR3=14.0 aa; Shannon H=8.52; D50=~1 clone

## Usage

```bash
pip install numpy scipy matplotlib
python tcr_repertoire_engine.py
```

## Tags

`tcr-repertoire`, `vdj-recombination`, `clonotype`, `cdr3`, `immune-diversity`, `clonal-expansion`
