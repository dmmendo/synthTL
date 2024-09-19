## Overview

SynthTL uses large language models (LLMs), model checkers, and oracle (human) guidance, to translate an natural language (NL) specification into a temporal logic (TL) specification that reflects the NL and holds (formally) on the design.

SynthTL performs *structured translation*, whereby it first decomposes a complex unstructured NL specification into a logical combination of simple NL sub-specifications. Then, it produces TL translations of the simple NL sub-specifications, called *sub-translations*, and mechanically combines these sub-translations to yield a TL translation of the complex NL specification.

Please see our paper for more explanation of SynthTL's design:

Daniel Mendoza, Chistopher Hahn, Caroline Trippel. [Translating Natural Language to Temporal Logics with Large Language Models and Model Checkers](), FMCAD 2024.

## Quickstart

Please see the test_TranslationGeneration.ipynb jupyter notebook for a short tutorial on using SynthTL.

## Installation

requirements.txt contains list of python modules which can be installed via pip:
```
pip install -r requirements.txt
```

In addition, SynthTL uses Spot's Python API for LTL model checking. It can be installed using this [installation guide](https://spot.lre.epita.fr/install.html) or using conda as follows:
```
conda install -c conda-forge spot
``` 

## Other Notebooks to test SynthTL functionality
test_TranslationGeneration.ipynb - short demonstration of using SynthTL to generate an LTL specification.

test_BatchMC.ipynb - runs experiments with SynthTL's batch model checking

test_CulpritIdentification.ipynb - runs experiments with SynthTL's culprit identification

test_TranslationSearch.ipynb - runs experiments with SynthTL's translation search

## Example Results
ex_translation_prompt.txt - example LLM prompt for TL generation

ex_decomposition_prompt.txt - example LLM prompt for decomposition

ex_amba_controller_tree.txt - example output SynthTL sub-translation tree with GPT3.5

ex_amba_worker_tree.txt - example output SynthTL sub-translation tree with GPT3.5

ex_amba_arbiter_tree.txt - example output SynthTL sub-translation tree with GPT3.5

## Code Structure
context_retriever.py contains code for retrieval augmented generation (RAG).

spot_utils.py defines functions for querying the Spot LTL model checker.

synthTL.py defines SynthTL sub-translation tree classes and algorithms.

utils.py defines functions to query LLMs.

specs/* contains natural language specifications for the AMBA AHB arbiter, controller, and worker.