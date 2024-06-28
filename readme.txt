This README serves as documentation for the SynthTL implementation.

Examples:
ex_translation_prompt.txt - example LLM prompt for TL generation
ex_decomposition_prompt.txt - example LLM prompt for decomposition
ex_amba_controller_tree.txt - example output synthTL sub-translation tree with GPT3.5
ex_amba_worker_tree.txt - example output synthTL sub-translation tree with GPT3.5
ex_amba_arbiter_tree.txt - example output synthTL sub-translation tree with GPT3.5

Code Structure:
Main.ipynb contains scripts for running experiments with SynthTL.
nl2spec_eval.ipynb contains scripts for running experiments with nl2spec.
context_retriever.py contains code for retrieval augmented generation (RAG).
spot_utils.py defines functions for querying the Spot LTL model checker.
synthTL.py defines SynthTL sub-translation tree classes and algorithms.
utils.py defines functions to query LLMs.
specs/* contains natural language specifications for the AMBA AHB arbiter, controller, and worker.