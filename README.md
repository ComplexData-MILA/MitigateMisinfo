# Towards Reliable Misinformation Mitigation: Generalization, Uncertainty, and GPT-4

Kellin Pelrine, Anne Imouza, Camille Thibault, Meilina Reksoprodjo, Caleb A. Gupta, Joel N. Christoph, Jean-François Godbout, Reihaneh Rabbany

Paper: https://arxiv.org/abs/2305.14928

LIAR-New dataset: available in LIAR-New directory

Running experiments:  
We recommend https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py to accelerate OpenAI API requests. Prompts are available in the paper. 
For RoBERTa-L, please see https://github.com/ComplexData-MILA/misinfo-baselines and corresponding paper https://arxiv.org/pdf/2104.06952.pdf.  
For error analysis in section 4.5, please see error-analysis.py and corresponding data here: https://drive.google.com/drive/folders/1Ur-JcWBfQ4HfB0Fr8Sm-TnRqpqATrsXQ?usp=drive_link.  
For other SLMs, please see the notebooks in the other-SLM-code directory.  

Output files (e.g., GPT-4 output) can be found at https://drive.google.com/drive/folders/1me5GcttWOs2J-Z10gpIg2AkobKMa21EA?usp=sharing  
Most of the output files are paired. "rawoutput" version is output from a script similar to above, while the version without that affix is slightly more processed. We recommend the latter if you just want to work with the main output, but include the former in case it provides any extra useful information.  
Please note that the formatting varies slightly between some of the files.
