# Files in this directory:

Main:
- `LIAR-New.jsonl` this is the LIAR-New dataset.

Retrieval:
- `LIAR-New_articles.jsonl` this file haas the text of the PolitiFact articles, used for Web Oracle and Web Answerless Oracle methods in the paper.
- `remove_politifact_article_verdict.py` this file contains the simple function used to remove the final verdict from the PolitiFact articles for the Web Answerless Oracle method.

Dataset Construction and Reproducibility:
- The raw html scraped from PolitiFact is available here: https://drive.google.com/drive/folders/1N_HiYo-JShJEk1fVktkIZVXYOS-D7ous?usp=sharing.
- `process_raw_html.py` processes above into LIAR-New minus annotations, and LIAR-New_articles
- `LIAR-New_raw.jsonl` this is a raw version of the LIAR-New dataset annotations, which includes the individual, original labels from the 3 annotators. Note that these individual labels are pre disagreement resolution, so the majority vote will not always match the final possibility_label. Besides reproducibility, this data might be used to look at finer-grained categories, e.g., separating "Possible" with all annotators unanimous from "Possible" with one annotator marking "Hard". However, we caution that such analysis could be noisy.
- `LIAR-New_oldversion1` this is the old labeling of the LIAR-New dataset, "V1" as described in the appendix of the paper. It should not be used directly as the labeling quality is higher in the main version.

We thank Yury Orlovskiy for contributions to the dataset labeling.
