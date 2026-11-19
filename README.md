# RAG from Scratch

## About the PDF

This project uses Chapter II, “Economic Review,” from the Reserve Bank of India Annual Report 2025–26. The included economic_review.pdf covers 46 printed pages, from page 13 to page 58, and discusses the economy, inflation, money and credit, financial markets, government finances and the external sector.

[Read the chapter on the RBI website](https://www.rbi.org.in/scripts/AnnualReportPublications.aspx?Id=1462).

## How I built the project

I started by extracting the chapter with pypdf and keeping its printed page numbers with the text. This let me keep a source reference for each passage. The app reads economic_review.pdf directly when it runs.

Next, I split each page into chunks and tried two sizes, 64 and 192 MiniLM tokens, with 16 tokens of overlap. I used all-MiniLM-L6-v2 to embed the chunks and the questions, then normalised the vectors.

For retrieval, I used NumPy to calculate cosine similarity and select the five highest-ranked passages. The retrieval logic is written directly in Python without LangChain or LlamaIndex.

I connected the retrieved passages to Groq using openai/gpt-oss-120b. The prompt asks the model to answer from the supplied excerpts, cite the printed pages and say when the evidence is missing. The API key is read from a local .env file.

To run it, install the packages with `python -m pip install -r requirements.txt`. Create a local .env file with `GROQ_API_KEY=your_key` and `GROQ_MODEL=openai/gpt-oss-120b`, each on its own line. Then run `python rag.py "What did banks issue when their funding fell short?"`. Add `--retrieve-only` to see the passages without an API call, or `--size 192` to use longer chunks. The first run downloads the embedding model unless it is already cached.

Finally, I prepared 15 questions with expected source passages, compared retrieval for both chunk sizes and checked the live Groq answers against those passages.

## Results

Both chunk sizes achieved recall@5 of 9/15, or 60.0%.  
The 64-token chunks achieved recall@1 of 8/15 (53.3%), compared with 7/15 (46.7%) for 192 tokens.  
I kept 64 tokens as the default because it placed the complete passage first one more time and sends less text to the answer model.  
The live Groq check produced 11 correct answers out of 15 (73.3%), two incorrect answers and two abstentions.  
Short chunks sometimes split useful evidence, while longer chunks sometimes ranked related material above the required passage.
