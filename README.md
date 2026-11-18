# Document Q&A Assistant with RAG & Vector Embeddings

A single-page Streamlit project for answering questions using uploaded
documents. Document loading, recursive text chunking, Gemini embedding
integration, persistent ChromaDB vector storage, the RAG question-answering
engine, and metadata-derived source citations are implemented. The Streamlit
upload/index/question workflow is implemented.

## Architecture and workflow

- **Document loading:** Load PDF pages and UTF-8 plain-text files, retaining
  source information and PDF page metadata.
- **Recursive chunking:** Split long document text into smaller, overlapping
  sections while preserving the original metadata.
- **Embeddings:** Convert each text chunk into a vector representation using
  Gemini Embedding 2.
- **Vector storage:** Store vectors and original text/metadata in a persistent
  local ChromaDB collection.
- **Semantic retrieval:** Find document chunks whose meaning is relevant to a
  user's question using the configured `RETRIEVER_TOP_K`.
- **Grounded answer generation:** Provide retrieved passages as untrusted
  evidence to a Gemini chat model and return a text answer.
- **PDF page citations:** Generate independent citations from retrieved
  document metadata, converting zero-based page indexes to page numbers.
- **Session isolation:** Keep each browser session's uploads and Chroma
  collection separate.

The technology stack is Python, Streamlit, LangChain Core, LangChain text
splitters, Google's Gemini integration, ChromaDB, and PyPDF.

## Document loading

`document_loader.py` supports `.pdf` and `.txt` files. Text files are decoded as
UTF-8, with a UTF-8 byte-order mark handled when present. Unsupported file
types, missing files, unreadable PDFs, encrypted PDFs, and files with no usable
text produce explicit loading errors.

PDF pages are extracted individually with PyPDF. Each page becomes its own
LangChain `Document`; metadata includes the original filename (`source`), its
zero-based `page_index`, the PDF's `total_pages`, and `page_label` when
available. Keeping pages separate preserves the location of source text as it
moves through chunking, retrieval, and future citation formatting. Page indexes
remain zero-based in metadata; citation formatting should add one when showing
a conventional human-readable page number.

PDF extraction only reads an existing text layer. Scanned/image-only PDFs
without extractable text are unsupported in this stage. They need a future OCR
workflow; this project intentionally does not install or run OCR software.

## Recursive text chunking

`text_splitter.py` uses LangChain's `RecursiveCharacterTextSplitter`, preferring
paragraph breaks, then line breaks, spaces, and finally character boundaries.
Chunks keep the original document metadata and receive a per-document
`chunk_index` and a `start_index` where the splitter can determine the text
offset.

The default `CHUNK_SIZE` is **1000 characters** and `CHUNK_OVERLAP` is **150
characters**, read from `config.py` and environment configuration. A chunk size
limits the target length of each chunk. Overlap repeats a small trailing part
of one chunk at the beginning of the next, helping avoid loss of context when a
sentence or related passage crosses a chunk boundary. Overlap must be
non-negative and smaller than chunk size.

## Initial setup

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Configure `GEMINI_API_KEY`, `GEMINI_CHAT_MODEL`, `GEMINI_EMBEDDING_MODEL`, and
`GEMINI_EMBEDDING_DIMENSIONS` in the existing project `.env` file. Use
`.env.example` as a reference; do not overwrite an existing `.env`. Document
loading, chunking, and offline tests do not require a Gemini API key.

From the project directory, start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In the browser:

1. Select one or more PDF or TXT files in **Add your documents**.
2. Select **Index documents**. Loading, chunking, embedding, and storage begin
   only after this explicit action. Per-file failures are reported without
   exposing local file paths or provider exceptions.
3. After chunks are indexed, ask a question in the chat input.
4. Review the answer and expand its separate **Sources** entries for filenames,
   PDF page numbers, page labels, and chunk indexes.
5. Use the sidebar to clear chat, clear this session's indexed collection, or
   reset the session.

Uploads are stored under a generated per-session temporary directory using
generated disk filenames; citation metadata retains the safe original basename.
Each Streamlit session receives a separate Chroma collection. Repeated
filenames within one upload receive distinct internal document IDs and retain
separate content. Re-indexing an updated file replaces chunks associated with
that stable file identity.

The local Nunito font files are not present in `assets/fonts`; the interface
uses a system-font fallback and does not load external fonts.

### Optional live Gemini embedding smoke test

After setting `GEMINI_API_KEY` in `.env`, run this command manually:

```powershell
.\.venv\Scripts\python.exe .\scripts\gemini_embedding_smoke_test.py
```

It embeds a synthetic document and question, then reports vector dimensions
and whether the values are finite. The script does not display credentials and
is not run by pytest. It makes live Gemini API requests only when you execute
it explicitly.

### Optional live Gemini chat smoke test

After configuring `GEMINI_API_KEY`, run this command manually:

```powershell
.\.venv\Scripts\python.exe .\scripts\gemini_chat_smoke_test.py
```

The script sends one harmless prompt to the configured chat model, prints the
text from either a string response or LangChain text blocks, and maps
authentication, quota, network, and model errors to safe messages. Empty
responses receive a sanitized diagnostic. Pytest does not execute this script.

## Configuration

`config.py` reads settings from the project's `.env` file or process
environment. Defaults are provided for local development; invalid chunk sizes,
overlap values, or retrieval counts raise configuration errors. API credentials
are not included in the source tree.

## Known limitations

- PDF support is limited to extractable text; scanned PDFs require future OCR.
- Gemini embeddings and chat require a valid API key, enabled API access,
  network connectivity, and available quota. Automated tests use fakes and do
  not make Gemini API calls.
- Local ChromaDB persistence uses files under `data/chroma_db` by default.
  Render Free's filesystem is ephemeral, so local vectors may disappear on
  redeploy or instance replacement. A durable hosted database or persistent
  disk is needed for production persistence.

## Gemini embeddings

An embedding maps text to a numeric vector whose dimensions encode semantic
features. Similar text can have vectors that are close in the embedding space,
allowing retrieval to find relevant chunks without relying only on exact word
matches.

The configured provider is **Gemini Embedding 2**. Documents are prepared as
`title: {document_title} | text: {chunk_text}`, using `title: none` when a title
is unavailable. Search questions use
`task: question answering | query: {user_question}`. These task-specific
prefixes tell the model how the text will be used; the old `task_type`
parameter is not used. Prepared strings are sent only for embedding. Chroma
stores the original chunk text and its metadata.

The default output dimensionality is **768**, configurable through
`GEMINI_EMBEDDING_DIMENSIONS`. Dimensions are the number of numeric components
in each vector. 768 is the requested compact configuration: it reduces storage
and vector-search work compared with wider embeddings while keeping the
embedding model's supported output dimensionality configurable.

## ChromaDB vector storage

`vector_store.py` stores document vectors, original text, and scalar metadata
in a named Chroma collection. Similarity search embeds the question with the
query prefix and asks Chroma for the configured number of nearest chunks.
Collection names can be supplied explicitly. The Streamlit interface creates a
stable unique collection name per browser session so separate sessions' stored
documents do not share a collection.

Chunk IDs are deterministic SHA-256 hashes of source/page/chunk location
metadata and the original chunk content. Re-indexing identical chunks updates
their existing IDs rather than creating duplicate vectors. When a batch
re-indexes a source, its stale chunks are removed only after new chunks are
stored; other sources in that collection are retained.

PDF `source`, zero-based `page_index`, optional `page_label`, and chunk location
metadata are retained in Chroma. Citation formatting can use the page index
and convert it to a human-readable page number later.

Chroma persists locally under `data/chroma_db` by default and automatically
saves changes; no manual `persist()` call is required. Render Free does not
guarantee durable local files between deployments or instance restarts, so
local persistence there is not a durable production storage strategy.

## RAG question answering

Retrieval-augmented generation (RAG) first retrieves document chunks relevant
to a question and then supplies those chunks to a language model as evidence.
Semantic retrieval is useful when relevant passages use words different from
the question; it compares embeddings rather than requiring exact phrase
matches. The user's question is embedded using the Gemini Embedding 2
question-answering query prefix described above.

`answer_question()` checks that the question is non-empty, checks that the
selected collection contains documents, and retrieves up to `RETRIEVER_TOP_K`
chunks (default 4). A caller can override `top_k` or supply a session-specific
collection name. If the collection is empty, the function returns a clear
message without initializing or calling Gemini. If retrieval is empty, it
returns: "I couldn't find relevant information in the uploaded documents."

Retrieved context is passed through a LangChain Expression Language (LCEL)
prompt/model/output-parser chain to `ChatGoogleGenerativeAI`, configured with
`GEMINI_CHAT_MODEL` and `GEMINI_API_KEY`. Temperature is set to `0.0` only for
models that support custom sampling; fixed-sampling models omit that parameter.
The system prompt requires answers to use only retrieved evidence. If evidence
does not support an answer, the exact response is "I couldn't find relevant
information in the uploaded documents." with no sources displayed. Supported
answers cite only assigned source labels. Document text is explicitly marked
as untrusted data; instructions contained in uploaded files are not to be followed.
The configured Gemini key and recognizable secret/configuration assignments in
retrieved text are redacted before context construction. This reduces
prompt-injection risk but cannot guarantee that a model will never make an
unsupported statement.

Source citations are generated separately from the model's answer using the
retrieved Python `Document` metadata. Each unique source page receives a stable
label such as `[S1]`; repeated chunks from that page share the citation. A
zero-based PDF `page_index` becomes a human-readable page number by adding one.
Text-file citations include the filename and omit page numbers. Chunk indexes
are included when available. Model-generated labels that were not assigned to
retrieved sources are removed, and the original metadata is not modified.

Chat initialization fails safely if `GEMINI_API_KEY` is missing. Request
failures are translated to safe messages for quota/rate limits, unavailable
service/network timeouts, and authentication failures; exception contents and
credentials are not returned to callers. Automatic retries are disabled for
the chat model.

The interface passes questions to the answer engine with the session-specific
collection and configured `RETRIEVER_TOP_K`. Retrieval relevance and model
responses depend on uploaded content and Gemini service; citations verify
source metadata, not the truth of a generated claim.
