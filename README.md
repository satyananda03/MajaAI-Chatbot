### Project Structure
```
app/
├── api/ # definisi API endpoint
├── core/ # manajemen konfigurasi env (pydantic settings) dan logging
├── modules/ # main Chatbot feature 
│ ├── graph/ # logic workflow Langgraph
│ ├── ingestion/ # logic fitur ingestion data untuk RAG
│ ├── lapor/ # logic fitur Lapor
│ ├── rag/ # logic fitur RAG
│ ├── ticket/ # logic fitur Ticket
├── prompts/ # tempat prompt per version
├── schemas/ # manajemen tipe data Request-Response dari API endpoint (pydantic schemas)
├── services/ # fungsi" untuk consume LLM, embedding, & prompt loader
├── utils/ # kumpulan helper function
data/ # tempat simpan chroma vector index, bm25 index, & stopwords
```

- API documentation (Swagger UI) : localhost:8000/docs

### Running API
```
cd majaai-chatbot
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### VS Code Extension
- One Dark Pro
- vscode-icons
- indent-rainbow
- Bracket pair color 
- Windsurf Plugin (formerly codeium)
- Python (Microsoft)
- Pylance (Microsoft)

### Consume /chat & /chat_rag endpoint (streaming mode) :
```
resp = requests.post("http://localhost:8000/chat", json={"question": "apa itu KIA?"}, stream=True)
for chunk in resp.iter_content(chunk_size=None): 
    if chunk:
        print(chunk.decode("utf-8"), end="", flush=True)
```