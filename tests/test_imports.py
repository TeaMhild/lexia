def test_imports():
    """Vérifie que tous les imports critiques fonctionnent."""
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_groq import ChatGroq
    import chromadb
    assert True


    