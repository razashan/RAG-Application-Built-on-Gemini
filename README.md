# RAG Application Built on Gemini Model

This project demonstrates a **Retrieval-Augmented Generation (RAG)** application built with the **Gemini model** for question-answering tasks. It uses **Streamlit** for the user interface and the **LangChain** library to integrate various tools like **Google Generative AI**, **Chroma vector store**, and **PyPDFLoader** for document processing.

## Features

- **Upload and process PDFs**: Users can upload PDF files (e.g., research papers) which are processed to extract and split text into chunks.
- **Context-aware Q&A**: Users can ask questions, and the system retrieves relevant sections from the document to generate answers using **Gemini** and **Google Generative AI** embeddings.
- **Built with LangChain**: Uses LangChain components like **Chroma**, **Google Generative AI**, and **Retrieval Chains** to build the RAG model.
  
## Technologies Used

- **Streamlit**: For the web interface
- **LangChain**: For chaining the components of the retrieval-augmented generation process
- **Chroma**: For storing document embeddings
- **Google Generative AI**: For embedding and generating answers
- **Gemini Model**: For generating responses based on retrieved document context
- **PyPDFLoader**: To extract text from PDF files


