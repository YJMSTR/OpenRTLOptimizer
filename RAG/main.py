import os
import argparse
import pathlib
import time
import tempfile
from typing import List, Dict, Any
from byaldi import RAGMultiModalModel

class MultimodalRetriever:
    """
    Retrieves relevant documents using the ColQwen2 model for document retrieval.
    Uses the byaldi library which provides a simple interface to the colpali-engine.
    Supports PDF and image files.
    """
    def __init__(self, model_path: str = "../models/colqwen2-v1.0", index_root: str = "../models/byaldi/", index_name: str = "rtl_optimize_rules"):
        self.model_path = model_path
        self.index_root = index_root
        self.index_name = index_name
        self.rag_model = None
        
        print(f"[INIT] Loading local model from: {model_path}")
        self.model_path = pathlib.Path(model_path).resolve()
        
        # Create a directory for the index if it doesn't exist
        pathlib.Path(index_root).mkdir(exist_ok=True, parents=True)
        print(f"[INIT] Index path: {pathlib.Path(index_root).resolve()}")
    
    def load_model(self):
        """Load the ColQwen2 model."""
        try:
            print("[MODEL] Starting model or index loading...")
            start_time = time.time()
            
            if os.path.exists(os.path.join(self.index_root, self.index_name)):
                print(f"[MODEL] Found existing index: {self.index_name}")
                print(f"[MODEL] Loading index, this may take a few minutes...")
                self.rag_model = RAGMultiModalModel.from_index(
                    self.index_name,
                    index_root=self.index_root
                )
                print(f"[MODEL] Index loading complete!")
            else:
                print(f"[MODEL] No existing index found, loading model: {self.model_path}")
                print(f"[MODEL] This may take a few minutes depending on model size and hardware...")
                self.rag_model = RAGMultiModalModel.from_pretrained(
                    self.model_path,
                    index_root=self.index_root
                )
                print(f"[MODEL] Model loading complete!")
                
            
            elapsed_time = time.time() - start_time
            print(f"[MODEL] Completed! Time elapsed: {elapsed_time:.2f} seconds")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
        
    
    def create_index(self, input_path: str, store_collection_with_index: bool = False, overwrite: bool = False):
        """
        Create an index for the documents in the input path.
        
        Args:
            input_path: Path to the directory containing documents or a single document file.
            store_collection_with_index: Whether to store the base64 encoded documents with the index.
            overwrite: Whether to overwrite an existing index.
        """
        if self.rag_model is None:
            print("[INDEX] Model not loaded, attempting to load model...")
            if not self.load_model():
                return False
        
        try:
            start_time = time.time()
            print(f"[INDEX] Creating index for path: {input_path}")
            print(f"[INDEX] Index name: {self.index_name}")
            
            # Force reload the model if overwrite is True to ensure clean state
            if overwrite and os.path.exists(os.path.join(self.index_root, self.index_name)):
                import shutil
                print(f"[INDEX] Completely removing existing index: {self.index_name}")
                shutil.rmtree(os.path.join(self.index_root, self.index_name))
                # Reload the model to ensure clean state
                print(f"[INDEX] Reloading model to ensure clean state...")
                self.rag_model = RAGMultiModalModel.from_pretrained(
                    self.model_path,
                    index_root=self.index_root
                )
            
            # Check if the input path is a directory and process markdown files first
            input_path_obj = pathlib.Path(input_path)
            if input_path_obj.is_dir():
                print(f"[INDEX] Processing documents and creating index, this may take minutes to hours...")
                print(f"[INDEX] Large PDFs and multiple documents will take longer...")
                
                self.rag_model.index(
                    input_path=input_path,
                    index_name=self.index_name,
                    store_collection_with_index=store_collection_with_index,
                    overwrite=overwrite
                )
            
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print(f"[INDEX] Index created successfully: {self.index_name}")
            print(f"[INDEX] Completed! Time elapsed: {minutes}m {seconds}s")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to create index: {e}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the indexed documents for relevant documents matching the query.
        
        Args:
            query: The search query.
            k: Number of results to return.
            
        Returns:
            A list of dictionaries containing the search results.
        """
        if self.rag_model is None:
            print("[SEARCH] Model not loaded, attempting to load model...")
            if not self.load_model():
                return []
        
        try:
            start_time = time.time()
            print(f"[SEARCH] Query: \"{query}\"")
            print(f"[SEARCH] Searching for relevant documents, returning top {k} results...")
            
            results = self.rag_model.search(query, k=k)
            
            elapsed_time = time.time() - start_time
            print(f"[SEARCH] Search completed! Time elapsed: {elapsed_time:.2f} seconds")
            print(f"[SEARCH] Found {len(results)} results")
            return results
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve documents using ColQwen2 model')
    parser.add_argument('--model_path', type=str, default="../models/colqwen2-v1.0",
                      help='Path to the ColQwen2 model or model ID (default: ../models/colqwen2-v1.0)')
    parser.add_argument('--index_root', type=str, default="../models/byaldi/",
                      help='Path to store the index (default: ../models/byaldi/)')
    parser.add_argument('--index_name', type=str, default="rtl_optimize_rules",
                      help='Name of the index (default: rtl_optimize_rules)')
    parser.add_argument('--input_path', type=str, default="../data",
                      help='Path to the documents to index')
    parser.add_argument('--query', type=str, default="Give me the optimize rules for rtl performance optimization",
                      help='Query to search for (default: "Give me the optimize rules for rtl performance optimization")')
    parser.add_argument('--k', type=int, default=3,
                      help='Number of results to return (default: 3)')
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = MultimodalRetriever(
        model_path=args.model_path,
        index_root=args.index_root,
        index_name=args.index_name
    )
    
    # Load model
    print("\n" + "="*50)
    print("Loading model, please wait...")
    print("="*50)
    if not retriever.load_model():
        print("[ERROR] Model loading failed, exiting.")
        exit(1)
    
    # Create index if input path is provided
    if args.input_path:
        print("\n" + "="*50)
        print(f"Creating index for documents: {args.input_path}")
        print("This may take some time, please be patient...")
        print("="*50)
        if not retriever.create_index(args.input_path, overwrite=True):
            print("[ERROR] Index creation failed, exiting.")
            exit(1)
    
    # Perform search
    print("\n" + "="*50)
    print(f"Executing search query: \"{args.query}\"")
    print("="*50)
    results = retriever.search(args.query, k=args.k)
    
    # Print results
    print("\n" + "="*50)
    print(f"Search query: \"{args.query}\"")
    print(f"Found {len(results)} results:")
    print("="*50)
    for i, result in enumerate(results):
        print("-" * 40)
        print(f"Result {i+1}:")
        print(f"Relevance score: {result['score']:.4f}")
        print(f"Document ID: {result['doc_id']}")
        try:
            page_num = result['page_num']
            print(f"Page: {page_num}")
        except (KeyError, TypeError, AttributeError):
            pass
        try:
            metadata = result['metadata']
            if metadata:
                print(f"Metadata: {metadata}")
        except (KeyError, TypeError, AttributeError):
            pass
        try:
            if result['base64']:
                print("Contains base64 encoded document content")
        except (KeyError, TypeError, AttributeError):
            pass
    print("-" * 40)
