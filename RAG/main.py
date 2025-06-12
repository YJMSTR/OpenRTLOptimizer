import base64
from io import BytesIO
import os
import argparse
import pathlib
import time
from tqdm import tqdm
from pdf2image import convert_from_path
from typing import List, Dict, Any
from byaldi import RAGMultiModalModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from rerankers import Reranker 
from PIL import Image

class MultimodalRAG:
    """
    Retrieves relevant documents using the ColQwen2 model for document retrieval.
    Uses the byaldi library which provides a simple interface to the colpali-engine.
    Supports PDF and image files.
    """
    def __init__(self, retriever_model_path: str = "../models/colqwen2-v1.0", index_root: str = "../models/byaldi/", index_name: str = "rtl_optimize_rules", reranker_model_path: str = "../models/MonoQwen2-VL-v0.1", vl_model_path: str = "../models/BAGEL-&B-MoT"):
        self.retriever_model_path = retriever_model_path
        self.index_root = index_root
        self.image_path = self.index_root + "image/"
        self.index_name = index_name
        self.reranker_model_path = reranker_model_path
        self.vl_model_path = vl_model_path
        self.retriever_model = None
        self.reranker_model = None
        self.vl_model = None

        
        print(f"[INIT] Loading local retriever model from: {retriever_model_path}")
        self.retriever_model_path = pathlib.Path(retriever_model_path).resolve()
        
        # Create a directory for the index if it doesn't exist
        pathlib.Path(index_root).mkdir(exist_ok=True, parents=True)
        print(f"[INIT] Index path: {pathlib.Path(index_root).resolve()}")

    def pdf_to_png(self, pdf_path: str):
        pdf_files = [f for f in os.listdir(pdf_path) if f.lower().endswith(".pdf")]
        data = []
        
        # Create image directory if it doesn't exist
        image_dir = os.path.join(pdf_path, "image")
        os.makedirs(image_dir, exist_ok=True)
        
        for pdf_file in tqdm(pdf_files, desc="Converting PDFs to images"):
            pdf_file_path = os.path.join(pdf_path, pdf_file)
            try:
                # Get PDF base name without extension
                pdf_base_name = os.path.splitext(pdf_file)[0]
                
                # Convert PDF to images
                pages = convert_from_path(pdf_file_path, dpi=300)
                
                # Save each page as PNG
                for i, page in enumerate(pages):
                    image_path = os.path.join(image_dir, f"{pdf_base_name}_page_{i+1}.png")
                    page.save(image_path, "PNG")
                    data.append({
                        "pdf_file": pdf_file,
                        "page_number": i+1,
                        "image_path": image_path
                    })
                
                print(f"Converted {pdf_file} - {len(pages)} pages")
            except Exception as e:
                print(f"{pdf_file} convert error: {e}")
                continue
                
        return data

    def load_model(self):
        """Load the ColQwen2 model."""
        try:
            print("[MODEL] Starting model or index loading...")
            start_time = time.time()
            
            if os.path.exists(os.path.join(self.index_root, self.index_name)):
                print(f"[MODEL] Found existing index: {self.index_name}")
                print(f"[MODEL] Loading index, this may take a few minutes...")
                self.retriever_model = RAGMultiModalModel.from_index(
                    self.index_name,
                    index_root=self.index_root
                )
                print(f"[MODEL] Index loading complete!")
            else:
                print(f"[MODEL] No existing index found, loading model: {self.retriever_model_path}")
                print(f"[MODEL] This may take a few minutes depending on model size and hardware...")
                self.retriever_model = RAGMultiModalModel.from_pretrained(
                    self.retriever_model_path,
                    index_root=self.index_root
                )
                print(f"[MODEL] Retriever Model loading complete!")
                
            print(f"[MODEL] Loading reranker model from: {self.reranker_model_path}")
            self.reranker_model = Reranker(self.reranker_model_path)
            print(f"[MODEL] Reranker model loading complete!")

            
            elapsed_time = time.time() - start_time
            print(f"[MODEL] Retriever Load Completed! Time elapsed: {elapsed_time:.2f} seconds")
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
        if self.retriever_model is None:
            print("[INDEX] Retriever model not loaded, attempting to load model...")
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
                self.retriever_model = RAGMultiModalModel.from_pretrained(
                    self.retriever_model_path,
                    index_root=self.index_root
                )
            
            # Check if the input path is a directory and process markdown files first
            input_path_obj = pathlib.Path(input_path)
            if input_path_obj.is_dir():
                print(f"[INDEX] Processing documents and creating index, this may take minutes to hours...")
                print(f"[INDEX] Large PDFs and multiple documents will take longer...")
                
                self.retriever_model.index(
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
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the indexed documents for relevant documents matching the query.
        
        Args:
            query: The search query.
            k: Number of results to return.
            
        Returns:
            A list of dictionaries containing the search results.
        """
        if self.retriever_model is None:
            print("[SEARCH] Model not loaded, attempting to load model...")
            if not self.load_model():
                return []
        
        try:
            start_time = time.time()
            print(f"[SEARCH] Query: \"{query}\"")
            print(f"[SEARCH] Searching for relevant documents, returning top {k} results...")
            
            results = self.retriever_model.search(query, k=k)
            
            elapsed_time = time.time() - start_time
            print(f"[SEARCH] Search completed! Time elapsed: {elapsed_time:.2f} seconds")
            print(f"[SEARCH] Found {len(results)} results")
            return results
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []
        
    def rerank(self, query: str, search_results, k: int = 3):
        """
        Rerank search results using the reranker model.
        
        Args:
            query: The search query.
            search_results: Results from the retriever model.
            k: Number of top results to return.
            
        Returns:
            List of top k reranked results with images.
        """
        if self.reranker_model is None:
            print("[RERANK] Reranker model not loaded, attempting to load model...")
            if not self.load_model():
                return []
                
        try:
            print(f"[RERANK] Reranking results for query: \"{query}\"")
            
            # Print original search results
            print("[RERANK] Original search results:")
            for i, res in enumerate(search_results):
                print(f"  Rank {i+1}: doc_id={res.doc_id}, score={res.score:.4f}")

            images = {}
            png_files = [f for f in os.listdir(self.image_path) if f.endswith(".png")]
            for idx, png in tqdm(enumerate(png_files), total=len(png_files), desc = "loading img"):
                img_path = os.path.join(self.image_path, png)
                image = Image.open(img_path)
                images[idx] = image
                
            grouped_images = []

            for res in search_results:
                grouped_images.append(images[res.doc_id])

            base64_img = []
            for img in grouped_images:
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                base64_img.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

            
            if not base64_img:
                print("[RERANK] No base64 encoded images found in search results")
                return []
                
            # Rerank the results
            result = self.reranker_model.rank(query, base64_img)
            
            # Extract top k results
            images = []
            print("[RERANK] Reranked results:")
            for i, doc in enumerate(result.top_k(k)):
                original_doc = search_results[doc.doc_id] if doc.doc_id < len(search_results) else None
                if original_doc:
                    print(f"  New Rank {i+1}: original_doc_id={original_doc.doc_id}, new_score={doc.score:.4f}")
                images.append({
                    'rank': i + 1,
                    'score': doc.score,
                    'image': doc.text,
                    'original_data': original_doc
                })
                
            print(f"[RERANK] Reranking completed, returning top {len(images)} results")
            return images
            
        except Exception as e:
            print(f"[ERROR] Reranking failed: {e}")
            return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve documents using ColQwen2 model')
    parser.add_argument('--model_path', type=str, default="../models/colqwen2-v1.0",
                      help='Path to the ColQwen2 model or model ID (default: ../models/colqwen2-v1.0)')
    parser.add_argument('--index_root', type=str, default="../data/",
                      help='Path to store the index (default: ../data/)')
    parser.add_argument('--index_name', type=str, default="rtl_optimize_rules",
                      help='Name of the index (default: rtl_optimize_rules)')
    parser.add_argument('--input_path', type=str, default="../data/image",
                      help='Path to the documents to index')
    parser.add_argument('--query', type=str, default="Give me the optimize rules for rtl performance optimization",
                      help='Query to search for (default: "Give me the optimize rules for rtl performance optimization")')
    parser.add_argument('--k', type=int, default=3,
                      help='Number of results to return (default: 3)')
    parser.add_argument('--force_index', action='store_true',
                      help='Force recreate index even if it already exists')
    parser.add_argument('--convert_pdfs', action='store_true',
                      help='Convert PDFs to PNGs in the input path')
    
    args = parser.parse_args()
    
    # Initialize RAG
    RAG = MultimodalRAG(
        retriever_model_path=args.model_path,
        index_root=args.index_root,
        index_name=args.index_name
    )
    
    indexing_path = args.input_path
    # Convert PDFs to PNGs if requested
    if args.convert_pdfs and args.input_path:
        print("\n" + "="*50)
        print(f"Converting PDFs in {args.input_path} to PNGs")
        print("="*50)
        result = RAG.pdf_to_png(args.input_path)
        print(f"Converted {len(result)} pages from PDFs to PNGs")
        image_dir = os.path.join(args.input_path, 'image')
        print(f"Images saved in {image_dir}")
        indexing_path = image_dir
    
    # Load model
    print("\n" + "="*50)
    print("Loading model, please wait...")
    print("="*50)
    if not RAG.load_model():
        print("[ERROR] Model loading failed, exiting.")
        exit(1)
    
    # Create index if input path is provided and index doesn't exist or force_index is True
    if args.input_path:
        index_path = os.path.join(args.index_root, args.index_name)
        if not os.path.exists(index_path) or args.force_index:
            print("\n" + "="*50)
            print(f"Creating index for documents: {indexing_path}")
            print("This may take some time, please be patient...")
            print("="*50)
            if not RAG.create_index(indexing_path, overwrite=args.force_index):
                print("[ERROR] Index creation failed, exiting.")
                exit(1)
        else:
            print("\n" + "="*50)
            print(f"Index already exists at: {index_path}")
            print("Skipping index creation. Use --force_index to recreate the index.")
            print("="*50)
    
    # Perform search
    print("\n" + "="*50)
    print(f"Executing search query: \"{args.query}\"")
    print("="*50)
    results = RAG.search(args.query, k=args.k)
    
    # Rerank results
    reranked_results = []
    if results:
        print("\n" + "="*50)
        print(f"Reranking top {args.k} results for query: \"{args.query}\"")
        print("="*50)
        reranked_results = RAG.rerank(args.query, results, k=args.k)

    # Print results
    print("\n" + "="*50)
    print(f"Search query: \"{args.query}\"")
    print(f"Found {len(results)} results:")
    print("="*50)
    for i, result in enumerate(results):
        print("-" * 40)
        print(f"Result {i+1}:")
        print(f"Relevance score: {result.score:.4f}")
        print(f"Document ID: {result.doc_id}")
        try:
            page_num = result.page_num
            print(f"Page: {page_num}")
        except (KeyError, TypeError, AttributeError):
            pass
        try:
            metadata = result.metadata
            if metadata:
                print(f"Metadata: {metadata}")
        except (KeyError, TypeError, AttributeError):
            pass
        try:
            if hasattr(result, 'base64') and result.base64:
                print("Contains base64 encoded document content")
        except (KeyError, TypeError, AttributeError):
            pass
    print("-" * 40)
    
    # Print reranked results
    if reranked_results:
        print("\n" + "="*50)
        print(f"Top {len(reranked_results)} Reranked results:")
        print("="*50)
        for i, result in enumerate(reranked_results):
            print("-" * 40)
            print(f"Reranked Result {i+1}:")
            print(f"Relevance score: {result['score']:.4f}")
            original_data = result.get('original_data')
            if original_data:
                print(f"Original Document ID: {original_data.doc_id}")
                try:
                    page_num = original_data.page_num
                    print(f"Page: {page_num}")
                except (KeyError, TypeError, AttributeError):
                    pass
                try:
                    metadata = original_data.metadata
                    if metadata:
                        print(f"Metadata: {metadata}")
                except (KeyError, TypeError, AttributeError):
                    pass
            try:
                if result['image']:
                    print("Contains base64 encoded document content")
            except (KeyError, TypeError, AttributeError):
                pass
        print("-" * 40)
        
    # Example of how to use the Vision-Language model for answer generation
    if reranked_results:
        print("\n" + "="*50)
        print("Generating answer using Vision-Language Model...")
