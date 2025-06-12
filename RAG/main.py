import base64
from io import BytesIO
import os
import sys
import argparse
import pathlib
import time
from tqdm import tqdm
from pdf2image import convert_from_path
from typing import List, Dict, Any
import numpy
from byaldi import RAGMultiModalModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from rerankers import Reranker
from PIL import Image
import torch




from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

class MultimodalRAG:
    """
    Retrieves relevant documents using the ColQwen2 model for document retrieval.
    Uses the byaldi library which provides a simple interface to the colpali-engine.
    Supports PDF and image files.
    """
    def __init__(self, input_path: str = "../data/", retriever_model_path: str = "../models/colqwen2-v1.0", index_root: str = "../models/byaldi/", index_name: str = "rtl_optimize_rules", reranker_model_path: str = "../models/MonoQwen2-VL-v0.1", vl_model_path: str = "../models/BAGEL-7B-MoT"):
        self.input_path = input_path
        self.retriever_model_path = retriever_model_path
        self.index_root = index_root
        self.image_path = self.input_path + "image/"
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
        """Load the retriever, reranker, and vision-language models."""
        try:
            start_time = time.time()
            print("[MODEL] Starting model or index loading...")
            
            if self.retriever_model is None:
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
                
            if self.reranker_model is None:
                print(f"[MODEL] Loading reranker model from: {self.reranker_model_path}")
                self.reranker_model = Reranker(self.reranker_model_path)
                print(f"[MODEL] Reranker model loading complete!")

            if self.vl_model is None:
                print(f"[MODEL] Loading VL model from: {self.vl_model_path}")
                model_path = self.vl_model_path
                
                # LLM config preparing
                llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
                llm_config.qk_norm = True
                llm_config.tie_word_embeddings = False
                llm_config.layer_module = "Qwen2MoTDecoderLayer"

                # ViT config preparing
                vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
                vit_config.rope = False
                vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

                # VAE loading
                print(f"[MODEL] Loading VAE model from: {os.path.join(model_path, 'ae.safetensors')}")
                vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
                vae_model.to(dtype=torch.bfloat16)
                print(f"[MODEL] VAE model loading complete!")

                # Bagel config preparing
                config = BagelConfig(
                    visual_gen=True,
                    visual_und=True,
                    llm_config=llm_config, 
                    vit_config=vit_config,
                    vae_config=vae_config,
                    vit_max_num_patch_per_side=70,
                    connector_act='gelu_pytorch_tanh',
                    latent_patch_size=2,
                    max_latent_size=64,
                )

                with init_empty_weights():
                    language_model = Qwen2ForCausalLM(llm_config)
                    vit_model      = SiglipVisionModel(vit_config)
                    model          = Bagel(language_model, vit_model, config)
                    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

                print(f"[MODEL] init_empty_weights complete!")

                # Tokenizer Preparing
                tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
                tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

                # Image Transform Preparing
                vae_transform = ImageTransform(1024, 512, 16)
                vit_transform = ImageTransform(980, 224, 14)
                
                max_mem_per_gpu = "16GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

                device_map = infer_auto_device_map(
                    model,
                    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
                    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
                )
                print(device_map)

                same_device_modules = [
                    'language_model.model.embed_tokens',
                    'time_embedder',
                    'latent_pos_embed',
                    'vae2llm',
                    'llm2vae',
                    'connector',
                    'vit_pos_embed'
                ]

                if torch.cuda.device_count() == 1:
                    first_device = device_map.get(same_device_modules[0], "cuda:0")
                    for k in same_device_modules:
                        if k in device_map:
                            device_map[k] = first_device
                        else:
                            device_map[k] = "cuda:0"
                else:
                    first_device = device_map.get(same_device_modules[0])
                    for k in same_device_modules:
                        if k in device_map:
                            device_map[k] = first_device
                
                model = load_checkpoint_and_dispatch(
                    model,
                    checkpoint=os.path.join(model_path, "ema.safetensors"),
                    device_map=device_map,
                    offload_buffers=True,
                    dtype=torch.bfloat16,
                    force_hooks=True,
                    offload_folder="/tmp/offload"
                )
                print(f"[MODEL] VL model checkpoint loading complete!")

                model = model.eval()
                
                inferencer = InterleaveInferencer(
                    model=model, 
                    vae_model=vae_model, 
                    tokenizer=tokenizer, 
                    vae_transform=vae_transform, 
                    vit_transform=vit_transform, 
                    new_token_ids=new_token_ids
                )
                self.vl_model = inferencer
                print("[MODEL] VL model loading complete!")
            
            elapsed_time = time.time() - start_time
            print(f"[MODEL] Model loading complete! Time elapsed: {elapsed_time:.2f} seconds")
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
        
        # We need tempfile for the clean directory approach
        import tempfile
        
        try:
            start_time = time.time()
            print(f"[INDEX] Creating index for image path: {input_path + '/image/'}")
            print(f"[INDEX] Index name: {self.index_name}")
            
            if overwrite and os.path.exists(os.path.join(self.index_root, self.index_name)):
                import shutil
                print(f"[INDEX] Completely removing existing index: {self.index_name}")
                shutil.rmtree(os.path.join(self.index_root, self.index_name))
                print(f"[INDEX] Reloading model to ensure clean state...")
                self.retriever_model = RAGMultiModalModel.from_pretrained(
                    self.retriever_model_path,
                    index_root=self.index_root
                )
            
            input_path_obj = pathlib.Path(input_path + '/image/')

            # Use a context manager for a temporary directory to ensure cleanup
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"[INDEX] Creating temporary directory for indexing: {temp_dir}")
                temp_path_obj = pathlib.Path(temp_dir)
                index_source_path = temp_dir

                if input_path_obj.is_dir():
                    # Only index .png files and sort them to have a consistent order for doc_id mapping
                    files_to_link = sorted(list(input_path_obj.rglob('*.png')))
                    print(f"[INDEX] Found {len(files_to_link)} PNG files to process for indexing.")

                    if not files_to_link:
                        print(f"[WARN] No suitable files found to index in {input_path}.")
                        return True

                    for source_path in files_to_link:
                        relative_path = source_path.relative_to(input_path_obj)
                        dest_path = temp_path_obj / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(source_path.resolve(), dest_path)
                
                elif input_path_obj.is_file():
                    os.symlink(input_path_obj.resolve(), temp_path_obj / input_path_obj.name)
                else:
                    print(f"[ERROR] Input path {input_path} is not a valid file or directory.")
                    return False

                print(f"[INDEX] Starting indexing from temporary path: {index_source_path}")
                self.retriever_model.index(
                    input_path=index_source_path,
                    index_name=self.index_name,
                    store_collection_with_index=store_collection_with_index,
                    # Overwrite is handled by deleting the index folder above, 
                    # so we pass False here to avoid issues with the library.
                    overwrite=False 
                )

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print(f"[INDEX] Index created successfully: {self.index_name}")
            print(f"[INDEX] Completed! Time elapsed: {minutes}m {seconds}s")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to create index: {e}")
            import traceback
            traceback.print_exc()
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

            # Get the list of all png files, sorted to match indexing order
            image_path_obj = pathlib.Path(self.image_path)
            all_png_files = sorted(list(image_path_obj.rglob('*.png')))

            if not all_png_files:
                print(f"[WARN] No PNG files found in {self.image_path}")
                return []

            grouped_images = []
            original_indices = []
            print("[RERANK] Loading images for reranking...")
            for res in tqdm(search_results, desc="Loading images"):
                try:
                    image_path = all_png_files[res.doc_id]
                    image = Image.open(image_path)
                    grouped_images.append(image)
                    original_indices.append(res.doc_id)
                except IndexError:
                    print(f"[WARN] doc_id {res.doc_id} is out of bounds. Total images: {len(all_png_files)}")
                    continue
                except FileNotFoundError:
                    print(f"[WARN] Image file not found for doc_id {res.doc_id} at path {image_path}")
                    continue
            
            if not grouped_images:
                print("[RERANK] No images could be loaded for reranking.")
                return []

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
                original_doc_index = doc.doc_id
                original_doc_id = original_indices[original_doc_index]
                
                # Find the original search result corresponding to the original doc_id
                original_doc = next((res for res in search_results if res.doc_id == original_doc_id), None)

                image_name = "N/A"
                if original_doc:
                    print(f"  New Rank {i+1}: original_doc_id={original_doc.doc_id}, new_score={doc.score:.4f}")
                    try:
                        image_name = all_png_files[original_doc.doc_id].name
                    except IndexError:
                        print(f"[WARN] doc_id {original_doc.doc_id} is out of bounds for image name lookup.")

                images.append({
                    'rank': i + 1,
                    'score': doc.score,
                    'image': doc.text,
                    'original_data': original_doc,
                    'image_name': image_name
                })
                
            print(f"[RERANK] Reranking completed, returning top {len(images)} results")
            return images
            
        except Exception as e:
            print(f"[ERROR] Reranking failed: {e}")
            return []

    def generate_answer(self, image: Image.Image, prompt: str):
        """
        Generate an answer using the Vision-Language model.
        """
        if self.vl_model is None:
            print("[VLMODEL] Vision-Language model not loaded, attempting to load model...")
            if not self.load_model():
                return "Failed to load VL model."

        try:
            start_time = time.time()
            print(f"[VLMODEL] Generating answer for prompt: \"{prompt}\"")

            # The hyperparameters are from the "Understanding" section of the notebook
            inference_hyper=dict(
                max_think_token_n=1000,
                do_sample=False,
                # text_temperature=0.3,
            )
            
            output_dict = self.vl_model(image=image, text=prompt, understanding_output=True, **inference_hyper)
            
            answer = output_dict['text']
            
            elapsed_time = time.time() - start_time
            print(f"[VLMODEL] Answer generation completed! Time elapsed: {elapsed_time:.2f} seconds")
            
            return answer
        except Exception as e:
            print(f"[ERROR] Answer generation failed: {e}")
            return f"Error during answer generation: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve documents using ColQwen2 model')
    parser.add_argument('--retriever_model_path', type=str, default='../models/colqwen2-v1.0', help='Path to the ColQwen2 model')
    parser.add_argument('--index_root', type=str, default='../index/', help='Path to the root directory for the index')
    parser.add_argument('--index_name', type=str, default='rtl_optimize_rules', help='Name of the index')
    parser.add_argument('--reranker_model_path', type=str, default='../models/MonoQwen2-VL-v0.1', help='Path to the reranker model')
    parser.add_argument('--vl_model_path', type=str, default='../models/BAGEL-7B-MoT', help='Path to the Vision-Language model')
    parser.add_argument('--input_path', type=str, default='../data/', help='Path to the directory containing documents or a single document file')
    parser.add_argument('--query', type=str, help='Query to search for')
    parser.add_argument('--image_path', type=str, default='../data/image/', help='Path to an image file for context')
    parser.add_argument('--task', type=str, choices=['create_index', 'search', 'rerank', 'generate_answer'], default='search', help='Task to perform')
    parser.add_argument('--k', type=int, default=5, help='Number of documents to retrieve')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing index')
    
    args = parser.parse_args()

    rag = MultimodalRAG(
        input_path=args.input_path,
        retriever_model_path=args.retriever_model_path,
        index_root=args.index_root,
        index_name=args.index_name,
        reranker_model_path=args.reranker_model_path,
        vl_model_path=args.vl_model_path
    )
    
    if args.task == 'create_index':
        rag.create_index(args.input_path, overwrite=args.overwrite)
    elif args.task == 'search':
        if not args.query:
            print("Please provide a query for searching.")
        else:
            results = rag.search(args.query, k=args.k)
            print("Search results:")
            for res in results:
                print(f" Doc_id: {res.doc_id}: (Score: {res.score})")
    elif args.task == 'rerank':
        if not args.query:
            print("Please provide a query for reranking.")
        else:
            search_results = rag.search(args.query, k=10) 
            reranked_results = rag.rerank(args.query, search_results, k=args.k)
            print("Reranked results:")
            for res in reranked_results:
                if res.get('original_data'):
                    print(f" Doc_id: {res['original_data'].doc_id}: (Score: {res['score']}) Image: {res.get('image_name', 'N/A')}")
                else:
                    print(f" Reranked item with score: {res['score']} but missing original data.")
    elif args.task == 'generate_answer':
        if not args.query:
            print("Please provide a query for answer generation.")
        else:
            print("[GENERATE_ANSWER] Searching for relevant documents...")
            search_results = rag.search(args.query, k=10)
            
            if not search_results:
                print("[GENERATE_ANSWER] No documents found for the query.")
            else:
                print("[GENERATE_ANSWER] Reranking search results to find the best image...")
                reranked_results = rag.rerank(args.query, search_results, k=1)
                
                if not reranked_results:
                    print("[GENERATE_ANSWER] Reranking failed or returned no results.")
                else:
                    top_result = reranked_results[0]
                    image_name = top_result.get('image_name')

                    if not image_name or image_name == "N/A":
                        print("[GENERATE_ANSWER] Reranking did not yield a valid image name.")
                    else:
                        try:
                            image_path = os.path.join(rag.image_path, image_name)
                            print(f"[GENERATE_ANSWER] Using image '{image_name}' for answer generation.")
                            image = Image.open(image_path)
                            
                            answer = rag.generate_answer(image, args.query)
                            print("\n--- Generated Answer ---")
                            print(answer)
                            print("------------------------")
                        except FileNotFoundError:
                            print(f"Error: Reranked image file not found at {image_path}")
                        except Exception as e:
                            print(f"An error occurred during answer generation: {e}")

    # Example usage:
    # 1. Create Index:
    # python main.py --task create_index --input_path ../data/pdf/ --overwrite
    #
    # 2. Search:
    # python main.py --task search --query "How to optimize LUT?" --input_path ../index/optimize_rules/
    #
    # 3. Rerank:
    # python main.py --task rerank --query "How to optimize LUT?" --input_path ../index/rtl_optimize_rules/ --k 3
    #
    # 4. Generate Answer:
    # python main.py --task generate_answer --query "How to optimize this circuit?" --image_path ../data/rtl_optimize_rules/image/d36_1.png --input_path ../data/rtl_optimize_rules/