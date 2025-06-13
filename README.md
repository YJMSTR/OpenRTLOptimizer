Use part of AMD UG949, AMD UG901 and some AI-gen optimize rules as knowledge base.


```bash
pip install -r requirements.txt

bash download_models.sh

cd models/colqwen2-v1.0 && git lfs pull

cd ../colqwen2-base && git lfs pull

cd ../MonoQwen2-VL-v0.1 && git lfs pull

cd ../../RAG

python main.py --model_path ../models/colqwen2-v1.0 --index_root ../models/byaldi/ --index_name rtl_optimize_rules --input_path ../data --query "Give me the optimize rules for rtl performance optimization" --k 3
```

Change the model_type in the config.json to "colqwen2"

Change the `base_model_name_or_path` in the adapter_config.json to "../models/colqwen2-base"


Example:

`cd RAG && python main.py --task generate_answer --query "how to decrease area of chip, based on the given image?" --k 4`

Example output:


>[GENERATE_ANSWER] Reranking search results to find the best image...
>[MODEL] Loading reranker model from: ../models/MonoQwen2-VL-v0.1
>Loading MonoVLMRanker model ../models/MonoQwen2-VL-v0.1 (this message can be suppressed by setting verbose=0)
>No device set
>Using device cuda
>bf16
>Using dtype torch.bfloat16
>Loading model ../models/MonoQwen2-VL-v0.1, this might take a while...
>Using device cuda.
>Using dtype torch.bfloat16.
>Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.81it/s]
>WARNING: Model ../models/MonoQwen2-VL-v0.1 does not have known True/False tokens. Defaulting token_false to `False`.
>WARNING: Model ../models/MonoQwen2-VL-v0.1 does not have known True/False tokens. Defaulting token_true to `True`.
>VLM true token set to True
>VLM false token set to False
>[MODEL] Reranker model loading complete!
>[MODEL] Reranker model loading complete! Time elapsed: 6.10 seconds
>[RERANK] Reranking results for query: "how to decrease area of chip, based on the given image?"
>[RERANK] Original search results:
>  Rank 1: doc_id=67, score=13.3125
>  Rank 2: doc_id=77, score=12.3125
>  Rank 3: doc_id=88, score=11.7500
>  Rank 4: doc_id=148, score=11.6250
>  Rank 5: doc_id=34, score=11.5000
>  Rank 6: doc_id=94, score=11.5000
>  Rank 7: doc_id=120, score=11.5000
>  Rank 8: doc_id=129, score=11.4375
>  Rank 9: doc_id=161, score=11.3125
>  Rank 10: doc_id=76, score=11.2500
>[RERANK] Loading images for reranking...
>Loading images: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 2516.99it/s]
>[RERANK] Reranked results:
>  New Rank 1: original_doc_id=129, new_score=0.0105
>  New Rank 2: original_doc_id=77, new_score=0.0074
>  New Rank 3: original_doc_id=67, new_score=0.0059
>  New Rank 4: original_doc_id=88, new_score=0.0052
>[RERANK] Reranking completed, returning top 4 results
>[MODEL] Unloading reranker model...
>[MODEL] Reranker model unloaded.
>[MODEL] Loading VL model from: ../models/BAGEL-7B-MoT
>[MODEL] Loading VAE model from: ../models/BAGEL-7B-MoT/ae.safetensors
>[MODEL] VAE model loading complete!
>[MODEL] init_empty_weights complete!
>OrderedDict([('language_model.model.embed_tokens', 0), ('language_model.model.layers.0', 0), ('language_model.model.layers.1', 0), ('language_model.model.layers.2', 0), ('language_model.model.layers.3', 0), ('language_model.model.layers.4', 0), ('language_model.model.layers.5', 0), ('language_model.model.layers.6', 0), ('language_model.model.layers.7', 0), ('language_model.model.layers.8', 0), ('language_model.model.layers.9', 0), ('language_model.model.layers.10', 0), ('language_model.model.layers.11', 1), ('language_model.model.layers.12', 1), ('language_model.model.layers.13', 1), ('language_model.model.layers.14', 1), ('language_model.model.layers.15', 1), ('language_model.model.layers.16', 1), ('language_model.model.layers.17', 1), ('language_model.model.layers.18', 1), ('language_model.model.layers.19', 1), ('language_model.model.layers.20', 1), ('language_model.model.layers.21', 1), ('language_model.model.layers.22', 1), ('language_model.model.layers.23', 1), ('language_model.model.layers.24', 2), ('language_model.model.layers.25', 2), ('language_model.model.layers.26', 2), ('language_model.model.layers.27', 2), ('language_model.model.norm', 2), ('language_model.model.norm_moe_gen', 2), ('language_model.model.rotary_emb', 2), ('language_model.lm_head', 2), ('time_embedder', 2), ('vae2llm', 2), ('llm2vae', 2), ('latent_pos_embed', 2), ('vit_model', 2), ('connector', 2), ('vit_pos_embed', 2)])
>The safetensors archive passed at ../models/BAGEL-7B-MoT/ema.safetensors does not contain metadata. Make sure to save your model with the `save_pretrained` method. Defaulting to 'pt' metadata.
>We've detected an older driver with an RTX 4000 series GPU. These drivers have issues with P2P. This can affect the multi-gpu inference when using accelerate device_map.Please make sure to update your driver to the latest version which resolves this.     
>[MODEL] VL model checkpoint loading complete!
>[MODEL] VL model loading complete!
>[MODEL] VL model loading complete! Time elapsed: 6.62 seconds
>[GENERATE_ANSWER] Using 4 images for answer generation: ['AMD UG901_page_91.png', 'AMD UG901_page_44.png', 'AMD UG901_page_35.png', 'AMD UG901_page_54.png']
>[VLMODEL] Generating answer for prompt: "how to decrease area of chip, based on the given image?" with 4 images.
>[VLMODEL] Answer generation completed! Time elapsed: 21.26 seconds
>[MODEL] Unloading VL model...
>You shouldn't move a model that is dispatched using accelerate hooks.
>[MODEL] VL model unloaded.
>
>--- Generated Answer ---
>To decrease the area of a chip, you can consider the following strategies based on the given images:
>
>1. **Optimize RAM Design:**
>   - **Asymmetric RAM:** The first image shows a simple dual-port asymmetric RAM example. By optimizing the RAM design, you can reduce the area by minimizing the number of bits and rows/columns used. For instance, using a smaller width (WIDTHA) and a smaller address width (ADDRWIDTHA) can reduce the area.
>   - **Pipelining the RAM:** The fourth image discusses pipelining the RAM. By adding pipeline registers to the RAM output in RTL, you can reduce the number of RAMs needed. This can significantly decrease the area. Calculate the number of pipeline registers by adding the number of rows and columns in the RAM matrix.
>
>2. **Efficient Use of Resources:**
>   - **Systolic Filters:** The third image shows an 8-tap even symmetric systolic FIR filter. Using systolic filters can reduce the area by leveraging parallel processing and shared resources. Systolic filters are particularly useful for reducing the area of filters.
>
>3. **Pattern Detection and Optimization:**
>   - **Convergent Rounding:** The second image demonstrates convergent rounding using pattern detection. By optimizing the pattern detection logic, you can reduce the area of the chip. This involves minimizing the number of logic gates and registers used in the pattern detection process.
>
>4. **Implementation Techniques:**
>   - **Using UltraRAMs:** The fourth image mentions the use of UltraRAMs (URAMs). By using URAMs, you can reduce the area by minimizing the number of RAMs needed. URAMs support pipelining registers, which can further reduce the area.
>
>5. **Design Optimization Tools:**
>   - Utilize design optimization tools provided by the synthesis suite (e.g., Vivado) to automatically optimize the design for area reduction. These tools can help in identifying and implementing the most efficient design.
>
>By applying these strategies, you can effectively decrease the area of the chip while maintaining or improving its performance.