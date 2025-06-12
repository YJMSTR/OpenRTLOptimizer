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
