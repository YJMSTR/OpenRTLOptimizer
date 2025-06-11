Use part of AMD UG949, AMD UG901 and some AI-gen optimize rules as knowledge base.


```bash
pip install -r requirements.txt

bash download_models.sh

cd models/colqwen2-v1.0 && git lfs pull

cd ../colqwen2-base && git lfs pull
```

Change the model_type in the config.json to "colqwen2"

Change the `base_model_name_or_path` in the adapter_config.json to "../models/colqwen2-base"
