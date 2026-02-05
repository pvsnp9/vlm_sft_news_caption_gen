from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from config.logistics import Logistics

def load_xl_munichus(splitname, lang=None, sample_size=None):
    """
    Loads the XL-MUNIChus dataset by split and language.
    If lang is None, it aggregates all languages for that split.
    """
    logistics = Logistics()
    path = logistics.hf_datatset_id
    cache_dir = logistics.hf_cache_dir
    use_sample = isinstance(sample_size, int) and sample_size is not None
    
    if lang is not None:
        # Load a specific language configuration
        print(f"Loading {lang} for split: {splitname}...")
        ds = load_dataset(path, lang, split=splitname, cache_dir=cache_dir)
        if use_sample:
            ds = ds.select(range(min(sample_size, len(ds))))
        return ds, [lang]
    
    else:
        # Fetch all available language configurations
        all_langs = get_dataset_config_names(path)
        print(f"Loading all languages ({len(all_langs)}) for split: {splitname}...")
        
        dataset_list = []
        failures = []
        for l in all_langs:
            try:
                # Load each language subset for the specific split
                ds = load_dataset(path, l, split=splitname, cache_dir=cache_dir)
                if use_sample:
                    ds = ds.select(range(min(sample_size, len(ds))))
                # Add a column to keep track of which language the row belongs to
                ds = ds.add_column("language_code", [l] * len(ds))
                dataset_list.append(ds)
            except Exception as e:
                failures.append(f"{l}: {e}")
                print(f"Could not load language {l}: {e}")
        
        # Merge all languages into a single Dataset object
        if not dataset_list:
            raise RuntimeError(
                f"No datasets loaded for split '{splitname}'. Failures: {', '.join(failures) or 'none'}"
            )
        return concatenate_datasets(dataset_list), all_langs



def load_train_eval_sft_dataset(sample_size=None):
    """
    load the training dataset using load_xl_munichus.
    out of loaded dataset, separate train and eval splits.
    the splits is 90% train, 10% eval.
    during split: apply it for each language to ensure all languages are represented.
    use seed=42 for reproducibility.
    Returns:
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
    """
    try:
        dataset, _langs = load_xl_munichus("train", lang=None, sample_size=sample_size)
        split = dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)
        return split["train"], split["test"]
    except Exception as exc:
        raise RuntimeError(f"load_train_eval_sft_dataset failed: {exc}") from exc


# if __name__ == "__main__":
#     # train_all, langs = load_xl_munichus("train")
#     train_ds, eval_ds = load_train_eval_sft_dataset(sample_size=1)
#     print(f"train samples loaded:{len(train_ds)} eval samples loaded:{len(eval_ds)}")
#     # lang = []
#     # for ev in eval_ds:
#     #     if ev['language'] not in lang:
#     #         print(f"Eval sample language: {ev['language']}")
#     #         lang.append(ev['language'])
        
#     print(f"sample languages loaded:{train_ds[0].keys()} \n Eval sample:{eval_ds[0]}")
#     # test_en = load_xl_munichus("test", lang="en")
