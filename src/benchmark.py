import json
import numpy as np
from dataset import clean, label_encode

METADATA_CATEGORIES = {
    "same_article_type": "articleType",
    "same_color":        "baseColour",
    "same_gender":       "gender",
}

GROUPING_CATEGORIES = {
    "footwear":   ["Heels", "Flats", "Formal Shoes", "Casual Shoes", "Sports Shoes", "Sandals", "Flip Flops"],
    "outerwear":  ["Jackets", "Sweaters", "Sweatshirts"],
    "tops":       ["Tshirts", "Tops", "Shirts", "Sweaters", "Sweatshirts", "Jackets"],
    "bottom_wear":["Jeans", "Trousers", "Shorts", "Track Pants", "Capris", "Leggings"],
    "accessories":["Sunglasses", "Watches", "Belts", "Ties", "Caps", "Socks",
                   "Earrings", "Necklace and Chains", "Pendant", "Handbags", "Clutches"],
    "bags":       ["Handbags", "Clutches"],
    "jewelry":    ["Earrings", "Necklace and Chains", "Pendant"],
    "innerwear":  ["Bra", "Briefs", "Innerwear Vests"],
    "casual_wear":["Tshirts", "Tops", "Shirts", "Shorts", "Jeans", "Casual Shoes"],
    "dresses":    ["Dresses"],
    "rare_class": ["Ties", "Capris"],
}

def get_ground_truth(query_id, category, df):
    """Returns a set of image ID strings that are relevant to query_id
    under the given category, excluding the query itself"""
    query_id = str(query_id)
    row = df[df['id'].astype(str) == query_id]
    if row.empty:
        return set()
 
    if category in METADATA_CATEGORIES:
        col = METADATA_CATEGORIES[category]
        query_val = row.iloc[0][col]
        relevant = df[df[col] == query_val]['id'].astype(str).tolist()
 
    elif category in GROUPING_CATEGORIES:
        relevant_types = GROUPING_CATEGORIES[category]
        relevant = df[df['articleType'].isin(relevant_types)]['id'].astype(str).tolist()
 
    else:
        raise ValueError(f"Unknown category: {category}")
 
    return set(relevant) - {query_id}

def select_query_images(df, n=10, min_gt_size=20, random_state=42):
    """
    For each category, selects n fixed query images whose ground truth set is at least min_gt_size. 
    Returns a dict where category maps to a list of n id strings.
    """
    rng = np.random.default_rng(random_state)
    benchmark = {}
 
    all_categories = list(METADATA_CATEGORIES.keys()) + list(GROUPING_CATEGORIES.keys())
 
    for category in all_categories:
        # build pool of valid query candidates
        if category in METADATA_CATEGORIES:
            candidates = df['id'].astype(str).tolist()
        else:
            relevant_types = GROUPING_CATEGORIES[category]
            candidates = df[df['articleType'].isin(relevant_types)]['id'].astype(str).tolist()
 
        # filter to candidates with ground truth size >= min_gt_size
        qualified = [img_id for img_id in candidates if len(get_ground_truth(img_id, category, df)) >= min_gt_size]
 
        if len(qualified) < n:
            print(f"IMPORTANT: {category} only has {len(qualified)} qualified candidates (need {n})")
            selected = qualified
        else:
            indices = rng.choice(len(qualified), size=n, replace=False)
            selected = [qualified[i] for i in sorted(indices)]
 
        benchmark[category] = selected
        print(f"  {category}: {len(selected)} queries selected")
 
    return benchmark
 
 
if __name__ == '__main__':
    df = clean()
    class_names, _ = label_encode(df)
 
    benchmark_queries = select_query_images(df, n=10, min_gt_size=20)
 
    for category, query_ids in benchmark_queries.items():
        gt_sizes = [len(get_ground_truth(qid, category, df)) for qid in query_ids]
        avg_gt = sum(gt_sizes) / len(gt_sizes) if gt_sizes else 0
        print(f"  {category}: {len(query_ids)} queries | avg ground truth size: {avg_gt:.0f}")
 
    with open('data/benchmark_queries.json', 'w') as f:
        json.dump(benchmark_queries, f, indent=2)
    print("\nsaved to data/benchmark_queries.json")
 