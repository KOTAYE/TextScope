# -*- coding: utf-8 -*-
"""
Text Clustering module.
Uses scikit-learn TF-IDF Vectorizer and Cosine Similarity to cluster similar reviews in batch analysis.
"""

from typing import List, Dict, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cluster_reviews(texts: List[str], threshold: float = 0.7) -> List[Dict[str, Union[int, List[int]]]]:
    """
    Groups similar reviews together based on TF-IDF and cosine similarity.
    
    Args:
        texts (List[str]): List of input review texts.
        threshold (float): Similarity threshold above which reviews are considered similar (default 0.7).
        
    Returns:
        List[Dict[str, Union[int, List[int]]]]: List of clusters where each cluster lists the row indices of members.
    """
    if not texts or len(texts) < 2:
        return []
        
    cleaned_texts = [t.lower().strip() for t in texts]
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    except ValueError:
        # Fallback vectorizer if default config encounters issues
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        except Exception:
            return []
            
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    assigned = set()
    clusters = []
    cluster_id = 1
    
    for i in range(len(texts)):
        if i in assigned:
            continue
            
        cluster_members = [i]
        assigned.add(i)
        
        for j in range(i + 1, len(texts)):
            if j not in assigned and sim_matrix[i][j] >= threshold:
                cluster_members.append(j)
                assigned.add(j)
                
        # Only list clusters containing actual multiple similar entries
        if len(cluster_members) > 1:
            clusters.append({
                'cluster_id': cluster_id,
                'review_indices': cluster_members
            })
            cluster_id += 1
            
    return clusters
