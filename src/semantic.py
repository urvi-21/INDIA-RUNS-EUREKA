from __future__ import annotations

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class SemanticSimilarityModel:
    """
    Lightweight Latent Semantic Analysis model.

    Unlike pure keyword matching, TF-IDF + SVD projects both the Job
    Description and candidate evidence into a latent semantic space.

    This allows related concepts to become close even when they do not
    literally share the same vocabulary.
    """

    def __init__(self,
                 n_components: int = 256,
                 random_state: int = 42):

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=2,
            max_features=75000,
            sublinear_tf=True
        )

        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=random_state
        )

        self._fitted = False

    def fit(self, corpus: list[str]):

        X = self.vectorizer.fit_transform(corpus)

        self.svd.fit(X)

        self._fitted = True

        return self

    def transform(self, texts: list[str]):

        if not self._fitted:
            raise RuntimeError("Model must be fit before transform().")

        X = self.vectorizer.transform(texts)

        X = self.svd.transform(X)

        return normalize(X)

    def similarity_to_jd(
        self,
        candidate_texts: list[str],
        jd_text: str
    ) -> np.ndarray:

        candidate_vectors = self.transform(candidate_texts)

        jd_vector = self.transform([jd_text])[0]

        similarities = candidate_vectors @ jd_vector

        return similarities

    def top_terms_for_text(
        self,
        text: str,
        n: int = 15
    ):

        vec = self.vectorizer.transform([text])

        feature_names = np.array(
            self.vectorizer.get_feature_names_out()
        )

        row = vec.toarray()[0]

        idx = row.argsort()[::-1][:n]

        return [
            feature_names[i]
            for i in idx
            if row[i] > 0
        ]