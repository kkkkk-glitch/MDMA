import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class SimilarityRetriever:
    """
    Similarity Retrieval module for MDMA framework.

    Uses a frozen pre-trained Transformer encoder (roberta-base) to compute
    text embeddings and retrieve similar samples from a reference set.
    """

    def __init__(self,
                 encoder_model_name: str = "roberta-base",
                 device: str = "cpu",
                 cache_dir: Optional[str] = None):
        """
        Initialize the Similarity Retriever.

        Args:
            encoder_model_name: Name of the pre-trained model for text encoding
            device: Device to run the model on ('cpu' or 'cuda')
            cache_dir: Directory to cache embeddings
        """
        self.encoder_model_name = encoder_model_name
        self.device = device
        self.cache_dir = cache_dir or "./embeddings_cache"

        # Load the pre-trained encoder
        print(f"Loading encoder: {encoder_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        self.model = AutoModel.from_pretrained(encoder_model_name)
        self.model.to(device)
        self.model.eval()

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        self.reference_embeddings = None
        self.reference_texts = None
        self.reference_ids = None

    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling to get sentence embeddings.

        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask from tokenizer

        Returns:
            Mean-pooled embeddings
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding

        Returns:
            NumPy array of embeddings (shape: n_texts x embedding_dim)
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Encode
            with torch.no_grad():
                outputs = self.model(**encoded)
                batch_embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def build_knowledge_base(self,
                             knowledge_base_file: str,
                             text_column: str = "text",
                             save_embeddings: bool = True):
        """
        Build knowledge base from a .txt file or JSON file.

        Args:
            knowledge_base_file: Path to the .txt or .json file
            text_column: Column name for text if JSON format
            save_embeddings: Whether to save embeddings to cache

        Returns:
            Number of samples in the knowledge base
        """
        print(f"Building knowledge base from: {knowledge_base_file}")

        # Load data based on file extension
        file_ext = Path(knowledge_base_file).suffix.lower()

        if file_ext == '.txt':
            texts = self._load_txt_file(knowledge_base_file)
            ids = [f"sample_{i}" for i in range(len(texts))]
        elif file_ext == '.json':
            texts, ids = self._load_json_file(knowledge_base_file, text_column)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .txt or .json")

        print(f"Loaded {len(texts)} samples from knowledge base")

        # Check cache
        cache_file = os.path.join(self.cache_dir, f"embeddings_{Path(knowledge_base_file).stem}.pkl")

        if os.path.exists(cache_file) and save_embeddings:
            print(f"Loading cached embeddings from: {cache_file}")
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            print("Encoding texts...")
            embeddings = self.encode_texts(texts)
            print(f"Encoded {len(texts)} texts, embedding dimension: {embeddings.shape[1]}")

            # Cache embeddings
            if save_embeddings:
                print(f"Saving embeddings to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(embeddings, f)

        self.reference_texts = texts
        self.reference_ids = ids
        self.reference_embeddings = embeddings

        return len(texts)

    def _load_txt_file(self, file_path: str) -> List[str]:
        """
        Load texts from a .txt file.

        Args:
            file_path: Path to .txt file

        Returns:
            List of text strings
        """
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    texts.append(line)
        return texts

    def _load_json_file(self, file_path: str, text_column: str = "text") -> Tuple[List[str], List[str]]:
        """
        Load texts from a JSON file.

        Args:
            file_path: Path to JSON file
            text_column: Column name for text field

        Returns:
            Tuple of (texts, ids)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        ids = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    texts.append(item.get(text_column, ''))
                    ids.append(item.get('id', f"sample_{len(ids)}"))
                else:
                    texts.append(str(item))
                    ids.append(f"sample_{len(ids)}")
        elif isinstance(data, dict):
            # Assume data is a dict with ids as keys and texts as values
            for k, v in data.items():
                if isinstance(v, dict):
                    texts.append(v.get(text_column, ''))
                    ids.append(k)
                else:
                    texts.append(str(v))
                    ids.append(k)

        return texts, ids

    def retrieve_similar(self,
                         target_text: str,
                         k: int = 4,
                         min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Retrieve top-K most similar texts from the knowledge base.

        Args:
            target_text: The target text to find similar samples for
            k: Number of similar samples to retrieve
            min_similarity: Minimum similarity threshold

        Returns:
            List of tuples (text, similarity_score) for top-K similar samples
        """
        if self.reference_embeddings is None:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")

        # Encode target text
        target_embedding = self.encode_texts([target_text])

        # Compute cosine similarity
        similarities = cosine_similarity(target_embedding, self.reference_embeddings)[0]

        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:k]

        # Filter by minimum similarity
        results = []
        for idx in top_indices:
            if similarities[idx] >= min_similarity:
                results.append((self.reference_texts[idx], float(similarities[idx])))

        return results

    def retrieve_similar_with_ids(self,
                                  target_text: str,
                                  k: int = 4,
                                  min_similarity: float = 0.0) -> List[Tuple[str, str, float]]:
        """
        Retrieve top-K most similar texts with their IDs.

        Args:
            target_text: The target text to find similar samples for
            k: Number of similar samples to retrieve
            min_similarity: Minimum similarity threshold

        Returns:
            List of tuples (id, text, similarity_score)
        """
        if self.reference_embeddings is None:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")

        # Encode target text
        target_embedding = self.encode_texts([target_text])

        # Compute cosine similarity
        similarities = cosine_similarity(target_embedding, self.reference_embeddings)[0]

        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:k]

        # Filter by minimum similarity
        results = []
        for idx in top_indices:
            if similarities[idx] >= min_similarity:
                results.append((self.reference_ids[idx], self.reference_texts[idx], float(similarities[idx])))

        return results

    def batch_retrieve(self,
                       target_texts: List[str],
                       k: int = 4,
                       min_similarity: float = 0.0) -> List[List[Tuple[str, float]]]:
        """
        Batch retrieve similar texts for multiple targets.

        Args:
            target_texts: List of target texts
            k: Number of similar samples to retrieve per target
            min_similarity: Minimum similarity threshold

        Returns:
            List of lists of (text, similarity_score)
        """
        if self.reference_embeddings is None:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")

        # Encode all target texts
        target_embeddings = self.encode_texts(target_texts)

        # Compute cosine similarities
        all_similarities = cosine_similarity(target_embeddings, self.reference_embeddings)

        results = []
        for i, similarities in enumerate(all_similarities):
            # Get top-K indices
            top_indices = np.argsort(similarities)[::-1][:k]

            # Filter by minimum similarity
            sample_results = []
            for idx in top_indices:
                if similarities[idx] >= min_similarity:
                    sample_results.append((self.reference_texts[idx], float(similarities[idx])))

            results.append(sample_results)

        return results

    def get_statistics(self) -> Dict:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with statistics
        """
        if self.reference_embeddings is None:
            return {"samples": 0, "embedding_dim": 0}

        return {
            "samples": len(self.reference_texts),
            "embedding_dim": self.reference_embeddings.shape[1],
            "mean_embedding": self.reference_embeddings.mean(axis=0).tolist()[:5]  # First 5 dims for preview
        }


# ==================== Utility Functions ====================

def load_test_set(test_file: str) -> List[Dict[str, str]]:
    """
    Load test set from .txt or .json file.

    Args:
        test_file: Path to test file

    Returns:
        List of dicts with 'id', 'text', and optionally 'label'
    """
    file_ext = Path(test_file).suffix.lower()

    if file_ext == '.txt':
        samples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    # Try to parse as JSON if possible
                    if line.startswith('{'):
                        try:
                            data = json.loads(line)
                            samples.append(data)
                        except:
                            samples.append({"id": f"test_{i}", "text": line})
                    else:
                        samples.append({"id": f"test_{i}", "text": line})
        return samples

    elif file_ext == '.json':
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [{"id": k, "text": v} for k, v in data.items()]

    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .txt or .json")


def save_retrieval_results(results: List[Dict], output_file: str):
    """
    Save retrieval results to a JSON file.

    Args:
        results: List of retrieval results
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ==================== Complete Retrieval Pipeline ====================

class RetrievalPipeline:
    """
    Complete retrieval pipeline for MDMA framework.
    Handles loading knowledge base, test set, and retrieving similar samples.
    """

    def __init__(self,
                 encoder_model_name: str = "roberta-base",
                 device: str = "cpu",
                 cache_dir: str = "./embeddings_cache"):
        """
        Initialize the retrieval pipeline.

        Args:
            encoder_model_name: Name of the pre-trained model for text encoding
            device: Device to run the model on
            cache_dir: Directory to cache embeddings
        """
        self.retriever = SimilarityRetriever(
            encoder_model_name=encoder_model_name,
            device=device,
            cache_dir=cache_dir
        )

    def build_from_directory(self,
                             knowledge_base_dir: str,
                             file_pattern: str = "*.txt") -> int:
        """
        Build knowledge base from all files in a directory.

        Args:
            knowledge_base_dir: Directory containing knowledge base files
            file_pattern: Pattern to match files

        Returns:
            Total number of samples in knowledge base
        """
        all_texts = []

        # Collect all matching files
        for file_path in Path(knowledge_base_dir).glob(file_pattern):
            print(f"Loading: {file_path}")
            texts = self.retriever._load_txt_file(str(file_path))
            all_texts.extend(texts)

        if not all_texts:
            raise ValueError(f"No files found matching pattern: {file_pattern}")

        # Create temporary JSON file for building
        temp_file = os.path.join(knowledge_base_dir, "_combined_knowledge.json")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(all_texts, f, ensure_ascii=False)

        # Build knowledge base
        count = self.retriever.build_knowledge_base(temp_file, save_embeddings=True)

        # Clean up temp file
        # os.remove(temp_file)

        return count

    def process_test_set(self,
                         test_file: str,
                         k: int = 4,
                         output_file: Optional[str] = None) -> List[Dict]:
        """
        Process a test set and retrieve similar samples for each test sample.

        Args:
            test_file: Path to test file (.txt or .json)
            k: Number of similar samples to retrieve
            output_file: Optional path to save results

        Returns:
            List of results with similar samples
        """
        # Load test set
        test_samples = load_test_set(test_file)
        print(f"Loaded {len(test_samples)} test samples")

        # Extract target texts
        target_texts = [sample['text'] for sample in test_samples]

        # Retrieve similar samples
        print(f"Retrieving top-{k} similar samples for {len(target_texts)} targets...")
        all_results = self.retriever.batch_retrieve(target_texts, k=k)

        # Format results
        formatted_results = []
        for i, sample in enumerate(test_samples):
            result = {
                "id": sample.get('id', f"test_{i}"),
                "text": sample['text'],
                "label": sample.get('label', None),
                "similar_samples": []
            }

            for similar_text, similarity in all_results[i]:
                result["similar_samples"].append({
                    "text": similar_text,
                    "similarity": similarity
                })

            formatted_results.append(result)

        # Save results if output_file is provided
        if output_file:
            save_retrieval_results(formatted_results, output_file)
            print(f"Results saved to: {output_file}")

        return formatted_results
