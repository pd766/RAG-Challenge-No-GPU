import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices.
        
        Args:
            all_reports_dir (Path): Directory containing the JSON report files
            output_dir (Path): Directory where to save the BM25 indices
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # Load the report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # Extract text chunks and create BM25 index
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # Save BM25 index
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

class VectorDBIngestor:
    def __init__(self, model_name: str = "text-embedding-v3"):
        """
        Initialize VectorDBIngestor with Alibaba Cloud Qwen Embedding API.
        
        Args:
            model_name: Name of the Qwen embedding model to use.
                       Options: 'text-embedding-v1', 'text-embedding-v2', 'text-embedding-v3'
                       Default is text-embedding-v3 (最新版本，支持中文)
        """
        load_dotenv()
        self.model_name = model_name
        
        # 从环境变量获取阿里云 API Key
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "请在 .env 文件中设置 DASHSCOPE_API_KEY 环境变量\n"
                "获取方式：https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen"
            )
        
        # 使用 OpenAI 兼容接口，指向阿里云通义千问的端点
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60.0,
            max_retries=2
        )
        print(f"已初始化阿里云通义千问 Embedding API，模型: {model_name}")

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(3))
    def _get_embeddings(self, text: Union[str, List[str]], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for text using Alibaba Cloud Qwen Embedding API.
        
        Args:
            text: Single string or list of strings to embed
            batch_size: Batch size for API calls (阿里云限制每次最多10条)
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if isinstance(text, str):
            if not text.strip():
                raise ValueError("Input text cannot be an empty string.")
            text = [text]
        
        # Filter out empty strings
        text = [t.strip() for t in text if t.strip()]
        if not text:
            raise ValueError("All input texts are empty.")
        
        # 分批调用 API（阿里云限制每次最多10条）
        all_embeddings = []
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                
                # 提取 embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"批次 {i//batch_size + 1} 调用 API 失败: {e}")
                raise
        
        return all_embeddings

    def _create_vector_db(self, embeddings: List[float]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")