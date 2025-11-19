"""임베딩 서비스 - 다양한 모델 지원"""

import logging

from ..core.enums import EmbeddingModel
from ..core.models import CodeChunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    임베딩 생성 서비스

    지원 모델:
    - Mistral Codestral Embed (코드 특화 최고)
    - OpenAI text-embedding-3-small/large
    - Sentence Transformers (로컬)
    """

    def __init__(
        self,
        model: EmbeddingModel,
        api_key: str | None = None,
        api_base: str | None = None,
        dimension: int | None = None,
    ):
        """
        Args:
            model: 임베딩 모델
            api_key: API 키 (Mistral, OpenAI 등)
            api_base: API 베이스 URL (Mistral 전용)
            dimension: 벡터 차원 (None이면 모델 기본값)
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base or "https://api.mistral.ai/v1"
        self.dimension = dimension

        # 모델별 초기화
        self._initialize_model()

    def _initialize_model(self):
        """모델별 클라이언트 초기화"""
        if self.model == EmbeddingModel.CODESTRAL_EMBED:
            # Mistral API
            self._init_mistral()
        elif self.model in (
            EmbeddingModel.OPENAI_3_SMALL,
            EmbeddingModel.OPENAI_3_LARGE,
            EmbeddingModel.OPENAI_ADA_002,
        ):
            # OpenAI API
            self._init_openai()
        elif self.model in (EmbeddingModel.ALL_MINI_LM_L6_V2, EmbeddingModel.ALL_MPNET_BASE_V2):
            # Sentence Transformers (로컬)
            self._init_sentence_transformer()
        elif self.model == EmbeddingModel.CODEBERT_BASE:
            # CodeBERT (로컬)
            self._init_codebert()
        else:
            raise ValueError(f"Unsupported embedding model: {self.model}")

    def _init_mistral(self):
        """Mistral Codestral Embed 초기화"""
        if not self.api_key:
            raise ValueError("Mistral API key required for Codestral Embed")

        try:
            from mistralai import Mistral

            self.client = Mistral(api_key=self.api_key)
            self.embed_func = self._embed_mistral
            logger.info("Initialized Mistral Codestral Embed")
        except ImportError as e:
            raise ImportError("Please install: pip install mistralai") from e

    def _init_openai(self):
        """OpenAI 초기화"""
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
            self.embed_func = self._embed_openai
            logger.info(f"Initialized OpenAI {self.model.value}")
        except ImportError as e:
            raise ImportError("Please install: pip install openai") from e

    def _init_sentence_transformer(self):
        """Sentence Transformer 초기화"""
        try:
            from sentence_transformers import SentenceTransformer

            self.st_model = SentenceTransformer(self.model.value)
            self.embed_func = self._embed_sentence_transformer
            logger.info(f"Initialized Sentence Transformer {self.model.value}")
        except ImportError as e:
            raise ImportError("Please install: pip install sentence-transformers") from e

    def _init_codebert(self):
        """CodeBERT 초기화"""
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model.value)
            self.codebert_model = AutoModel.from_pretrained(self.model.value)
            self.embed_func = self._embed_codebert
            logger.info(f"Initialized CodeBERT {self.model.value}")
        except ImportError as e:
            raise ImportError("Please install: pip install transformers torch") from e

    def embed_text(self, text: str) -> list[float]:
        """
        단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            벡터 (List[float])
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        배치 텍스트 임베딩

        Args:
            texts: 텍스트 리스트

        Returns:
            벡터 리스트
        """
        if not texts:
            return []

        return self.embed_func(texts)

    def embed_chunk(self, chunk: CodeChunk) -> list[float]:
        """단일 청크 임베딩"""
        text = self._prepare_chunk_text(chunk)
        return self.embed_text(text)

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """배치 청크 임베딩"""
        texts = [self._prepare_chunk_text(chunk) for chunk in chunks]
        return self.embed_texts(texts)

    def _prepare_chunk_text(self, chunk: CodeChunk) -> str:
        """
        청크를 임베딩 텍스트로 변환

        전략: 코드 + docstring (권장)
        """
        parts = [chunk.text]

        # Docstring 추가
        if chunk.attrs.get("docstring"):
            parts.append(f"# Purpose: {chunk.attrs['docstring']}")

        return "\n".join(parts)

    # === 모델별 임베딩 함수 ===

    def _embed_mistral(self, texts: list[str]) -> list[list[float]]:
        """
        Mistral Codestral Embed API 호출

        Args:
            texts: 텍스트 리스트

        Returns:
            벡터 리스트
        """
        # Mistral 최대 토큰 제한
        MAX_TOKENS_PER_TEXT = 8192  # 개별 텍스트 최대 토큰
        MAX_TOKENS_PER_BATCH = 16384  # 배치 전체 최대 토큰 (안전 마진 포함)

        # 토큰 수 추정 및 텍스트 자르기
        processed_texts = []
        for text in texts:
            # 간단한 토큰 추정: 1토큰 ≈ 4글자
            estimated_tokens = len(text) // 4

            if estimated_tokens > MAX_TOKENS_PER_TEXT:
                # 최대 토큰에 맞게 텍스트 자르기
                max_chars = MAX_TOKENS_PER_TEXT * 4
                truncated = text[:max_chars]
                logger.warning(f"Text truncated from {estimated_tokens} to {MAX_TOKENS_PER_TEXT} tokens")
                processed_texts.append(truncated)
            else:
                processed_texts.append(text)

        # 배치 전체 토큰 수 확인 및 분할
        all_vectors = []
        current_batch = []
        current_batch_tokens = 0

        for text in processed_texts:
            text_tokens = len(text) // 4
            
            # 현재 배치에 추가하면 제한 초과하는 경우
            if current_batch and (current_batch_tokens + text_tokens > MAX_TOKENS_PER_BATCH):
                # 현재 배치 처리
                try:
                    response = self.client.embeddings.create(
                        model=self.model.value,
                        inputs=current_batch,
                    )
                    batch_vectors = [data.embedding for data in response.data]
                    all_vectors.extend(batch_vectors)
                    logger.debug(f"Mistral embedded {len(current_batch)} texts ({current_batch_tokens} tokens)")
                except Exception as e:
                    logger.error(f"Mistral embedding failed: {e}")
                    raise
                
                # 새 배치 시작
                current_batch = [text]
                current_batch_tokens = text_tokens
            else:
                current_batch.append(text)
                current_batch_tokens += text_tokens

        # 마지막 배치 처리
        if current_batch:
            try:
                response = self.client.embeddings.create(
                    model=self.model.value,
                    inputs=current_batch,
                )
                batch_vectors = [data.embedding for data in response.data]
                all_vectors.extend(batch_vectors)
                logger.debug(f"Mistral embedded {len(current_batch)} texts ({current_batch_tokens} tokens)")
            except Exception as e:
                logger.error(f"Mistral embedding failed: {e}")
                raise

        return all_vectors

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """
        OpenAI API 호출

        Args:
            texts: 텍스트 리스트

        Returns:
            벡터 리스트
        """
        # OpenAI 최대 토큰 제한: 8,191 (모델별로 다름)
        MAX_TOKENS = 8191

        # 토큰 수 추정 및 텍스트 자르기
        processed_texts = []
        for text in texts:
            # 간단한 토큰 추정: 1토큰 ≈ 4글자
            estimated_tokens = len(text) // 4

            if estimated_tokens > MAX_TOKENS:
                # 최대 토큰에 맞게 텍스트 자르기
                max_chars = MAX_TOKENS * 4
                truncated = text[:max_chars]
                logger.warning(f"Text truncated from {estimated_tokens} to {MAX_TOKENS} tokens")
                processed_texts.append(truncated)
            else:
                processed_texts.append(text)

        try:
            response = self.client.embeddings.create(
                model=self.model.value,
                input=processed_texts,  # OpenAI는 'input' 사용
                dimensions=self.dimension,  # 선택적 차원 축소
            )

            vectors = [data.embedding for data in response.data]
            logger.debug(f"OpenAI embedded {len(texts)} texts")
            return vectors

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    def _embed_sentence_transformer(self, texts: list[str]) -> list[list[float]]:
        """
        Sentence Transformer (로컬)

        Args:
            texts: 텍스트 리스트

        Returns:
            벡터 리스트
        """
        try:
            vectors = self.st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            logger.debug(f"Sentence Transformer embedded {len(texts)} texts")
            result: list[list[float]] = vectors.tolist()
            return result

        except Exception as e:
            logger.error(f"Sentence Transformer embedding failed: {e}")
            raise

    def _embed_codebert(self, texts: list[str]) -> list[list[float]]:
        """
        CodeBERT (로컬)

        Args:
            texts: 텍스트 리스트

        Returns:
            벡터 리스트
        """
        try:
            import torch

            vectors = []
            for text in texts:
                # 토큰화
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512, padding=True
                )

                # 임베딩 생성
                with torch.no_grad():
                    outputs = self.codebert_model(**inputs)
                    # Mean pooling
                    vector = outputs.last_hidden_state.mean(dim=1)[0]
                    vectors.append(vector.cpu().numpy().tolist())

            logger.debug(f"CodeBERT embedded {len(texts)} texts")
            return vectors

        except Exception as e:
            logger.error(f"CodeBERT embedding failed: {e}")
            raise

    def get_dimension(self) -> int:
        """
        모델의 벡터 차원 반환

        Returns:
            벡터 차원
        """
        if self.dimension:
            return self.dimension

        # 모델별 기본 차원
        dimension_map = {
            EmbeddingModel.CODESTRAL_EMBED: 1536,  # 실제 API 응답 차원
            EmbeddingModel.OPENAI_3_SMALL: 1536,
            EmbeddingModel.OPENAI_3_LARGE: 3072,
            EmbeddingModel.OPENAI_ADA_002: 1536,
            EmbeddingModel.ALL_MINI_LM_L6_V2: 384,
            EmbeddingModel.ALL_MPNET_BASE_V2: 768,
            EmbeddingModel.CODEBERT_BASE: 768,
            EmbeddingModel.COHERE_V3: 1024,
        }

        return dimension_map.get(self.model, 768)
