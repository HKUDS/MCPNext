import re
import numpy as np
from typing import Iterable, List, Tuple, Dict, Optional, Any
from enum import Enum
import json
import pickle
from pathlib import Path
from datetime import datetime

from .tool import BaseTool
from .types import BackendType
from mcpnext.llm import LLMClient
from mcpnext.utils.logging import Logger

logger = Logger.get_logger(__name__)


class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class ToolRanker:
    """
    ToolRanker: rank tools by keyword, semantic or hybrid
    """
    # Cache version for persistent storage - increment when cache format changes
    CACHE_VERSION = 1
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        enable_cache_persistence: bool = False
    ):
        """Initialize ToolRanker.
        
        Args:
            model_name: Embedding model name. If None, will use the value from config.
            cache_dir: Directory to store persistent embedding cache. If None, uses ~/.mcpnext/embedding_cache
            enable_cache_persistence: Whether to persist embeddings to disk. Default: False (memory only)
        """
        if model_name is None:
            try:
                from mcpnext.config import get_config
                config = get_config()
                model_name = config.tool_search.embedding_model
            except Exception as exc:
                logger.warning(f"Failed to load config, using default model: {exc}")
                model_name = "BAAI/bge-small-en-v1.5"
        
        self._model_name = model_name
        self._embed_model = None  # lazy load
        self._embedding_fn = None
        
        # Persistent cache settings
        self._enable_cache_persistence = enable_cache_persistence
        if cache_dir is None:
            cache_dir = Path.home() / ".mcpnext" / "embedding_cache"
        self._cache_dir = Path(cache_dir)
        
        # Log cache settings
        logger.info(
            f"ToolRanker initialized: enable_cache_persistence={enable_cache_persistence}, "
            f"cache_dir={self._cache_dir}"
        )
        
        # Structured in-memory cache
        # Structure: {backend: {server: {tool_name: {"embedding": np.ndarray, "description": str, "cached_at": str}}}}
        self._structured_cache: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        
        # For backward compatibility and quick lookup: {text -> (backend, server, tool_name)}
        self._text_to_key: Dict[str, Tuple[str, str, str]] = {}
        
        # Load persistent cache if enabled
        if self._enable_cache_persistence:
            logger.info(f"Loading persistent cache from {self._cache_dir}")
            self._load_persistent_cache()
    
    def _get_cache_key(self, tool: BaseTool) -> Tuple[str, str, str]:
        """Get structured cache key (backend, server, tool_name) from tool."""
        if tool.is_bound:
            backend = tool.runtime_info.backend.value
            server = tool.runtime_info.server_name or "default"
        else:
            if not tool.backend_type or tool.backend_type == BackendType.NOT_SET:
                backend = "UNKNOWN"
            else:
                backend = tool.backend_type.value
            server = "default"
        
        return (backend, server, tool.name)
    
    def _get_cache_file_path(self) -> Path:
        """Get the cache file path for the current model."""
        # Use model name in filename to support multiple models
        safe_model_name = self._model_name.replace("/", "_").replace("\\", "_")
        return self._cache_dir / f"embeddings_{safe_model_name}_v{self.CACHE_VERSION}.pkl"
    
    def _load_persistent_cache(self) -> None:
        """Load embeddings from disk cache."""
        cache_file = self._get_cache_file_path()
        
        if not cache_file.exists():
            logger.debug(f"No persistent cache found at {cache_file}")
            return
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Validate cache version
            if isinstance(data, dict) and data.get("version") == self.CACHE_VERSION:
                self._structured_cache = data.get("embeddings", {})
                self._rebuild_text_index()
                
                # Count total embeddings
                total = sum(
                    len(tools) 
                    for backend in self._structured_cache.values() 
                    for tools in backend.values()
                )
                logger.info(f"Loaded {total} embeddings from cache: {cache_file}")
            else:
                logger.warning(f"Cache version mismatch or invalid format, starting fresh")
                self._structured_cache = {}
        except Exception as exc:
            logger.warning(f"Failed to load persistent cache: {exc}")
            self._structured_cache = {}
    
    def _rebuild_text_index(self) -> None:
        """Rebuild text-to-key mapping for quick lookup."""
        self._text_to_key.clear()
        for backend, servers in self._structured_cache.items():
            for server, tools in servers.items():
                for tool_name, tool_data in tools.items():
                    desc = tool_data.get("description", "")
                    text = f"{tool_name}: {desc}"
                    self._text_to_key[text] = (backend, server, tool_name)
    
    def _save_persistent_cache(self) -> None:
        """Save embeddings to disk cache."""
        if not self._enable_cache_persistence or not self._structured_cache:
            return
        
        cache_file = self._get_cache_file_path()
        
        try:
            # Create directory if it doesn't exist
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Build cache data with metadata
            cache_data = {
                "version": self.CACHE_VERSION,
                "model_name": self._model_name,
                "last_updated": datetime.now().isoformat(),
                "embeddings": self._structured_cache
            }
            
            # Save cache
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Count total embeddings
            total = sum(
                len(tools) 
                for backend in self._structured_cache.values() 
                for tools in backend.values()
            )
            logger.debug(f"Saved {total} embeddings to cache: {cache_file}")
        except Exception as exc:
            logger.warning(f"Failed to save persistent cache: {exc}")

    def rank(
        self,
        query: str,
        tools: List[BaseTool],
        *,
        top_k: int = 50,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> List[Tuple[BaseTool, float]]:
        if mode == SearchMode.KEYWORD:
            return self._keyword_search(query, tools, top_k)
        if mode == SearchMode.SEMANTIC:
            return self._semantic_search(query, tools, top_k)
        # hybrid
        return self._hybrid_search(query, tools, top_k)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.split(r"[^\w]+", text.lower())
        tokens = [tok for tok in tokens if tok]
        return tokens

    def _keyword_search(
        self, query: str, tools: Iterable[BaseTool], top_k: int
    ) -> List[Tuple[BaseTool, float]]:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError:
            BM25Okapi = None  # fallback below

        tool_list = list(tools)
        if not tool_list:
            return []
        
        corpus_tokens: list[list[str]] = [self._tokenize(f"{t.name} {t.description}") for t in tool_list]
        query_tokens = self._tokenize(query)

        if BM25Okapi and corpus_tokens:
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
            scored = [(t, float(s)) for t, s in zip(tool_list, scores, strict=True)]
        else:
            # fallback: simple term overlap ratio
            q_set = set(query_tokens)
            scored = []
            for t, toks in zip(tool_list, corpus_tokens, strict=True):
                if not toks:
                    scored.append((t, 0.0))  # Include tool with 0 score
                    continue
                overlap = q_set.intersection(toks)
                score = len(overlap) / len(q_set) if len(q_set) > 0 else 0.0
                scored.append((t, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        result = scored[:top_k]
        
        # If no matches found (all scores are 0), return all tools
        if not result or all(score == 0.0 for _, score in result):
            logger.debug(f"Keyword search found no matches, returning all {len(tool_list)} tools")
            return [(t, 0.0) for t in tool_list]
        
        return result

    def _ensure_model(self) -> bool:
        if self._embed_model:
            return True
        try:
            from fastembed import TextEmbedding 
            logger.debug(f"fastembed imported successfully, loading model: {self._model_name}")
        except ImportError as e:
            logger.warning(
                f"fastembed not installed (ImportError: {e}), semantic search unavailable. "
                f"Install with: pip install fastembed"
            )
            return False
        
        try:
            logger.info(f"Loading embedding model: {self._model_name}...")
            self._embed_model = TextEmbedding(model_name=self._model_name)
            self._embedding_fn = lambda txts: list(self._embed_model.embed(txts))
            logger.info(f"Embedding model '{self._model_name}' loaded successfully")
            return True
        except Exception as exc:
            logger.error(f"Embedding model '{self._model_name}' loading failed: {exc}")
            return False

    def _get_embedding(self, tool: BaseTool) -> Optional[np.ndarray]:
        """Get embedding from structured cache."""
        backend, server, tool_name = self._get_cache_key(tool)
        
        if backend not in self._structured_cache:
            return None
        if server not in self._structured_cache[backend]:
            return None
        if tool_name not in self._structured_cache[backend][server]:
            return None
        
        return self._structured_cache[backend][server][tool_name].get("embedding")
    
    def _set_embedding(self, tool: BaseTool, embedding: np.ndarray) -> None:
        """Store embedding in structured cache."""
        backend, server, tool_name = self._get_cache_key(tool)
        
        # Initialize nested structure if needed
        if backend not in self._structured_cache:
            self._structured_cache[backend] = {}
        if server not in self._structured_cache[backend]:
            self._structured_cache[backend][server] = {}
        
        # Store embedding with metadata
        self._structured_cache[backend][server][tool_name] = {
            "embedding": embedding,
            "description": tool.description or "",
            "cached_at": datetime.now().isoformat()
        }
        
        # Update text index for backward compatibility
        text = f"{tool.name}: {tool.description}"
        self._text_to_key[text] = (backend, server, tool_name)
    
    def _semantic_search(
        self, query: str, tools: Iterable[BaseTool], top_k: int
    ) -> List[Tuple[BaseTool, float]]:
        if not self._ensure_model():
            logger.debug("Semantic search unavailable, returning empty list")
            return []
        
        tools_list = list(tools)
        
        # Collect embeddings with cache reuse
        missing_tools = [t for t in tools_list if self._get_embedding(t) is None]
        cache_updated = False
        
        if missing_tools:
            try:
                # Generate embeddings for missing tools
                missing_texts = [f"{t.name}: {t.description}" for t in missing_tools]
                new_embs = self._embedding_fn(missing_texts)
                
                for tool, emb in zip(missing_tools, new_embs, strict=True):
                    self._set_embedding(tool, emb)
                
                cache_updated = True
                logger.debug(f"Computed embeddings for {len(missing_tools)} new tools")
            except Exception as exc:
                logger.error("Failed to generate embeddings: %s", exc)
                return []
        
        # Save to persistent cache if updated
        if cache_updated:
            self._save_persistent_cache()

        try:
            q_emb = self._embedding_fn([query])[0]
        except Exception as exc:
            logger.error("Failed to embed query: %s", exc)
            return []

        scored: list[tuple[BaseTool, float]] = []
        for t in tools_list:
            emb = self._get_embedding(t)
            if emb is None:
                # Should not happen, but handle gracefully
                logger.warning(f"No embedding found for tool: {t.name}")
                scored.append((t, 0.0))
                continue
            
            # Calculate cosine similarity with zero-division protection
            q_norm = np.linalg.norm(q_emb)
            emb_norm = np.linalg.norm(emb)
            if q_norm == 0 or emb_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q_emb, emb) / (q_norm * emb_norm))
            scored.append((t, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _hybrid_search(
        self, query: str, tools: Iterable[BaseTool], top_k: int
    ) -> List[Tuple[BaseTool, float]]:
        # keyword filter
        kw_top = self._keyword_search(query, tools, top_k * 3)
        if not kw_top:
            # No keyword matches, try semantic search
            semantic_results = self._semantic_search(query, tools, top_k)
            if semantic_results:
                return semantic_results
            # Both failed, return top N tools
            logger.warning("Both keyword and semantic search failed, returning top N tools")
            return [(t, 0.0) for t in list(tools)[:top_k]]
        
        # semantic ranking on keyword results
        semantic_results = self._semantic_search(query, [t for t, _ in kw_top], top_k)
        if semantic_results:
            return semantic_results
        
        # Semantic unavailable, return keyword results
        logger.debug("Semantic search unavailable, using keyword results only")
        return kw_top[:top_k]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache.
        
        Returns:
            Dict with structure: {
                "total_embeddings": int,
                "backends": {
                    "backend_name": {
                        "total": int,
                        "servers": {
                            "server_name": int  # count of tools
                        }
                    }
                }
            }
        """
        stats = {
            "total_embeddings": 0,
            "backends": {}
        }
        
        for backend, servers in self._structured_cache.items():
            backend_total = 0
            server_stats = {}
            
            for server, tools in servers.items():
                tool_count = len(tools)
                backend_total += tool_count
                server_stats[server] = tool_count
            
            stats["backends"][backend] = {
                "total": backend_total,
                "servers": server_stats
            }
            stats["total_embeddings"] += backend_total
        
        return stats
    
    def clear_cache(self, backend: Optional[str] = None, server: Optional[str] = None) -> int:
        """Clear embeddings from cache.
        
        Args:
            backend: If provided, only clear this backend. If None, clear all.
            server: If provided (and backend is provided), only clear this server.
        
        Returns:
            Number of embeddings cleared.
        """
        cleared_count = 0
        
        if backend is None:
            # Clear everything
            for b in self._structured_cache.values():
                for s in b.values():
                    cleared_count += len(s)
            self._structured_cache.clear()
            self._text_to_key.clear()
        elif server is None:
            # Clear specific backend
            if backend in self._structured_cache:
                for s in self._structured_cache[backend].values():
                    cleared_count += len(s)
                del self._structured_cache[backend]
                # Rebuild text index
                self._rebuild_text_index()
        else:
            # Clear specific backend+server
            if backend in self._structured_cache and server in self._structured_cache[backend]:
                cleared_count = len(self._structured_cache[backend][server])
                del self._structured_cache[backend][server]
                # Clean up empty backend
                if not self._structured_cache[backend]:
                    del self._structured_cache[backend]
                # Rebuild text index
                self._rebuild_text_index()
        
        # Save after clearing
        if cleared_count > 0 and self._enable_cache_persistence:
            self._save_persistent_cache()
            logger.info(f"Cleared {cleared_count} embeddings from cache")
        
        return cleared_count


class SearchCoordinator(BaseTool):
    _name = "_filter_tools"
    _description = "Internal helper: filter & rank tools from a given list."

    def __init__(
        self,
        *,
        max_tools: Optional[int] = None,
        llm: LLMClient = LLMClient(),
        enable_llm_filter: Optional[bool] = None,
        llm_filter_threshold: Optional[int] = None,
        enable_cache_persistence: Optional[bool] = None,
        cache_dir: Optional[str | Path] = None,
    ):
        """Create a SearchCoordinator.

        Args:
            max_tools: max number of tools to return. If None, will use the value from config.
            llm: optional async LLM, used to filter backend/server first
            enable_llm_filter: whether to use LLM to pre-filter by backend/server. 
                If None, uses config value.
            llm_filter_threshold: only apply LLM filter when tool count > this threshold.
                If None, always apply (when enabled).
            enable_cache_persistence: whether to persist embeddings to disk. If None, uses config value.
            cache_dir: directory to store persistent embedding cache. If None, uses config value or default.
        """
        super().__init__()
        
        # Load configuration with fallback to defaults
        try:
            from mcpnext.config import get_config
            config = get_config()
            
            # Read from config if not provided
            max_tools = max_tools if max_tools is not None else config.tool_search.max_tools
            enable_llm_filter = enable_llm_filter if enable_llm_filter is not None else config.tool_search.enable_llm_filter
            llm_filter_threshold = llm_filter_threshold if llm_filter_threshold is not None else getattr(config.tool_search, 'llm_filter_threshold', 50)
            enable_cache_persistence = enable_cache_persistence if enable_cache_persistence is not None else getattr(config.tool_search, 'enable_cache_persistence', False)
            cache_dir = cache_dir if cache_dir is not None else getattr(config.tool_search, 'cache_dir', None)
            self._default_mode = config.tool_search.search_mode
        except Exception as exc:
            logger.warning(f"Failed to load config, using defaults: {exc}")
            max_tools = max_tools if max_tools is not None else 20
            enable_llm_filter = enable_llm_filter if enable_llm_filter is not None else True
            llm_filter_threshold = llm_filter_threshold if llm_filter_threshold is not None else 50
            enable_cache_persistence = enable_cache_persistence if enable_cache_persistence is not None else False
            self._default_mode = "hybrid"
        
        self.max_tools = max_tools
        
        # Log cache settings for debugging
        logger.info(
            f"SearchCoordinator initialized with cache settings: "
            f"enable_cache_persistence={enable_cache_persistence}, cache_dir={cache_dir}"
        )
        
        self._ranker = ToolRanker(
            enable_cache_persistence=enable_cache_persistence,
            cache_dir=cache_dir
        )
        self._llm: LLMClient = llm
        
        # LLM filter settings
        self._enable_llm_filter = enable_llm_filter
        self._llm_filter_threshold = llm_filter_threshold

    async def _arun(
        self,
        task_prompt: str,
        candidate_tools: list[BaseTool],
        *,
        max_tools: int | None = None,
        mode: str | None = None, # "semantic" | "keyword" | "hybrid"
    ) -> list[BaseTool]:
        max_tools = self.max_tools if max_tools is None else max_tools
        mode = self._default_mode if mode is None else mode

        cache_key = (id(candidate_tools), task_prompt, mode, max_tools)
        if not hasattr(self, "_query_cache"):
            self._query_cache: Dict[tuple, list[BaseTool]] = {}
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        if len(candidate_tools) <= max_tools:
            self._query_cache[cache_key] = candidate_tools
            return candidate_tools

        # Decide whether to use LLM filter based on tool count and threshold
        tools_count = len(candidate_tools)
        should_use_llm_filter = (
            self._llm and 
            self._enable_llm_filter and 
            tools_count > self._llm_filter_threshold
        )
        
        if should_use_llm_filter:
            logger.info(
                f"Tool count ({tools_count}) > threshold ({self._llm_filter_threshold}), "
                f"applying LLM pre-filter..."
            )
            shortlist = await self._llm_filter(task_prompt, candidate_tools)
            
            # If LLM filter returned empty list, use all candidates
            if not shortlist:
                logger.warning("LLM filter returned empty list, using all candidate tools")
                shortlist = candidate_tools
        else:
            if not self._enable_llm_filter:
                logger.debug("LLM filter disabled, using all candidate tools")
            else:
                logger.debug(
                    f"Tool count ({tools_count}) <= threshold ({self._llm_filter_threshold}), "
                    f"skipping LLM filter"
                )
            shortlist = candidate_tools

        try:
            ranked = self._ranker.rank(task_prompt, shortlist, top_k=max_tools, mode=SearchMode(mode))
            result = [t for t, _ in ranked]
        except Exception as exc:
            import traceback
            logger.warning("Ranking failed (%s), falling back to keyword search", exc)
            logger.debug("Ranking error traceback:\n%s", traceback.format_exc())
            try:
                result = [t for t, _ in self._ranker._keyword_search(task_prompt, shortlist, max_tools)]
            except Exception as fallback_exc:
                logger.error("Keyword search also failed (%s), returning top N tools", fallback_exc)
                logger.debug("Keyword search error traceback:\n%s", traceback.format_exc())
                result = shortlist[:max_tools]

        # Log search results for user visibility
        self._log_search_results(candidate_tools, result, mode)
        
        self._query_cache[cache_key] = result
        return result

    async def _llm_filter(self, prompt: str, tools: list[BaseTool]) -> list[BaseTool]:
        """
        Let LLM judge which backend/server based on actual tool capabilities.
        Shows representative tool examples to help LLM understand what each server does.
        """
        # Group tools by backend and server
        from collections import defaultdict
        backend_server_tools: Dict[str, Dict[str | None, list[BaseTool]]] = defaultdict(lambda: defaultdict(list))
        
        for t in tools:
            # Get backend and server info
            if t.is_bound:
                backend = t.runtime_info.backend.value
                server = t.runtime_info.server_name if backend.lower() == "mcp" else None
            else:
                if not t.backend_type or t.backend_type == BackendType.NOT_SET:
                    logger.warning(f"Tool {t.name} has no backend info, skipping in LLM filter")
                    continue
                backend = t.backend_type.value
                server = None
            
            backend_server_tools[backend][server].append(t)

        # Build information block with all tool names and some descriptions
        lines: list[str] = ["Available backends and their tools:"]
        lines.append("")
        
        for backend, srv_map in backend_server_tools.items():
            total = sum(len(tool_list) for tool_list in srv_map.values())
            lines.append(f"## {backend} Backend ({total} tools)")

            if backend.lower() == "mcp":
                # For MCP, show each server with all tool names and some descriptions
                for srv, tool_list in srv_map.items():
                    srv_display = srv or "<default>"
                    lines.append(f"\n### Server: {srv_display} ({len(tool_list)} tools)")
                    
                    # Show all tool names
                    tool_names = [t.name for t in tool_list]
                    lines.append(f"  All tools: {', '.join(tool_names)}")
                    
                    # Show up to 5 tool descriptions as examples
                    if tool_list:
                        lines.append(f"  Example capabilities:")
                        examples = tool_list[:5]
                        for tool in examples:
                            tool_desc = tool.description or "No description"
                            # Truncate long descriptions
                            if len(tool_desc) > 100:
                                tool_desc = tool_desc[:97] + "..."
                            lines.append(f"    - {tool.name}: {tool_desc}")
            else:
                # For non-MCP backends, show all tool names and some descriptions
                for srv, tool_list in srv_map.items():
                    tool_names = [t.name for t in tool_list]
                    lines.append(f"  All tools: {', '.join(tool_names)}")
                    
                    # Show up to 5 tool descriptions as examples
                    if tool_list:
                        lines.append(f"  Example capabilities:")
                        examples = tool_list[:5]
                        for tool in examples:
                            tool_desc = tool.description or "No description"
                            if len(tool_desc) > 100:
                                tool_desc = tool_desc[:97] + "..."
                            lines.append(f"    - {tool.name}: {tool_desc}")
            
            lines.append("")

        capabilities_block = "\n".join(lines)

        # Build conversation history with system prompt
        TOOL_FILTER_SYSTEM_PROMPT = f"""You are an expert tool selection assistant.

# Your task
Analyze the given task and determine which backend(s) and server(s) provide the capabilities needed.

# Important guidelines
- **Focus on tool names and capabilities**: Carefully examine the tool names listed below to understand what each backend/server can do
- **Be inclusive**: If a backend/server has tools that might be relevant, include it
- **When in doubt, include it**: It's better to include a backend/server than miss relevant tools
- **Consider tool names as hints**: Tool names often indicate their functionality (e.g., "create_file", "search_web", "read_calendar")

{capabilities_block}

# Output format
Return ONLY a JSON array (no markdown, no explanation):
[{{"backend": "...", "server": "..."}}, ...]

For non-MCP backends, set "server" to null.
If unsure, include all backends that might be relevant."""

        user_query = f"Task: {prompt}\n\nBased on the task above, which backend(s) and server(s) should be used?"

        conversation_history = [
            {"role": "system", "content": TOOL_FILTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        # Format conversation and call LLM
        messages_text = LLMClient.format_messages_to_text(conversation_history)
        resp = await self._llm.complete(messages_text)
        
        content = resp["message"]["content"].strip()
        
        # Try to extract JSON from markdown code block or plain text
        # regex: match ``` code block or directly find JSON array/object
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(code_block_pattern, content, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            # if no code block, try to directly extract JSON array
            json_pattern = r'\[.*\]'
            match = re.search(json_pattern, content, re.DOTALL)
            if match:
                content = match.group(0)
        
        try:
            choices = json.loads(content)
            if not isinstance(choices, list):
                logger.warning(f"Expected list but got {type(choices)}")
                return tools
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {content[:200]}... Error: {e}")
            return tools  # return all tools if parsing failed
        
        # Filter tools matching any of the chosen backend/server combinations
        result = []
        for t in tools:
            if t.is_bound:
                t_backend = t.runtime_info.backend.value
                t_server = t.runtime_info.server_name
            else:
                if not t.backend_type or t.backend_type == BackendType.NOT_SET:
                    logger.warning(f"Tool {t.name} has no backend info, skipping")
                    continue
                t_backend = t.backend_type.value
                t_server = None
            
            for choice in choices:
                # Case-insensitive backend comparison
                choice_backend = choice.get("backend", "").upper()
                if t_backend.upper() == choice_backend and (
                    choice.get("server") is None or t_server == choice.get("server")
                ):
                    result.append(t)
                    break  # Avoid duplicates
        
        # If no tools matched, log warning and return all tools
        if not result:
            logger.warning(
                f"LLM filter matched 0 tools. LLM selected: {choices}. "
                f"Returning all {len(tools)} tools."
            )
            return tools
        
        logger.info(f"LLM filter: {len(tools)} tools → {len(result)} tools")
        return result

    def _log_search_results(self, all_tools: list[BaseTool], filtered_tools: list[BaseTool], mode: str) -> None:
        """
        Log search results in a concise, grouped format.
        Shows backend/server breakdown and tool names (truncated if too many).
        """
        from collections import defaultdict
        
        # Group filtered tools by backend and server
        grouped: Dict[str, Dict[str | None, list[str]]] = defaultdict(lambda: defaultdict(list))
        
        for t in filtered_tools:
            # Get backend and server info
            if t.is_bound:
                backend = t.runtime_info.backend.value
                server = t.runtime_info.server_name if backend.lower() == "mcp" else None
            else:
                if not t.backend_type or t.backend_type == BackendType.NOT_SET:
                    backend = "UNKNOWN"
                    server = None
                else:
                    backend = t.backend_type.value
                    server = None
            
            grouped[backend][server].append(t.name)
        
        # Build concise summary
        lines = [f"\n{'='*60}"]
        lines.append(f"🔍 Tool Search Results (mode: {mode})")
        lines.append(f"   {len(all_tools)} candidates → {len(filtered_tools)} selected tools")
        lines.append(f"{'='*60}")
        
        for backend, srv_map in sorted(grouped.items()):
            backend_total = sum(len(tools) for tools in srv_map.values())
            lines.append(f"\n📦 {backend} ({backend_total} tools)")
            
            for server, tool_names in sorted(srv_map.items()):
                if backend.lower() == "mcp" and server:
                    prefix = f"   └─ {server}: "
                else:
                    prefix = f"   └─ "
                
                # Limit display to avoid overwhelming output
                if len(tool_names) <= 8:
                    tools_display = ", ".join(tool_names)
                else:
                    tools_display = ", ".join(tool_names[:8]) + f" ... (+{len(tool_names)-8} more)"
                
                lines.append(f"{prefix}{tools_display}")
        
        lines.append(f"{'='*60}\n")
        
        # Use info level so users can see it
        logger.info("\n".join(lines))

    @staticmethod
    def _format_tool_list(tools: list[BaseTool]) -> str:
        rows = [f"{i}. **{t.name}**: {t.description}" for i, t in enumerate(tools, 1)]
        return f"Total {len(tools)} tools, list out directly:\n\n" + "\n".join(rows)

    @staticmethod
    def _format_ranked(results: list[tuple[BaseTool, float]], mode: SearchMode) -> str:
        lines = [f"Search results (mode={mode}) total {len(results)}:\n"]
        for i, (tool, score) in enumerate(results, 1):
            lines.append(f"{i}. {tool.name}  (score: {score:.3f})\n    {tool.description}")
        return "\n".join(lines)

    def _run(self, *args, **kwargs):
        raise NotImplementedError("SearchCoordinator only supports asynchronous calls. Use _arun instead.")
    
    def get_embedding_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache.
        
        Returns:
            Dict with cache statistics including total embeddings and breakdown by backend/server.
        """
        return self._ranker.get_cache_stats()
    
    def clear_embedding_cache(self, backend: Optional[str] = None, server: Optional[str] = None) -> int:
        """Clear embeddings from cache.
        
        Args:
            backend: If provided, only clear this backend. If None, clear all.
            server: If provided (and backend is provided), only clear this server.
        
        Returns:
            Number of embeddings cleared.
        """
        return self._ranker.clear_cache(backend=backend, server=server)