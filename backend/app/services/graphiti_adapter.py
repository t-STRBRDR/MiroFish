"""
Graphiti sync adapter
Wraps graphiti-core async API for Flask synchronous use.
Replaces all zep_cloud usage, backed by Neo4j graph database.

Core design:
- Dedicated daemon thread event loop for async-to-sync bridging
- Neo4j as graph database backend
- Native Gemini support for LLM and embeddings
- API compatible with existing Zep-based code
"""

import asyncio
import threading
import os
import json
import shutil
import uuid as uuid_mod
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable

from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from pydantic import BaseModel, Field

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.graphiti_adapter')

# ---------------------------------------------------------------------------
# Module-level event loop for async operations
# ---------------------------------------------------------------------------

_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _lock:
        if _loop is None or not _loop.is_running():
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_loop.run_forever, daemon=True)
            _thread.start()
    return _loop


def _run_async(coro):
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=600)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GraphitiAdapter:
    """Synchronous adapter for graphiti-core async API backed by Neo4j."""

    _instances: Dict[str, 'GraphitiAdapter'] = {}
    _instances_lock = threading.Lock()

    def __init__(self, graph_id: str):
        self.graph_id = graph_id
        self.db_dir = os.path.join(Config.UPLOAD_FOLDER, 'graphs', graph_id)
        self._graphiti: Optional[Graphiti] = None
        self._gemini_api_key: str = Config.LLM_API_KEY or ''

    @classmethod
    def get_or_create(cls, graph_id: str) -> 'GraphitiAdapter':
        with cls._instances_lock:
            if graph_id not in cls._instances:
                cls._instances[graph_id] = cls(graph_id)
            return cls._instances[graph_id]

    @classmethod
    def remove_instance(cls, graph_id: str):
        with cls._instances_lock:
            inst = cls._instances.pop(graph_id, None)
        if inst is not None:
            try:
                inst.close()
            except Exception:
                logger.warning("Failed to close GraphitiAdapter: %s", graph_id, exc_info=True)

    # --------------------------------------------------------------------------
    # Lazy Graphiti initialisation (Neo4j + Gemini)
    # --------------------------------------------------------------------------

    def _get_graphiti(self) -> Graphiti:
        if self._graphiti is None:
            self._graphiti = _run_async(self._init_graphiti())
        return self._graphiti

    async def _init_graphiti(self) -> Graphiti:
        os.makedirs(self.db_dir, exist_ok=True)

        llm_client = GeminiClient(
            config=LLMConfig(
                api_key=self._gemini_api_key,
                model="gemini-2.5-flash-lite",
            )
        )

        embedder = GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=self._gemini_api_key,
                embedding_model="gemini-embedding-001",
            )
        )

        cross_encoder = OpenAIRerankerClient(
            config=LLMConfig(
                api_key=self._gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                model="gemini-2.5-flash-lite",
            )
        )

        graphiti = Graphiti(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

        await graphiti.build_indices_and_constraints()
        logger.info("Graphiti initialized: graph_id=%s, neo4j=%s", self.graph_id, Config.NEO4J_URI)
        return graphiti

    # --------------------------------------------------------------------------
    # Graph lifecycle
    # --------------------------------------------------------------------------

    def create_graph(self, name: str = "MiroFish Graph") -> str:
        os.makedirs(self.db_dir, exist_ok=True)

        meta_path = os.path.join(self.db_dir, '_meta.json')
        meta = {
            'graph_id': self.graph_id,
            'name': name,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self._get_graphiti()
        logger.info("Graph created: graph_id=%s, name=%s", self.graph_id, name)
        return self.graph_id

    def delete_graph(self):
        self.close()
        if os.path.isdir(self.db_dir):
            shutil.rmtree(self.db_dir, ignore_errors=True)
            logger.info("Graph deleted: graph_id=%s", self.graph_id)

    def graph_exists(self) -> bool:
        return os.path.isdir(self.db_dir)

    # --------------------------------------------------------------------------
    # Ontology (stored as JSON metadata)
    # --------------------------------------------------------------------------

    def _ontology_path(self) -> str:
        return os.path.join(self.db_dir, '_ontology.json')

    def set_ontology(self, ontology_dict: Dict[str, Any]):
        os.makedirs(self.db_dir, exist_ok=True)
        with open(self._ontology_path(), 'w', encoding='utf-8') as f:
            json.dump(ontology_dict, f, ensure_ascii=False, indent=2)
        logger.info("Ontology saved: graph_id=%s", self.graph_id)

    def get_ontology(self) -> Optional[Dict[str, Any]]:
        path = self._ontology_path()
        if not os.path.isfile(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_entity_types(self) -> Optional[Dict[str, type]]:
        """Convert stored ontology into Pydantic models for Graphiti entity typing."""
        ontology = self.get_ontology()
        if not ontology:
            return None

        reserved = set(EntityNode.model_fields.keys())
        entity_types = {}

        for et in ontology.get("entity_types", []):
            name = et.get("name", "")
            if not name:
                continue
            description = et.get("description", f"A {name} entity")

            # Build fields dict for dynamic Pydantic model
            fields = {}
            annotations = {}
            for attr in et.get("attributes", []):
                attr_name = attr.get("name", "")
                if not attr_name or attr_name.lower() in reserved:
                    attr_name = f"entity_{attr_name}"
                attr_desc = attr.get("description", attr_name)
                fields[attr_name] = Field(default=None, description=attr_desc)
                annotations[attr_name] = Optional[str]

            # Create dynamic Pydantic model
            model_attrs = {"__annotations__": annotations, "__doc__": description}
            model_attrs.update(fields)
            model_class = type(name, (BaseModel,), model_attrs)
            entity_types[name] = model_class

        return entity_types if entity_types else None

    # --------------------------------------------------------------------------
    # Episodes
    # --------------------------------------------------------------------------

    def add_episode(self, text: str, source_description: str = "") -> str:
        g = self._get_graphiti()
        episode_id = str(uuid_mod.uuid4())
        source = source_description or "mirofish"
        entity_types = self._build_entity_types()

        async def _add():
            await g.add_episode(
                name=episode_id,
                episode_body=text,
                source_description=source,
                source=EpisodeType.text,
                reference_time=datetime.now(timezone.utc),
                group_id=self.graph_id,
                entity_types=entity_types,
            )
            return episode_id

        result = _run_async(_add())
        logger.debug("Episode added: graph_id=%s, len=%d", self.graph_id, len(text))
        return result

    def add_episodes_bulk(self, texts: List[str]):
        total = len(texts)
        for idx, text in enumerate(texts):
            try:
                self.add_episode(text, source_description="mirofish_bulk")
            except Exception:
                logger.error("Episode add failed (%d/%d): graph_id=%s",
                             idx + 1, total, self.graph_id, exc_info=True)

    # --------------------------------------------------------------------------
    # Node / edge retrieval via Neo4j Cypher
    # --------------------------------------------------------------------------

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        g = self._get_graphiti()

        async def _fetch():
            driver = g.driver
            records, _, _ = await driver.execute_query(
                "MATCH (n:Entity) WHERE n.group_id = $gid "
                "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, "
                "n.summary AS summary, n.created_at AS created_at",
                gid=self.graph_id,
            )
            nodes = []
            for r in records:
                labels = [l for l in (r['labels'] or []) if l not in ('Entity', '__Entity__')]
                nodes.append({
                    'uuid': str(r['uuid'] or ''),
                    'name': str(r['name'] or ''),
                    'labels': labels,
                    'summary': str(r['summary'] or ''),
                    'attributes': {},
                    'created_at': str(r['created_at'] or ''),
                })
            return nodes

        return _run_async(_fetch())

    def get_all_edges(self) -> List[Dict[str, Any]]:
        g = self._get_graphiti()

        async def _fetch():
            driver = g.driver
            records, _, _ = await driver.execute_query(
                "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "WHERE r.group_id = $gid "
                "RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact, "
                "a.uuid AS source_uuid, b.uuid AS target_uuid, "
                "r.created_at AS created_at, r.valid_at AS valid_at, "
                "r.invalid_at AS invalid_at, r.expired_at AS expired_at",
                gid=self.graph_id,
            )
            edges = []
            for r in records:
                edges.append({
                    'uuid': str(r['uuid'] or ''),
                    'name': str(r['name'] or ''),
                    'fact': str(r['fact'] or ''),
                    'source_node_uuid': str(r['source_uuid'] or ''),
                    'target_node_uuid': str(r['target_uuid'] or ''),
                    'attributes': {},
                    'created_at': str(r['created_at'] or ''),
                    'valid_at': str(r['valid_at'] or ''),
                    'invalid_at': str(r['invalid_at'] or ''),
                    'expired_at': str(r['expired_at'] or ''),
                    'episodes': [],
                })
            return edges

        return _run_async(_fetch())

    def get_node(self, node_uuid: str) -> Optional[Dict[str, Any]]:
        g = self._get_graphiti()

        async def _fetch():
            driver = g.driver
            records, _, _ = await driver.execute_query(
                "MATCH (n:Entity) WHERE n.uuid = $uuid "
                "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, "
                "n.summary AS summary, n.created_at AS created_at",
                uuid=node_uuid,
            )
            if not records:
                return None
            r = records[0]
            labels = [l for l in (r['labels'] or []) if l not in ('Entity', '__Entity__')]
            return {
                'uuid': str(r['uuid'] or ''),
                'name': str(r['name'] or ''),
                'labels': labels,
                'summary': str(r['summary'] or ''),
                'attributes': {},
                'created_at': str(r['created_at'] or ''),
            }

        return _run_async(_fetch())

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        g = self._get_graphiti()

        async def _fetch():
            driver = g.driver
            records, _, _ = await driver.execute_query(
                "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "WHERE a.uuid = $uuid OR b.uuid = $uuid "
                "RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact, "
                "a.uuid AS source_uuid, b.uuid AS target_uuid, "
                "r.created_at AS created_at, r.valid_at AS valid_at, "
                "r.invalid_at AS invalid_at, r.expired_at AS expired_at",
                uuid=node_uuid,
            )
            edges = []
            for r in records:
                edges.append({
                    'uuid': str(r['uuid'] or ''),
                    'name': str(r['name'] or ''),
                    'fact': str(r['fact'] or ''),
                    'source_node_uuid': str(r['source_uuid'] or ''),
                    'target_node_uuid': str(r['target_uuid'] or ''),
                    'attributes': {},
                    'created_at': str(r['created_at'] or ''),
                    'valid_at': str(r['valid_at'] or ''),
                    'invalid_at': str(r['invalid_at'] or ''),
                    'expired_at': str(r['expired_at'] or ''),
                    'episodes': [],
                })
            return edges

        return _run_async(_fetch())

    # --------------------------------------------------------------------------
    # Search
    # --------------------------------------------------------------------------

    def search(self, query: str, limit: int = 10, scope: str = "edges") -> Dict[str, Any]:
        g = self._get_graphiti()

        async def _do_search():
            edges: List[Dict[str, Any]] = []
            nodes: List[Dict[str, Any]] = []

            try:
                raw = await g.search(
                    query,
                    num_results=limit,
                    group_ids=[self.graph_id],
                )
                for item in (raw if isinstance(raw, list) else []):
                    edges.append(_entity_edge_to_dict(item))
            except Exception:
                logger.error("Search failed: query=%s", query, exc_info=True)

            return {"edges": edges, "nodes": nodes}

        return _run_async(_do_search())

    # --------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------

    def close(self):
        if self._graphiti is not None:
            try:
                _run_async(self._graphiti.close())
            except Exception:
                logger.debug("Graphiti close error", exc_info=True)
            self._graphiti = None
            logger.info("GraphitiAdapter closed: graph_id=%s", self.graph_id)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _entity_edge_to_dict(edge) -> Dict[str, Any]:
    if isinstance(edge, dict):
        return {
            'uuid': str(edge.get('uuid', '')),
            'name': str(edge.get('name', '')),
            'fact': str(edge.get('fact', '')),
            'source_node_uuid': str(edge.get('source_node_uuid', '')),
            'target_node_uuid': str(edge.get('target_node_uuid', '')),
            'attributes': edge.get('attributes', {}),
            'created_at': str(edge.get('created_at', '')),
            'valid_at': str(edge.get('valid_at', '')),
            'invalid_at': str(edge.get('invalid_at', '')),
            'expired_at': str(edge.get('expired_at', '')),
            'episodes': list(edge.get('episodes', [])) if edge.get('episodes') else [],
        }

    return {
        'uuid': str(getattr(edge, 'uuid', '')),
        'name': str(getattr(edge, 'name', '')),
        'fact': str(getattr(edge, 'fact', '')),
        'source_node_uuid': str(getattr(edge, 'source_node_uuid', '')),
        'target_node_uuid': str(getattr(edge, 'target_node_uuid', '')),
        'attributes': getattr(edge, 'attributes', {}) or {},
        'created_at': str(getattr(edge, 'created_at', '')),
        'valid_at': str(getattr(edge, 'valid_at', '')),
        'invalid_at': str(getattr(edge, 'invalid_at', '')),
        'expired_at': str(getattr(edge, 'expired_at', '')),
        'episodes': list(getattr(edge, 'episodes', [])) if getattr(edge, 'episodes', None) else [],
    }
