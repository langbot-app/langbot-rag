import unittest

from benchmarks.sdk_stubs import install_stubs

install_stubs()

from components.knowledge_engine.langrag import LangRAG
from langbot_plugin.api.entities.builtin.rag import (
    DocumentStatus,
    FileMetadata,
    FileObject,
    IngestionContext,
    ParseResult,
)


class RecordingIngestPlugin:
    def __init__(self):
        self.read_called = False
        self.upserts = []

    async def get_knowledge_file_stream(self, storage_path):
        self.read_called = True
        raise AssertionError("external parser content should skip file reads")

    async def invoke_embedding(self, embedding_model_uuid, texts):
        return [[float(i)] for i, _ in enumerate(texts)]

    async def vector_upsert(self, **kwargs):
        self.upserts.append(kwargs)


class RecordingVectorPlugin:
    def __init__(self, items):
        self.items = items
        self.requested_ids = []

    async def vector_get_by_ids(self, collection_id, ids):
        self.requested_ids = ids
        return [self.items[item_id] for item_id in ids if item_id in self.items]


class LangRAGTests(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_uses_external_parse_result_without_reading_file(self):
        engine = LangRAG()
        plugin = RecordingIngestPlugin()
        engine.plugin = plugin

        context = IngestionContext(
            file_object=FileObject(
                metadata=FileMetadata(
                    filename="sample.txt",
                    file_size=12,
                    mime_type="text/plain",
                    document_id="doc1",
                    knowledge_base_id="kb1",
                ),
                storage_path="/missing/sample.txt",
            ),
            knowledge_base_id="kb1",
            creation_settings={
                "embedding_model_uuid": "emb1",
                "chunk_size": 100,
                "overlap": 0,
            },
            parsed_content=ParseResult(text="external parser text"),
        )

        result = await engine.ingest(context)

        self.assertEqual(result.status, DocumentStatus.COMPLETED)
        self.assertFalse(plugin.read_called)
        self.assertEqual(result.chunks_created, 1)
        self.assertEqual(plugin.upserts[0]["documents"], ["external parser text"])

    async def test_context_window_uses_parent_child_id_scheme(self):
        engine = LangRAG()
        engine.plugin = RecordingVectorPlugin(
            {
                "doc1_p1_c0": {
                    "id": "doc1_p1_c0",
                    "metadata": {"text": "previous parent"},
                },
                "doc1_p3_c0": {
                    "id": "doc1_p3_c0",
                    "metadata": {"text": "next parent"},
                },
            }
        )
        results = [
            {
                "metadata": {
                    "document_id": "doc1",
                    "index_type": "parent_child",
                    "parent_index": 2,
                }
            }
        ]

        await engine._expand_context(results, "kb1", 1)

        self.assertEqual(set(engine.plugin.requested_ids), {"doc1_p1_c0", "doc1_p3_c0"})
        self.assertEqual(results[0]["metadata"]["context_before"], "previous parent")
        self.assertEqual(results[0]["metadata"]["context_after"], "next parent")

    def test_neighbor_id_supports_all_index_types(self):
        self.assertEqual(
            LangRAG._neighbor_id(
                {"document_id": "doc1", "index_type": "chunk", "chunk_index": 2},
                -1,
            ),
            "doc1_1",
        )
        self.assertEqual(
            LangRAG._neighbor_id(
                {"document_id": "doc1", "index_type": "qa", "chunk_index": "2"},
                1,
            ),
            "doc1_3_qa0",
        )
        self.assertEqual(
            LangRAG._neighbor_id(
                {
                    "document_id": "doc1",
                    "index_type": "parent_child",
                    "parent_index": 2,
                },
                1,
            ),
            "doc1_p3_c0",
        )


if __name__ == "__main__":
    unittest.main()
