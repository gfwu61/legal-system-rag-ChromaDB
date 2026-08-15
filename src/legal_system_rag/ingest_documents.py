
# Import ingestion pipeline from the package namespace
from legal_system_rag.pipelines.ingest import run_ingestion


def main():
    """Main execution entry point for legal document parsing and indexing."""
    print("🚀 Starting legal document ingestion pipeline...")
    run_ingestion()
    print("✅ Ingestion process completed successfully.")


if __name__ == "__main__":
    main()