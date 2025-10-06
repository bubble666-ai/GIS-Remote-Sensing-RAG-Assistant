#!/usr/bin/env python3
"""
Integration test for XAI Grok model with the RAG system.
"""

import os
import sys
sys.path.append('app')

from dotenv import load_dotenv
from app.model_providers import ModelProviderFactory
from app.retriever import GISDocumentRetriever
from app.rag_chain import GISRAGChain

# Load environment variables
load_dotenv()

def test_xai_integration():
    """Test XAI Grok integration with the RAG system."""
    print("XAI Grok Integration Test")
    print("="*50)

    try:
        # Step 1: Test model provider directly
        print("Step 1: Testing XAI model provider...")
        provider = ModelProviderFactory.create_provider("grok-3", temperature=0.1, max_tokens=500)

        if not provider.is_available():
            print("FAILED: XAI Grok provider not available")
            return False

        print("SUCCESS: XAI Grok provider is available")

        # Test direct response
        messages = [
            {"role": "system", "content": "You are a GIS expert."},
            {"role": "user", "content": "What is the formula for NDVI?"}
        ]

        response = provider.generate_response(messages)
        print(f"Direct response: {response[:100]}...")

        # Step 2: Test with retriever
        print("\nStep 2: Testing document retriever...")
        retriever = GISDocumentRetriever()
        retriever.initialize_or_load()

        # Test retrieval
        docs = retriever.retrieve("NDVI calculation", top_k=2)
        print(f"Retrieved {len(docs)} documents")

        # Step 3: Test full RAG chain
        print("\nStep 3: Testing full RAG chain...")
        rag_chain = GISRAGChain(
            retriever=retriever,
            model_name="grok-3",
            temperature=0.1,
            max_tokens=500
        )

        # Test query
        result = rag_chain.generate_response(
            query="How is NDVI calculated from satellite imagery?",
            top_k=2,
            include_sources=True
        )

        print("SUCCESS: Full RAG chain test completed")
        print(f"Model: {result.get('model_name', 'Unknown')}")
        print(f"Response time: {result.get('total_time', 0):.2f}s")
        print(f"Sources used: {len(result.get('sources', []))}")
        print(f"Response: {result.get('response', '')[:200]}...")

        return True

    except Exception as e:
        print(f"FAILED: Integration test error: {str(e)}")
        return False

def main():
    """Main test function."""
    success = test_xai_integration()

    if success:
        print("\n" + "="*50)
        print("SUCCESS: All XAI Grok integration tests passed!")
        print("The model is ready for use in the full application")
        print("\nTo start the application:")
        print("streamlit run app/main.py")
    else:
        print("\nFAILED: Integration tests failed")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()