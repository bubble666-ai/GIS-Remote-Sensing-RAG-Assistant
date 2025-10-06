"""
RAG (Retrieval-Augmented Generation) pipeline for GIS and Remote Sensing assistant.
Combines document retrieval with LLM generation to provide accurate, sourced answers.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain

from .retriever import GISDocumentRetriever
from .model_providers import ModelProviderFactory, BaseLLMProvider

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GISRAGChain:
    """
    RAG chain specialized for GIS and Remote Sensing queries.
    """

    def __init__(
        self,
        retriever: GISDocumentRetriever,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        provider: Optional[BaseLLMProvider] = None
    ):
        """
        Initialize the RAG chain.

        Args:
            retriever: Document retriever instance
            model_name: Name of the language model
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            provider: Custom LLM provider (optional)
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM provider
        if provider:
            self.llm_provider = provider
        else:
            self.llm_provider = self._initialize_llm_provider()

        # Set up prompts
        self.system_prompt = self._create_system_prompt()
        self.rag_prompt = self._create_rag_prompt()

        # Initialize conversation history
        self.conversation_history = []

    def _initialize_llm_provider(self) -> BaseLLMProvider:
        """
        Initialize the language model provider based on configuration.

        Returns:
            Initialized LLM provider instance
        """
        try:
            provider = ModelProviderFactory.create_provider(
                self.model_name,
                self.temperature,
                self.max_tokens
            )
            logger.info(f"Initialized {provider.__class__.__name__} with model {self.model_name}")
            return provider
        except Exception as e:
            logger.error(f"Error initializing model provider for {self.model_name}: {str(e)}")
            # Fallback to a basic response generator
            return None

    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for the GIS assistant.

        Returns:
            System prompt string
        """
        return """You are a specialized GIS (Geographic Information System) and Remote Sensing assistant. Your expertise includes:

1. Satellite imagery and remote sensing platforms (Landsat, Sentinel, MODIS, etc.)
2. Spectral indices and vegetation analysis (NDVI, EVI, SAVI, etc.)
3. Image processing and atmospheric correction techniques
4. GIS data analysis and spatial statistics
5. Cartography and map projections
6. Geographic coordinate systems and transformations
7. Land use/land cover classification
8. Change detection and time series analysis
9. Digital elevation models and terrain analysis
10. GIS software and tools (ArcGIS, QGIS, ENVI, etc.)

When answering questions:
- Provide accurate, technical information
- Include specific details and methodologies when relevant
- Cite the source documents you used
- Explain complex concepts clearly
- Mention limitations or considerations when applicable
- Use proper GIS terminology

Always acknowledge your sources and indicate when information is based on the provided documents versus general knowledge."""

    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """
        Create the RAG prompt template.

        Returns:
            ChatPromptTemplate instance
        """
        template = """Based on the following GIS and Remote Sensing documents, please answer the user's question.

Context Documents:
{context}

User Question: {question}

Instructions:
1. Use the provided context documents to answer the question
2. If the documents don't contain enough information, supplement with your general GIS knowledge
3. Provide a comprehensive, technically accurate answer
4. Include specific details, formulas, or methodologies when relevant
5. Cite the sources you used from the context documents
6. If you use general knowledge, clearly indicate that

Answer:"""

        return ChatPromptTemplate.from_template(template)

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source = doc['metadata'].get('source', 'Unknown')
            page = doc['metadata'].get('page', 'N/A')
            content = doc['content']

            context_part = f"Document {i+1} (Source: {source}, Page: {page}):\n{content}\n"
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def generate_response(
        self,
        query: str,
        top_k: int = 3,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG pipeline.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            include_sources: Whether to include source citations

        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()

        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
            retrieval_time = time.time() - start_time

            # Format context
            context = self.format_context(retrieved_docs)

            # Generate response
            generation_start = time.time()

            if self.llm_provider is None or not self.llm_provider.is_available():
                # Fallback response when no LLM is available
                response = self._generate_fallback_response(query, retrieved_docs)
                generation_time = time.time() - generation_start
                token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            else:
                # Format messages for the model
                messages = self._format_messages(context, query)

                # Generate response
                response = self.llm_provider.generate_response(messages)
                generation_time = time.time() - generation_start

                # Token usage (placeholder - actual tracking depends on provider)
                token_usage = {
                    "prompt_tokens": len(str(messages)) // 4,  # Rough estimate
                    "completion_tokens": len(response) // 4,  # Rough estimate
                    "total_tokens": (len(str(messages)) + len(response)) // 4
                }

            # Format response with sources
            if include_sources and retrieved_docs:
                sources_info = self._format_sources(retrieved_docs)
                full_response = f"{response}\n\n**Sources:**\n{sources_info}"
            else:
                full_response = response

            total_time = time.time() - start_time

            result = {
                "query": query,
                "response": full_response,
                "sources": retrieved_docs,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "token_usage": token_usage,
                "model_provider": self.llm_provider.__class__.__name__ if self.llm_provider else "None",
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat()
            }

            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat(),
                "sources": retrieved_docs
            })

            logger.info(f"Generated response in {total_time:.2f}s using {self.model_name}")
            return result

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "query": query,
                "response": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "retrieval_time": 0,
                "generation_time": 0,
                "total_time": time.time() - start_time,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model_provider": self.llm_provider.__class__.__name__ if self.llm_provider else "None",
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def _format_messages(self, context: str, query: str) -> List[Dict[str, str]]:
        """
        Format messages for the model provider.

        Args:
            context: Retrieved context documents
            query: User query

        Returns:
            List of formatted messages
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"""Based on the following GIS and Remote Sensing documents, please answer the user's question.

Context Documents:
{context}

User Question: {query}

Instructions:
1. Use the provided context documents to answer the question
2. If the documents don't contain enough information, supplement with your general GIS knowledge
3. Provide a comprehensive, technically accurate answer
4. Include specific details, formulas, or methodologies when relevant
5. Cite the sources you used from the context documents
6. If you use general knowledge, clearly indicate that

Answer:"""
            }
        ]
        return messages

    def _generate_fallback_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a fallback response when no LLM is available.

        Args:
            query: User query
            retrieved_docs: Retrieved documents

        Returns:
            Fallback response
        """
        if retrieved_docs:
            response = f"Based on the documents I found, here's what I can tell you about '{query}':\n\n"
            for i, doc in enumerate(retrieved_docs[:2]):  # Use top 2 documents
                source = doc['metadata'].get('source', 'Unknown')
                content = doc['content'][:300]  # First 300 characters
                response += f"From {source}: {content}...\n\n"

            response += "Note: This is a simplified response. For more detailed answers, please configure an OpenAI API key."
        else:
            response = f"I couldn't find specific information about '{query}' in the available documents. This might be because:\n\n"
            response += "1. The documents don't contain this specific information\n"
            response += "2. The query needs to be rephrased\n"
            response += "3. Additional documents need to be added to the knowledge base\n\n"
            response += "Please try rephrasing your question or add more relevant GIS documents to the data folder."

        return response

    def _format_sources(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format sources for citation.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            Formatted sources string
        """
        if not retrieved_docs:
            return "No sources found."

        sources = []
        for i, doc in enumerate(retrieved_docs):
            source = doc['metadata'].get('source', 'Unknown')
            page = doc['metadata'].get('page', 'N/A')
            score = doc.get('score', 0)

            source_info = f"{i+1}. {source}"
            if page != 'N/A':
                source_info += f" (Page {page})"
            source_info += f" [Relevance: {score:.3f}]"
            sources.append(source_info)

        return "\n".join(sources)

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.

        Returns:
            Dictionary with conversation statistics
        """
        if not self.conversation_history:
            return {"message": "No conversation history"}

        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.conversation_history if msg["role"] == "assistant"]

        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "first_message_time": self.conversation_history[0]["timestamp"],
            "last_message_time": self.conversation_history[-1]["timestamp"]
        }

    def update_model_settings(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> None:
        """
        Update model settings.

        Args:
            model_name: New model name
            temperature: New temperature
            max_tokens: New max tokens
        """
        if model_name is not None:
            self.model_name = model_name
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens

        # Reinitialize LLM provider with new settings
        self.llm_provider = self._initialize_llm_provider()
        logger.info(f"Updated model settings: {self.model_name}, temp={self.temperature}, max_tokens={self.max_tokens}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        if self.llm_provider:
            info = self.llm_provider.get_model_info()
            info["rag_chain_initialized"] = True
            return info
        else:
            return {
                "model_name": self.model_name,
                "provider": "None",
                "available": False,
                "rag_chain_initialized": False,
                "error": "Model provider not initialized"
            }

    def switch_model(self, model_name: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> bool:
        """
        Switch to a different model.

        Args:
            model_name: New model name
            temperature: New temperature (optional)
            max_tokens: New max tokens (optional)

        Returns:
            True if model switch was successful, False otherwise
        """
        try:
            # Update settings
            old_model = self.model_name
            self.model_name = model_name

            if temperature is not None:
                self.temperature = temperature
            if max_tokens is not None:
                self.max_tokens = max_tokens

            # Initialize new provider
            self.llm_provider = self._initialize_llm_provider()

            if self.llm_provider and self.llm_provider.is_available():
                logger.info(f"Successfully switched from {old_model} to {model_name}")
                return True
            else:
                logger.error(f"Failed to initialize {model_name}")
                # Revert to old model
                self.model_name = old_model
                self.llm_provider = self._initialize_llm_provider()
                return False

        except Exception as e:
            logger.error(f"Error switching model: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    from retriever import GISDocumentRetriever

    # Initialize retriever and RAG chain
    retriever = GISDocumentRetriever()
    retriever.initialize_or_load()

    rag_chain = GISRAGChain(retriever)

    # Test the chain
    query = "What is NDVI and how is it calculated from satellite imagery?"
    result = rag_chain.generate_response(query, top_k=3)

    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Retrieval time: {result['retrieval_time']:.2f}s")
    print(f"Generation time: {result['generation_time']:.2f}s")
    print(f"Total time: {result['total_time']:.2f}s")
    print(f"Token usage: {result['token_usage']}")