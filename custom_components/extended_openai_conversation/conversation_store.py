"""Conversation memory store for OpenAI Conversation integration."""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional
import threading

_LOGGER = logging.getLogger(__name__)

# Default conversation expiration time (10 minutes)
DEFAULT_EXPIRATION_TIME = 600  # in seconds


class ConversationStore:
    """Class to manage conversation memory with timeout expiration."""

    def __init__(self, expiration_time: int = DEFAULT_EXPIRATION_TIME) -> None:
        """Initialize the conversation store.
        
        Args:
            expiration_time: Time in seconds before a conversation is considered expired
                             and can be cleaned up.
        """
        self._store: Dict[str, Dict[str, Any]] = {}
        self._expiration_time = expiration_time
        self._lock = threading.RLock()
        
    def get_conversation(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history for a given conversation_id.
        
        Args:
            conversation_id: The ID of the conversation to retrieve.
            
        Returns:
            The conversation messages if found, None otherwise.
        """
        with self._lock:
            if conversation_id in self._store:
                # Update last accessed time
                self._store[conversation_id]["last_accessed"] = time.time()
                return self._store[conversation_id]["messages"]
            return None
            
    def save_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save or update a conversation in the store.
        
        Args:
            conversation_id: The ID of the conversation to save.
            messages: The list of message objects to store.
        """
        with self._lock:
            if conversation_id in self._store:
                self._store[conversation_id]["messages"] = messages
                self._store[conversation_id]["last_accessed"] = time.time()
            else:
                self._store[conversation_id] = {
                    "messages": messages,
                    "last_accessed": time.time(),
                    "sent_domains": set(),
                    "auth_data": None  # Initialize auth_data for new conversations
                }

    def get_sent_domains(self, conversation_id: str) -> set:
        """Get the set of domains that have already been sent in this conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            A set of domain names or an empty set if conversation not found.
        """
        with self._lock:
            if conversation_id in self._store:
                if "sent_domains" not in self._store[conversation_id]:
                    self._store[conversation_id]["sent_domains"] = set()
                return self._store[conversation_id]["sent_domains"]
            return set()
    
    def add_sent_domain(self, conversation_id: str, domain: str) -> None:
        """Add a domain to the list of domains that have been sent in this conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            domain: The domain to mark as sent.
        """
        with self._lock:
            if conversation_id in self._store:
                if "sent_domains" not in self._store[conversation_id]:
                    self._store[conversation_id]["sent_domains"] = set()
                self._store[conversation_id]["sent_domains"].add(domain)
            
    def clean_expired_conversations(self) -> int:
        """Remove conversations that have exceeded the expiration time.
        
        Returns:
            The number of conversations removed.
        """
        current_time = time.time()
        removed_count = 0
        
        with self._lock:
            # Identify expired conversations
            expired_ids = [
                conversation_id 
                for conversation_id, data in self._store.items() 
                if (current_time - data["last_accessed"]) > self._expiration_time
            ]
            
            # Remove expired conversations
            for conversation_id in expired_ids:
                del self._store[conversation_id]
                removed_count += 1
                
        if removed_count > 0:
            _LOGGER.debug(f"Cleaned up {removed_count} expired conversations")
            
        return removed_count
        
    def get_all_conversation_ids(self) -> List[str]:
        """Get a list of all active conversation IDs.
        
        Returns:
            A list of conversation IDs.
        """
        with self._lock:
            return list(self._store.keys())
            
    def get_auth_data(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get voice authentication data for a specific conversation.
        
        Args:
            conversation_id: The ID of the conversation to retrieve auth data for.
            
        Returns:
            The authentication data if found, None otherwise.
        """
        with self._lock:
            if conversation_id in self._store:
                # Update last accessed time
                self._store[conversation_id]["last_accessed"] = time.time()
                return self._store[conversation_id].get("auth_data")
            return None
            
    def save_auth_data(self, conversation_id: str, auth_data: Dict[str, Any]) -> None:
        """Save or update voice authentication data for a conversation.
        
        Args:
            conversation_id: The ID of the conversation to save auth data for.
            auth_data: The authentication data to store.
        """
        with self._lock:
            if conversation_id in self._store:
                self._store[conversation_id]["auth_data"] = auth_data
                self._store[conversation_id]["last_accessed"] = time.time()
            else:
                # Create a new conversation entry if it doesn't exist
                self._store[conversation_id] = {
                    "messages": [],
                    "last_accessed": time.time(),
                    "sent_domains": set(),
                    "auth_data": auth_data
                }
    
    def clear_all(self) -> None:
        """Clear all stored conversations."""
        with self._lock:
            self._store.clear() 