"""Voice authentication middleware for extended_openai_conversation."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, List, Tuple

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.components import conversation

from .const import (
    CONF_VOICE_USERS,
    DEFAULT_VOICE_USERS,
    CONF_VOICE_AUTH_THRESHOLD_MARGIN,
    DEFAULT_VOICE_AUTH_THRESHOLD_MARGIN,
    CONF_VOICE_AUTH_CACHE_EXPIRATION,
    DEFAULT_VOICE_AUTH_CACHE_EXPIRATION,
)

_LOGGER = logging.getLogger(__name__)


class VoiceAuthorizationMiddleware:
    """Middleware to check authorization for voice commands."""

    def __init__(self, hass: HomeAssistant, config: Optional[Dict[str, Any]] = None):
        """Initialize the middleware."""
        self.hass = hass
        self._config = config or {}
        self._voice_users = self._config.get(CONF_VOICE_USERS, DEFAULT_VOICE_USERS)
        
        # Configuration for authentication thresholds
        self._threshold_margin = self._config.get(
            CONF_VOICE_AUTH_THRESHOLD_MARGIN, DEFAULT_VOICE_AUTH_THRESHOLD_MARGIN
        )
        self._cache_expiration = self._config.get(
            CONF_VOICE_AUTH_CACHE_EXPIRATION, DEFAULT_VOICE_AUTH_CACHE_EXPIRATION
        )
        
        # Authentication cache format: {conversation_id: (timestamp, profile_name, authenticated, confidence)}
        # This allows for maintaining authentication state within a conversation
        self._auth_cache = {}

    async def check_domain_authorization(
        self, domain: str, user_input: conversation.ConversationInput
    ) -> bool:
        """Check if the speaker is authorized to access a domain.
        
        Returns True if authorized, False otherwise.
        """
        # Get metadata from the agent instance if possible
        speech_metadata = None
        caller_self = None
        
        from inspect import currentframe, getouterframes
        caller_frame = getouterframes(currentframe(), 2)
        
        # Try to get the agent instance from the caller
        for frame in caller_frame:
            if 'self' in frame.frame.f_locals and hasattr(frame.frame.f_locals['self'], 'current_auth_metadata'):
                caller_self = frame.frame.f_locals['self']
                break
                
        if caller_self and hasattr(caller_self, 'current_auth_metadata'):
            speech_metadata = caller_self.current_auth_metadata
        elif hasattr(user_input, "metadata") and user_input.metadata:
            speech_metadata = user_input.metadata.get("speech_metadata", {})
            
        if not speech_metadata:
            _LOGGER.error("⛔ No speech metadata available for authorization - access denied")
            return False

        # Get the profile name and check if user is authenticated
        profile_name = speech_metadata.get("profile_name", "default")
        authenticated = speech_metadata.get("authenticated", False)
        confidence = speech_metadata.get("confidence", 0.0)
        
        # If confidence is critically low, reject all access
        if confidence < 0.4 and profile_name != "default":
            _LOGGER.error(f"\u26d4 CRITICALLY LOW CONFIDENCE: {confidence:.4f} - Domain access rejected")
            return False
            
        # Log authentication status but don't modify profiles
        if not authenticated and profile_name != "default" and profile_name != "unknown":
            _LOGGER.warning(f"User '{profile_name}' not authenticated (conf: {confidence:.4f}) for domain check")
        
        # Get user profile permissions
        profile = self._voice_users.get(profile_name, self._voice_users.get("default", {}))
        
        # Get permissions
        permissions = profile.get("permissions", {})
        allow = permissions.get("allow", {})
        deny = permissions.get("deny", {})
        
        # Check allow/deny domains
        allowed_domains = allow.get("domains", [])
        denied_domains = deny.get("domains", [])
        
        # Check if domain is explicitly denied
        if domain in denied_domains:
            _LOGGER.error(f"⛔ DENIED: Domain '{domain}' is explicitly denied for user '{profile_name}'")
            return False
            
        # Check if domain is explicitly allowed ("*" means all domains)
        if allowed_domains == "*" or (isinstance(allowed_domains, list) and domain in allowed_domains):
            _LOGGER.warning(f"✅ AUTHORIZED: User '{profile_name}' has permission to access domain '{domain}'")
            return True
            
        # Default to denying access
        _LOGGER.error(f"❌ DENIED: User '{profile_name}' does not have permission to access domain '{domain}'")
        return False

    async def check_entity_authorization(
        self, entity_id: str, user_input: conversation.ConversationInput
    ) -> bool:
        """Check if the speaker is authorized to access an entity.
        
        Returns True if authorized, False otherwise.
        """
        # Get metadata from the agent instance if possible
        speech_metadata = None
        caller_self = None
        
        from inspect import currentframe, getouterframes
        caller_frame = getouterframes(currentframe(), 2)
        
        # Try to get the agent instance from the caller
        for frame in caller_frame:
            if 'self' in frame.frame.f_locals and hasattr(frame.frame.f_locals['self'], 'current_auth_metadata'):
                caller_self = frame.frame.f_locals['self']
                break
                
        if caller_self and hasattr(caller_self, 'current_auth_metadata'):
            speech_metadata = caller_self.current_auth_metadata
        elif hasattr(user_input, "metadata") and user_input.metadata:
            speech_metadata = user_input.metadata.get("speech_metadata", {})
            
        if not speech_metadata:
            _LOGGER.error("⛔ No speech metadata available for entity authorization - access denied")
            return False

        # Get the profile name and authentication status
        profile_name = speech_metadata.get("profile_name", "default")
        authenticated = speech_metadata.get("authenticated", False)
        confidence = speech_metadata.get("confidence", 0.0)
        
        # If confidence is critically low, reject all access
        if confidence < 0.4 and profile_name != "default":
            _LOGGER.error(f"\u26d4 CRITICALLY LOW CONFIDENCE: {confidence:.4f} - Entity access rejected")
            return False
            
        # Log authentication status but don't modify profiles
        if not authenticated and profile_name != "default" and profile_name != "unknown":
            _LOGGER.warning(f"User '{profile_name}' not authenticated (conf: {confidence:.4f}) for entity access")
            
        # Get entity domain
        domain = entity_id.split(".")[0] if "." in entity_id else None
        if not domain:
            _LOGGER.error(f"⛔ Invalid entity ID format: '{entity_id}'")
            return False
            
        # Check domain authorization first
        if not await self.check_domain_authorization(domain, user_input):
            _LOGGER.error(f"⛔ Domain '{domain}' authorization failed for entity '{entity_id}'")
            return False
            
        # Get user profile permissions
        profile = self._voice_users.get(profile_name, self._voice_users.get("default", {}))
        
        # Get permissions
        permissions = profile.get("permissions", {})
        allow = permissions.get("allow", {})
        deny = permissions.get("deny", {})
        
        # Check allow/deny entities
        allowed_entities = allow.get("entities", [])
        denied_entities = deny.get("entities", [])
        
        # Check if entity is explicitly denied
        if entity_id in denied_entities:
            _LOGGER.error(f"⛔ DENIED: Entity '{entity_id}' is explicitly denied for user '{profile_name}'")
            return False
            
        # Check if entity is explicitly allowed ("*" means all entities)
        if allowed_entities == "*" or (isinstance(allowed_entities, list) and entity_id in allowed_entities):
            _LOGGER.warning(f"✅ AUTHORIZED: User '{profile_name}' has permission to access entity '{entity_id}'")
            return True
            
        # Default to denying access
        _LOGGER.error(f"❌ DENIED: User '{profile_name}' does not have permission to access entity '{entity_id}'")
        return False

    async def authorize_tool_call(
        self, tool_call: Dict[str, Any], user_input: conversation.ConversationInput
    ) -> Dict[str, Any]:
        """Authorize a tool call based on the speaker's authorization level.
        
        Returns a dict with:
        - authorized: bool - Whether the call is authorized
        - error: Optional[str] - Error message if not authorized
        """
        # Get the metadata from the agent instance, not from user_input
        # We check if the agent has the current_auth_metadata attribute
        from inspect import currentframe, getouterframes
        caller_frame = getouterframes(currentframe(), 2)
        caller_self = None
        
        # First check if we have cached authentication for this conversation
        cache_key = self._get_cache_key(user_input)
        cached_auth = self._get_from_auth_cache(cache_key)
        
        if cached_auth:
            cached_profile, cached_authenticated, cached_confidence = cached_auth
            _LOGGER.info(f"Found cached auth for function call: profile={cached_profile}, authenticated={cached_authenticated}, confidence={cached_confidence:.4f}")
            
            # If we have a cached authentication and it's not default, use it
            if cached_profile != "default" and cached_profile != "unknown":
                if cached_authenticated:
                    # Authenticated profile from cache - proceed with authorization checks
                    _LOGGER.info(f"Using cached authenticated profile: {cached_profile}")
                    speech_metadata = {
                        "profile_name": cached_profile,
                        "authenticated": True,
                        "confidence": cached_confidence
                    }
                else:
                    # Profile identified but not authenticated - deny function calls
                    _LOGGER.warning(f"Cached profile {cached_profile} is not authenticated - denying function call")
                    return {
                        "authorized": False,
                        "error": f"Speaker identified as {cached_profile} but not authenticated. Authentication required for function calls."
                    }
            elif cached_profile == "default":
                # Default profile with appropriate permissions
                _LOGGER.info("Using default profile for function authorization")
                speech_metadata = {
                    "profile_name": "default",
                    "authenticated": False,
                    "confidence": cached_confidence
                }
            else:
                # Unknown profile - deny function calls
                _LOGGER.error("Unknown profile in cache - denying function call")
                return {
                    "authorized": False,
                    "error": "Unknown profile, authorization denied."
                }
        else:
            # No cached authentication - try to get from agent or metadata
            # Try to get the agent instance from the caller
            for frame in caller_frame:
                if 'self' in frame.frame.f_locals and hasattr(frame.frame.f_locals['self'], 'current_auth_metadata'):
                    caller_self = frame.frame.f_locals['self']
                    break
                    
            if caller_self and hasattr(caller_self, 'current_auth_metadata'):
                speech_metadata = caller_self.current_auth_metadata
                _LOGGER.warning(f"\ud83d\udcca Retrieved auth metadata from agent: {speech_metadata.get('profile_name', 'unknown')}")
            elif hasattr(user_input, "metadata") and user_input.metadata:
                speech_metadata = user_input.metadata.get("speech_metadata", {})
            else:
                _LOGGER.error("\u26d4 NO AUTH METADATA FOUND - DENYING ALL TOOL CALLS")
                return {
                    "authorized": False,
                    "error": "No authentication metadata available"
                }
                
            # Process the newly retrieved metadata to update cache
            auth_data = self.process_speaker_recognition(user_input)
            
            # Use the processed data for authorization
            if auth_data:
                # Update our local copy of speech_metadata with processed data
                speech_metadata = auth_data
            
        # Extract authentication details
        profile_name = speech_metadata.get("profile_name", "unknown")
        authenticated = speech_metadata.get("authenticated", False)
        confidence = speech_metadata.get("confidence", 0.0)
        
        # If confidence is critically low, reject completely (don't even use default profile)
        if confidence < 0.4 and profile_name != "default":  # Very low confidence
            _LOGGER.error(f"\u26d4 CRITICALLY LOW CONFIDENCE: {confidence:.4f} - Rejecting all function calls")
            return {
                "authorized": False,
                "error": f"Voice confidence too low ({confidence:.4f}) to authorize any actions"
            }
        
        # Check if the user should be authenticated for function calls
        if not authenticated and profile_name != "default" and profile_name != "unknown":
            _LOGGER.warning(f"User '{profile_name}' not authenticated (conf: {confidence:.4f}) - Denying function calls")
            return {
                "authorized": False,
                "error": f"Speaker identified as {profile_name} but not authenticated (confidence: {confidence:.4f}). Authentication required for function calls."
            }
            
        _LOGGER.warning(f"\ud83d\udd11 Function authorization using profile: {profile_name}, authenticated={authenticated}")
            
        # Check function name
        function_name = tool_call.get("function", {}).get("name")
        if not function_name:
            return {"authorized": True}
            
        # Extract arguments
        try:
            arguments = tool_call.get("function", {}).get("arguments", "{}")
            if isinstance(arguments, str):
                import json
                arguments = json.loads(arguments)
        except Exception as err:
            _LOGGER.warning("Error parsing tool arguments: %s", err)
            # On error, default to denying for security
            return {
                "authorized": False,
                "error": f"Error parsing tool arguments: {err}"
            }
            
        # Check authorization based on function
        if function_name == "execute_services":
            # This function takes a list of service calls
            for service_call in arguments.get("list", []):
                domain = service_call.get("domain")
                if not domain:
                    continue
                    
                # Check domain authorization
                is_authorized = await self.check_domain_authorization(domain, user_input)
                if not is_authorized:
                    speaker = speech_metadata.get("user_id", profile_name)
                    return {
                        "authorized": False,
                        "error": f"Speaker '{speaker}' is not authorized to access domain '{domain}'."
                    }
                    
                # Check entity authorization if entity_id is present
                service_data = service_call.get("service_data", {})
                entity_id = service_data.get("entity_id")
                if entity_id:
                    # Handle both single entity_id and lists
                    if isinstance(entity_id, list):
                        for eid in entity_id:
                            is_authorized = await self.check_entity_authorization(eid, user_input)
                            if not is_authorized:
                                speaker = speech_metadata.get("user_id", profile_name)
                                return {
                                    "authorized": False,
                                    "error": f"Speaker '{speaker}' is not authorized to access entity '{eid}'."
                                }
                    else:
                        is_authorized = await self.check_entity_authorization(entity_id, user_input)
                        if not is_authorized:
                            speaker = speech_metadata.get("user_id", profile_name)
                            return {
                                "authorized": False,
                                "error": f"Speaker '{speaker}' is not authorized to access entity '{entity_id}'."
                            }
        
        # For authenticated users with appropriate permissions, allow the call
        return {"authorized": True}
        
    def _get_cache_key(self, user_input: Any) -> str:
        """Get a cache key for the user input.
        
        This attempts to extract a conversation ID or a unique identifier
        to use as a cache key for tracking authentication within a conversation.
        
        Args:
            user_input: The user input object, typically a ConversationInput
            
        Returns:
            A string that can be used as a cache key
        """
        # Try to extract a conversation ID from metadata
        if hasattr(user_input, "conversation_id") and user_input.conversation_id:
            return f"conversation_{user_input.conversation_id}"
            
        # If no conversation ID is available, try to use a unique ID
        if hasattr(user_input, "metadata") and user_input.metadata:
            conversation_id = user_input.metadata.get("conversation_id")
            if conversation_id:
                return f"conversation_{conversation_id}"
                
        # If we can't get a meaningful ID, use a fallback based on the object id
        return f"fallback_{id(user_input)}"
        
    def _update_auth_cache(self, key: str, profile_name: str, authenticated: bool, confidence: float) -> None:
        """Update the authentication cache with new information.
        
        Args:
            key: The cache key (typically conversation ID)
            profile_name: The profile name to cache
            authenticated: Whether the profile is authenticated
            confidence: The confidence score of the authentication
        """
        self._auth_cache[key] = (time.time(), profile_name, authenticated, confidence)
        _LOGGER.info(f"Updated auth cache for {key}: profile={profile_name}, authenticated={authenticated}, confidence={confidence:.4f}")
        
    def _get_from_auth_cache(self, key: str) -> Optional[Tuple[str, bool, float]]:
        """Get authentication information from the cache.
        
        Args:
            key: The cache key (typically conversation ID)
            
        Returns:
            A tuple of (profile_name, authenticated, confidence) if found and not expired,
            or None if not found or expired
        """
        if key not in self._auth_cache:
            return None
            
        timestamp, profile_name, authenticated, confidence = self._auth_cache[key]
        
        # Check if the cache entry has expired
        if time.time() - timestamp > self._cache_expiration:
            _LOGGER.info(f"Auth cache for {key} has expired, removing")
            del self._auth_cache[key]
            return None
            
        return (profile_name, authenticated, confidence)
        
    def _clean_expired_cache(self) -> None:
        """Clean expired entries from the auth cache."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, (timestamp, _, _, _) in self._auth_cache.items():
            if current_time - timestamp > self._cache_expiration:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self._auth_cache[key]
            
        if keys_to_remove:
            _LOGGER.debug(f"Cleaned {len(keys_to_remove)} expired auth cache entries")
            
    def find_profile_by_voice(self, voice_name: str) -> str:
        """Find a user profile that contains the given voice.
        
        Args:
            voice_name: The voice/speaker name from recognition
            
        Returns:
            The profile name that contains this voice, or "default" if not found
        """
        if not voice_name or voice_name == "unknown":
            return "default"
            
        # Check each profile to see if it contains this voice
        for profile_name, profile in self._voice_users.items():
            voices = profile.get("voices", [])
            if voice_name in voices:
                return profile_name
                
        # If no profile matches, return default
        return "default"
        
    def process_speaker_recognition(self, data: Any) -> Dict[str, Any]:
        """Process speaker recognition data.
    
        Data can be either a ConversationInput object or a metadata dictionary.
        Returns a dict with appropriate profile information.
        """
        # Handle different input types
        speech_metadata = {}
        user_input = data
        
        if isinstance(data, dict):
            # Direct metadata dictionary
            speech_metadata = data.get("speech_metadata", {})
        elif hasattr(data, "metadata") and data.metadata:
            # ConversationInput object with metadata
            speech_metadata = data.metadata.get("speech_metadata", {})
        else:
            _LOGGER.warning("No metadata found in input - voice metadata won't be available")
            return {}
            
        if not speech_metadata:
            _LOGGER.warning("No speech_metadata found in metadata")
            return {}
            
        # Clean expired cache entries
        self._clean_expired_cache()
            
        # Extract data from speech metadata    
        profile_name = speech_metadata.get("profile_name", "unknown")
        speaker = speech_metadata.get("speaker", "unknown")
        confidence = speech_metadata.get("speaker_confidence", 0.0)
        authenticated = speech_metadata.get("authenticated", False)
        user_id = speech_metadata.get("user_id", "unknown")
        raw_result = speech_metadata.get("raw_result", {})
        
        # Check for best match info even if below threshold
        best_match = raw_result.get("best_match", None)
        similarity = raw_result.get("similarity", 0.0)
        threshold = raw_result.get("threshold", 0.75)

        # Find the user profile that matches the detected voice
        # This is a critical step that maps the voice name to a user profile
        mapped_profile_name = "default"
        if best_match and best_match != "unknown":
            mapped_profile_name = self.find_profile_by_voice(best_match)
            _LOGGER.warning(f"🎙️ Voice identification: speaker={best_match}, looking for matching profile")
            _LOGGER.warning(f"🔍 DEBUG: Raw similarity/confidence from recognition: {similarity:.4f}")
            if mapped_profile_name != "default":
                _LOGGER.warning(f"🎙️ Found voice '{best_match}' in profile '{mapped_profile_name}'")
            else:
                _LOGGER.warning(f"🎙️ No profile found for voice '{best_match}'")
        elif speaker and speaker != "unknown":
            mapped_profile_name = self.find_profile_by_voice(speaker)
            _LOGGER.warning(f"🎙️ Voice identification: speaker={speaker}, looking for matching profile")
            _LOGGER.warning(f"🔍 DEBUG: Raw confidence from metadata: {confidence:.4f}")
            if mapped_profile_name != "default":
                _LOGGER.warning(f"🎙️ Found voice '{speaker}' in profile '{mapped_profile_name}'")
            else:
                _LOGGER.warning(f"🎙️ No profile found for voice '{speaker}'")
        
        _LOGGER.warning(f"🎙️ Final voice mapping: speaker={best_match or speaker}, profile={mapped_profile_name}")
        _LOGGER.warning(f"🔍 DEBUG INPUT METADATA: {speech_metadata}")
        
        # Get cache key for this conversation
        cache_key = self._get_cache_key(user_input)
        cached_data = self._get_from_auth_cache(cache_key)
        
        # Get previous authentication state if available
        previous_profile = None
        previous_authenticated = False
        previous_confidence = 0.0
        
        if cached_data:
            previous_profile, previous_authenticated, previous_confidence = cached_data
            _LOGGER.info(f"Found cached auth for {cache_key}: profile={previous_profile}, authenticated={previous_authenticated}, confidence={previous_confidence:.4f}")
        
        # Use the mapped profile name instead of the speaker name
        current_profile = mapped_profile_name
        current_confidence = similarity or confidence
        
        # Determine authentication status based on confidence and threshold
        _LOGGER.warning(f"🔍 DEBUG: Current confidence before auth check: {current_confidence:.4f}, threshold: {threshold}")
        if current_confidence >= threshold:
            current_authenticated = True
            _LOGGER.warning(f"🔒 Voice authentication: mapping {best_match or speaker} to profile '{mapped_profile_name}' with confidence {current_confidence:.4f}")
        elif current_confidence >= (threshold - self._threshold_margin):
            current_authenticated = False
            _LOGGER.warning(f"🔒 Voice authentication: mapping {best_match or speaker} to profile '{mapped_profile_name}' with confidence {current_confidence:.4f}")
        else:
            # Confidence too low, use default profile
            current_profile = "default"
            current_authenticated = False
            _LOGGER.warning(f"🔒 Voice authentication: confidence too low ({current_confidence:.4f}), using default profile")
        
        # Now evaluate against previous cached authentication state
        final_profile = current_profile
        final_authenticated = current_authenticated 
        final_confidence = current_confidence
        
        # Rules for authentication changes:
        # 1. If current authentication is successful, use it regardless of previous state
        # 2. If current match is below threshold but within margin, use previous authentication if available
        # 3. If current confidence is very low (below threshold-margin), downgrade to default regardless
        
        if current_authenticated:
            # Rule 1: Current fully authenticated - highest priority
            final_profile = current_profile
            final_authenticated = True
            final_confidence = current_confidence
            _LOGGER.info(f"Using current authentication: profile='{final_profile}', auth=True, confidence={final_confidence:.4f}")
            
        elif cached_data and previous_authenticated and current_profile == previous_profile:
            # Previous authentication for same profile exists - maintain it
            final_profile = previous_profile
            final_authenticated = True
            final_confidence = previous_confidence  # Keep previous confidence score
            _LOGGER.info(f"Maintaining previous authentication for {previous_profile}")
            
        elif current_confidence >= (threshold - self._threshold_margin):
            # Within threshold margin but not authenticated - use current profile
            final_profile = current_profile
            final_authenticated = False
            final_confidence = current_confidence
            _LOGGER.info(f"Using current profile but not authenticated: {final_profile} (confidence: {final_confidence:.4f})")
            
        else:
            # Confidence too low, use default profile
            final_profile = "default"
            final_authenticated = False
            final_confidence = current_confidence
            _LOGGER.warning(f"Confidence too low for authentication, using default profile")
        
        # Update the authentication cache with the final state
        self._update_auth_cache(cache_key, final_profile, final_authenticated, final_confidence)
        
        _LOGGER.warning(f"💾 SAVED AUTH STATE: conversation={cache_key}, speaker={best_match or speaker}, profile={final_profile}, fully_authenticated={final_authenticated}")
        
        # Return structured data with the final authentication state
        if final_profile != "unknown":
            # Use the profile_name to find the user profile
            user_profile = self._voice_users.get(final_profile, self._voice_users.get("default", {}))
            
            result = {
                "profile_name": final_profile,
                "authenticated": final_authenticated,
                "speaker": best_match or speaker,
                "confidence": final_confidence,
                "user_id": user_profile.get('user_id', 'default'),
                "user_profile": user_profile,
                "raw_result": raw_result,
                "fully_authenticated": final_authenticated
            }
            
            _LOGGER.warning(f"🔍 DEBUG RETURNING AUTH RESULT: {result}")
            return result
            
        return {}
