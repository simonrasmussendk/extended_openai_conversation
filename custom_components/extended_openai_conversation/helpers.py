from abc import ABC, abstractmethod
from datetime import timedelta
from functools import partial
import asyncio
import logging
import os
import re
import sqlite3
import time
from typing import Any
from urllib import parse
import json
import glob
import yaml

from bs4 import BeautifulSoup
from openai import AsyncAzureOpenAI, AsyncOpenAI
import voluptuous as vol
import yaml

from homeassistant.components import (
    automation,
    conversation,
    energy,
    recorder,
    rest,
    scrape,
)
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.components.script.config import SCRIPT_ENTITY_SCHEMA
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import (
    CONF_ATTRIBUTE,
    CONF_METHOD,
    CONF_NAME,
    CONF_PAYLOAD,
    CONF_RESOURCE,
    CONF_RESOURCE_TEMPLATE,
    CONF_TIMEOUT,
    CONF_VALUE_TEMPLATE,
    CONF_VERIFY_SSL,
    SERVICE_RELOAD,
)
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.script import Script
from homeassistant.helpers.template import Template
import homeassistant.util.dt as dt_util

from .const import CONF_PAYLOAD_TEMPLATE, DOMAIN, EVENT_AUTOMATION_REGISTERED
from .exceptions import (
    CallServiceError,
    EntityNotExposed,
    EntityNotFound,
    FunctionNotFound,
    InvalidFunction,
    NativeNotFound,
    UnauthorizedAccess,
)

from homeassistant.components.recorder.db_schema import States, RecorderRuns, StateAttributes, StatisticsShortTerm
from homeassistant.components.recorder.util import session_scope

from homeassistant.components.rest import RESOURCE_SCHEMA
from homeassistant.components.rest.data import RestData
from homeassistant.components import rest, scrape, automation

from homeassistant.helpers.template import Template

_LOGGER = logging.getLogger(__name__)


def verify_voice_profile_permissions(user_input, domain=None, entity_id=None):
    """Verify that the voice profile has permission to access the specified domain or entity.
    
    This provides an additional layer of security beyond the middleware checks.
    
    Args:
        user_input: The conversation user input object
        domain: The domain to check permissions for
        entity_id: The entity_id to check permissions for
        
    Returns:
        Tuple of (authorized: bool, profile_name: str)
        
    Raises:
        UnauthorizedAccess: If access is not authorized
    """
    from .const import DEFAULT_VOICE_USERS
    
    # Extract speech metadata from different possible sources
    profile_name = "default"  # Default to default profile instead of unknown
    speech_metadata = {}
    
    # First try to get metadata from the user_input object
    if hasattr(user_input, "metadata") and user_input.metadata:
        # Direct metadata from conversation input
        speech_metadata = user_input.metadata.get("speech_metadata", {})
    
    # If no metadata found, try to see if user_input itself is the metadata
    # This handles cases where the function is called with the metadata dictionary directly
    if not speech_metadata and isinstance(user_input, dict) and "speech_metadata" in user_input:
        speech_metadata = user_input.get("speech_metadata", {})
    elif not speech_metadata and isinstance(user_input, dict) and "profile_name" in user_input:
        # The input might be the speech_metadata itself
        speech_metadata = user_input
    
    # If we still don't have metadata, check for user_input.current_auth_metadata
    if not speech_metadata and hasattr(user_input, "current_auth_metadata"):
        speech_metadata = user_input.current_auth_metadata
    
    # Log if we didn't find any metadata and default to default profile
    if not speech_metadata:
        _LOGGER.info(f"No speech metadata found, defaulting to 'default' profile permissions")
        default_profile = DEFAULT_VOICE_USERS.get("default", {})
        return check_permissions_for_profile("default", default_profile, domain, entity_id)
    
    # Get voice profile information from metadata
    profile_name = speech_metadata.get("profile_name", "default")
    authenticated = speech_metadata.get("authenticated", False)
    user_profile = speech_metadata.get("user_profile", {})
    
    # If critical data is missing, use default profile
    if not user_profile:
        _LOGGER.info(f"Missing user profile for '{profile_name}', using default profile")
        default_profile = DEFAULT_VOICE_USERS.get("default", {})
        return check_permissions_for_profile("default", default_profile, domain, entity_id)
    
    # For non-default profiles with low confidence, fall back to default profile
    if profile_name != "default" and not authenticated:
        _LOGGER.info(f"Profile '{profile_name}' not authenticated, checking default profile permissions")
        default_profile = DEFAULT_VOICE_USERS.get("default", {})
        authorized, _ = check_permissions_for_profile("default", default_profile, domain, entity_id)
        
        # If default profile allows access, grant it even for unauthenticated users
        if authorized:
            _LOGGER.info(f"Default profile allows access to {domain or entity_id}, granting access")
            return (True, "default")
    
    # Check permissions using the actual profile
    return check_permissions_for_profile(profile_name, user_profile, domain, entity_id)


def check_permissions_for_profile(profile_name, user_profile, domain=None, entity_id=None):
    """Check if the given user profile has permission for the specified domain or entity.
    
    Args:
        profile_name: The name of the profile
        user_profile: The user profile dictionary
        domain: The domain to check permissions for
        entity_id: The entity_id to check permissions for
        
    Returns:
        Tuple of (authorized: bool, profile_name: str)
    """
    # Get permissions from the profile
    permissions = user_profile.get("permissions", {})
    allow = permissions.get("allow", {})
    deny = permissions.get("deny", {})
    
    # Domain-specific authorization check
    if domain:
        # Check if domain is explicitly denied
        denied_domains = deny.get("domains", [])
        if domain in denied_domains:
            _LOGGER.warning(f"Domain '{domain}' is explicitly denied for profile '{profile_name}'")
            return (False, profile_name)
            
        # Check if domain is explicitly allowed ("*" means all domains)
        allowed_domains = allow.get("domains", [])
        if allowed_domains == "*" or (isinstance(allowed_domains, list) and domain in allowed_domains):
            _LOGGER.info(f"Domain '{domain}' is allowed for profile '{profile_name}'")
            return (True, profile_name)
        else:
            _LOGGER.warning(f"Domain '{domain}' is not in allowed domains for profile '{profile_name}'")
            return (False, profile_name)
    
    # Entity-specific authorization check
    if entity_id:
        # Extract domain from entity_id
        entity_domain = entity_id.split(".")[0] if "." in entity_id else ""
        
        # Check if entity is explicitly denied
        denied_entities = deny.get("entities", [])
        if entity_id in denied_entities:
            _LOGGER.warning(f"Entity '{entity_id}' is explicitly denied for profile '{profile_name}'")
            return (False, profile_name)
            
        # Check if entity's domain is explicitly denied
        denied_domains = deny.get("domains", [])
        if entity_domain and entity_domain in denied_domains:
            _LOGGER.warning(f"Entity '{entity_id}' belongs to denied domain '{entity_domain}' for profile '{profile_name}'")
            return (False, profile_name)
            
        # Check if entity is explicitly allowed
        allowed_entities = allow.get("entities", [])
        if entity_id in allowed_entities:
            _LOGGER.info(f"Entity '{entity_id}' is explicitly allowed for profile '{profile_name}'")
            return (True, profile_name)
            
        # Check if entity's domain is allowed
        allowed_domains = allow.get("domains", [])
        if allowed_domains == "*" or (isinstance(allowed_domains, list) and entity_domain in allowed_domains):
            _LOGGER.info(f"Entity '{entity_id}' belongs to allowed domain '{entity_domain}' for profile '{profile_name}'")
            return (True, profile_name)
        else:
            _LOGGER.warning(f"Entity '{entity_id}' domain '{entity_domain}' is not allowed for profile '{profile_name}'")
            return (False, profile_name)
    
    # Default to denying access
    return (False, profile_name)


AZURE_DOMAIN_PATTERN = r"\.(openai\.azure\.com|azure-api\.net)"


def get_function_executor(value: str):
    function_executor = FUNCTION_EXECUTORS.get(value)
    if function_executor is None:
        raise FunctionNotFound(value)
    return function_executor


def is_azure(base_url: str):
    if base_url and re.search(AZURE_DOMAIN_PATTERN, base_url):
        return True
    return False


def convert_to_template(
    settings,
    template_keys=["data", "event_data", "target", "service"],
    hass: HomeAssistant | None = None,
):
    _convert_to_template(settings, template_keys, hass, [])


def _convert_to_template(settings, template_keys, hass, parents: list[str]):
    if isinstance(settings, dict):
        for key, value in settings.items():
            if isinstance(value, str) and (
                key in template_keys or set(parents).intersection(template_keys)
            ):
                settings[key] = Template(value, hass)
            if isinstance(value, dict):
                parents.append(key)
                _convert_to_template(value, template_keys, hass, parents)
                parents.pop()
            if isinstance(value, list):
                parents.append(key)
                for item in value:
                    _convert_to_template(item, template_keys, hass, parents)
                parents.pop()
    if isinstance(settings, list):
        for setting in settings:
            _convert_to_template(setting, template_keys, hass, parents)


def _get_rest_data(hass, rest_config, arguments):
    rest_config.setdefault(CONF_METHOD, rest.const.DEFAULT_METHOD)
    rest_config.setdefault(CONF_VERIFY_SSL, rest.const.DEFAULT_VERIFY_SSL)
    rest_config.setdefault(CONF_TIMEOUT, rest.data.DEFAULT_TIMEOUT)
    rest_config.setdefault(rest.const.CONF_ENCODING, rest.const.DEFAULT_ENCODING)

    resource_template: Template | None = rest_config.get(CONF_RESOURCE_TEMPLATE)
    if resource_template is not None:
        rest_config.pop(CONF_RESOURCE_TEMPLATE)
        rest_config[CONF_RESOURCE] = resource_template.async_render(
            arguments, parse_result=False
        )

    payload_template: Template | None = rest_config.get(CONF_PAYLOAD_TEMPLATE)
    if payload_template is not None:
        rest_config.pop(CONF_PAYLOAD_TEMPLATE)
        rest_config[CONF_PAYLOAD] = payload_template.async_render(
            arguments, parse_result=False
        )

    return rest.create_rest_data_from_config(hass, rest_config)


async def validate_authentication(
    hass: HomeAssistant,
    api_key: str,
    base_url: str,
    api_version: str,
    organization: str = None,
    skip_authentication=False,
) -> None:
    if skip_authentication:
        return

    if is_azure(base_url):
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            organization=organization,
            http_client=get_async_client(hass),
        )
    else:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            http_client=get_async_client(hass),
        )

    await hass.async_add_executor_job(partial(client.models.list, timeout=10))


async def get_domain_entity_attributes(
    hass: HomeAssistant,
    domain: str,
    exposed_entities,
    attributes=None,
):
    """Retrieve attributes for all entities in a specific domain.
    
    Args:
        hass: The Home Assistant instance.
        domain: The domain for which to get attributes.
        exposed_entities: A list of exposed entities.
        attributes: Optional list of specific attributes to include.
        
    Returns:
        A dictionary mapping entity_id to attributes for entities in the specified domain.
    """
    # Filter entities by the specified domain
    domain_entities = [
        entity for entity in exposed_entities 
        if entity["entity_id"].split(".")[0] == domain
    ]
    
    if not domain_entities:
        return {}
    
    result = {}
    for entity in domain_entities:
        entity_id = entity["entity_id"]
        state = hass.states.get(entity_id)
        
        if state:
            # Get only specified attributes or all attributes
            if attributes:
                entity_attrs = {
                    attr: state.attributes.get(attr)
                    for attr in attributes
                    if attr in state.attributes
                }
            else:
                entity_attrs = dict(state.attributes)
                
            # Add the current state
            entity_attrs["state"] = state.state
            
            # Add to result
            result[entity_id] = entity_attrs
            
    return result


class FunctionExecutor(ABC):
    def __init__(self, data_schema=vol.Schema({})) -> None:
        """initialize function executor"""
        self.data_schema = data_schema.extend({vol.Required("type"): str})

    def to_arguments(self, arguments):
        """to_arguments function"""
        try:
            return self.data_schema(arguments)
        except vol.error.Error as e:
            function_type = next(
                (key for key, value in FUNCTION_EXECUTORS.items() if value == self),
                None,
            )
            raise InvalidFunction(function_type) from e

    def validate_entity_ids(self, hass: HomeAssistant, entity_ids, exposed_entities):
        if any(hass.states.get(entity_id) is None for entity_id in entity_ids):
            raise EntityNotFound(entity_ids)
        exposed_entity_ids = map(lambda e: e["entity_id"], exposed_entities)
        if not set(entity_ids).issubset(exposed_entity_ids):
            raise EntityNotExposed(entity_ids)

    @abstractmethod
    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        """execute function"""


class NativeFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize native function"""
        super().__init__(vol.Schema({vol.Required("name"): str}))

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        name = function["name"]
        if name == "execute_service":
            return await self.execute_service(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "execute_service_single":
            return await self.execute_service_single(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "add_automation":
            return await self.add_automation(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "get_history":
            return await self.get_history(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "get_energy":
            return await self.get_energy(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "get_statistics":
            return await self.get_statistics(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "get_user_from_user_id":
            return await self.get_user_from_user_id(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "get_shopping_list_items":
            return await self.get_shopping_list_items(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "add_shopping_list_item":
            return await self.add_shopping_list_item(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "complete_shopping_list_item":
            return await self.complete_shopping_list_item(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "complete_all_shopping_list_items":
            return await self.complete_all_shopping_list_items(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "remove_shopping_list_item":
            return await self.remove_shopping_list_item(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "clear_shopping_list":
            return await self.clear_shopping_list(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "get_entity_attributes":
            return await self.get_entity_attributes(
                hass, function, arguments, user_input, exposed_entities
            )

        raise NativeNotFound(name)

    async def execute_service_single(
        self,
        hass: HomeAssistant,
        function,
        service_argument,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        domain = service_argument["domain"]
        service = service_argument["service"]
        service_data = service_argument.get(
            "service_data", service_argument.get("data", {})
        )
        entity_id = service_data.get("entity_id", service_argument.get("entity_id"))
        area_id = service_data.get("area_id")
        device_id = service_data.get("device_id")

        if isinstance(entity_id, str):
            entity_id = [e.strip() for e in entity_id.split(",")]
        service_data["entity_id"] = entity_id

        if entity_id is None and area_id is None and device_id is None:
            raise CallServiceError(domain, service, service_data)
        if not hass.services.has_service(domain, service):
            raise ServiceNotFound(domain, service)
        self.validate_entity_ids(hass, entity_id or [], exposed_entities)
        
        # Additional security check based on voice profile authorization
        # This provides defense-in-depth beyond the middleware
        is_authorized = True
        profile_name = "default"
        
        # Extract or create metadata for verification if none exists
        metadata_for_verification = None
        
        # Check if we have a direct metadata object from the agent
        if hasattr(user_input, "metadata") and user_input.metadata and "speech_metadata" in user_input.metadata:
            # Use from conversation input
            metadata_for_verification = user_input
            _LOGGER.debug(f"Using metadata from conversation input for authorization check")
        elif hasattr(user_input, "current_auth_metadata"):
            # This might be an OpenAI agent with auth metadata
            metadata_for_verification = {"speech_metadata": user_input.current_auth_metadata}
            _LOGGER.debug(f"Using current_auth_metadata from agent for authorization check")
        else:
            # Try to find auth agent from local data
            from inspect import currentframe, getouterframes
            caller_frame = getouterframes(currentframe(), 2)
            caller_self = None
            
            # Try to get the agent instance from the caller frame
            for frame in caller_frame:
                if 'self' in frame.frame.f_locals and hasattr(frame.frame.f_locals['self'], 'current_auth_metadata'):
                    caller_self = frame.frame.f_locals['self']
                    _LOGGER.debug(f"Found agent in caller frame with auth metadata")
                    break
                    
            # Use metadata from the agent if found
            if caller_self and hasattr(caller_self, 'current_auth_metadata'):
                metadata_for_verification = {"speech_metadata": caller_self.current_auth_metadata}
                _LOGGER.debug(f"Using metadata from caller agent: {caller_self.current_auth_metadata.get('profile_name', 'unknown')}")
        
        # First verify domain-level access
        domain_authorized, profile_name = verify_voice_profile_permissions(
            metadata_for_verification or user_input, domain=domain
        )
        
        if not domain_authorized:
            _LOGGER.warning(f"\ud83d\udd12 DOMAIN ACCESS DENIED: Profile '{profile_name}' not authorized for domain '{domain}'")
            raise UnauthorizedAccess(profile_name, "domain", domain)
        
        # Then verify entity-level access for each entity if applicable
        if entity_id:
            for eid in entity_id:
                entity_authorized, _ = verify_voice_profile_permissions(
                    metadata_for_verification or user_input, entity_id=eid
                )
                if not entity_authorized:
                    _LOGGER.warning(f"\ud83d\udd12 ENTITY ACCESS DENIED: Profile '{profile_name}' not authorized for entity '{eid}'")
                    raise UnauthorizedAccess(profile_name, "entity", eid)
        
        _LOGGER.info(f"Authorization check passed for profile '{profile_name}', domain '{domain}'")
        
        try:
            await hass.services.async_call(
                domain=domain,
                service=service,
                service_data=service_data,
            )
            return {"success": True}
        except HomeAssistantError as e:
            _LOGGER.error(e)
            return {"error": str(e)}

    async def execute_service(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        result = []
        for service_argument in arguments.get("list", []):
            result.append(
                await self.execute_service_single(
                    hass, function, service_argument, user_input, exposed_entities
                )
            )
        return result

    async def add_automation(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        automation_config = yaml.safe_load(arguments["automation_config"])
        config = {"id": str(round(time.time() * 1000))}
        if isinstance(automation_config, list):
            config.update(automation_config[0])
        if isinstance(automation_config, dict):
            config.update(automation_config)

        await _async_validate_config_item(hass, config, True, False)

        automations = [config]
        with open(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH),
            "r",
            encoding="utf-8",
        ) as f:
            current_automations = yaml.safe_load(f.read())

        with open(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH),
            "a" if current_automations else "w",
            encoding="utf-8",
        ) as f:
            raw_config = yaml.dump(automations, allow_unicode=True, sort_keys=False)
            f.write("\n" + raw_config)

        await hass.services.async_call(automation.config.DOMAIN, SERVICE_RELOAD)
        hass.bus.async_fire(
            EVENT_AUTOMATION_REGISTERED,
            {"automation_config": config, "raw_config": raw_config},
        )
        return "Success"

    async def get_history(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        start_time = arguments.get("start_time")
        end_time = arguments.get("end_time")
        entity_ids = arguments.get("entity_ids", [])
        include_start_time_state = arguments.get("include_start_time_state", True)
        significant_changes_only = arguments.get("significant_changes_only", True)
        minimal_response = arguments.get("minimal_response", True)
        include_attributes = arguments.get("include_attributes", False)
        no_attributes = not include_attributes

        now = dt_util.utcnow()
        one_day = timedelta(days=1)
        start_time = self.as_utc(start_time, now - one_day, "start_time not valid")
        end_time = self.as_utc(end_time, start_time + one_day, "end_time not valid")

        self.validate_entity_ids(hass, entity_ids, exposed_entities)

        with recorder.util.session_scope(hass=hass, read_only=True) as session:
            history_instance = recorder.get_instance(hass)

            if significant_changes_only:
                history_func = recorder.history.get_significant_states_with_session
                result = await history_instance.async_add_executor_job(
                    history_func,
                    hass,
                    session,
                    start_time,
                    end_time,
                    entity_ids,
                    None,
                    include_start_time_state,
                    significant_changes_only,
                    minimal_response,
                    no_attributes,
                )
            else:
                history_func = recorder.history.get_states_with_session
                result = await history_instance.async_add_executor_job(
                    history_func,
                    hass,
                    session,
                    start_time,
                    end_time,
                    entity_ids,
                    include_start_time_state,
                )

        return [[self.as_dict(item) for item in sublist] for sublist in result.values()]

    async def get_energy(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        energy_manager: energy.data.EnergyManager = await energy.async_get_manager(hass)
        return energy_manager.data

    async def get_user_from_user_id(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        user = await hass.auth.async_get_user(user_input.context.user_id)
        return {'name': user.name if user and hasattr(user, 'name') else 'Unknown'}

    async def get_statistics(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        statistic_ids = arguments.get("statistic_ids", [])
        start_time = dt_util.as_utc(dt_util.parse_datetime(arguments["start_time"]))
        end_time = dt_util.as_utc(dt_util.parse_datetime(arguments["end_time"]))

        return await recorder.get_instance(hass).async_add_executor_job(
            recorder.statistics.statistics_during_period,
            hass,
            start_time,
            end_time,
            statistic_ids,
            arguments.get("period", "day"),
            arguments.get("units"),
            arguments.get("types", {"change"}),
        )

    async def get_shopping_list_items(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        """Retrieve all non-completed items from the shopping list.
        
        Args:
            hass: The Home Assistant instance.
            function: The function configuration.
            arguments: Arguments passed to the function.
            user_input: The conversation input.
            exposed_entities: List of entities exposed to the conversation.
            
        Returns:
            A dictionary containing the shopping list items or error.
        """
        try:
            # Try to read shopping list data directly from the file
            _LOGGER.debug("Getting shopping list items from the JSON file")
            
            # The shopping list is typically stored in .shopping_list.json
            shopping_list_path = hass.config.path('.shopping_list.json')
            
            # Use a helper function to read the file in the executor
            def _read_shopping_list(path):
                if not os.path.exists(path):
                    return None
                with open(path, 'r') as f:
                    return json.loads(f.read())
            
            # Run file I/O in the executor to prevent blocking the event loop
            items = await hass.async_add_executor_job(_read_shopping_list, shopping_list_path)
            
            if items is None:
                _LOGGER.warning("Shopping list file not found at %s", shopping_list_path)
                return {"error": "unavailable"}
                
            # Filter out completed items
            active_items = [item["name"] for item in items if not item.get("complete", False)]
            
            return {"items": active_items}
                
        except Exception as err:
            _LOGGER.warning("Error accessing shopping list file: %s", err)
            return {"error": "unavailable"}

    async def get_entity_attributes(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        """Get attributes for multiple entities in one call."""
        entity_ids = arguments.get("entity_ids", [])
        specific_attributes = arguments.get("attributes", [])
        
        self.validate_entity_ids(hass, entity_ids, exposed_entities)
        
        result = {}
        for entity_id in entity_ids:
            state = hass.states.get(entity_id)
            if state:
                if specific_attributes:
                    # Only include requested attributes
                    attributes = {}
                    for attr in specific_attributes:
                        if attr in state.attributes:
                            attributes[attr] = state.attributes[attr]
                    result[entity_id] = {
                        "state": state.state,
                        "attributes": attributes
                    }
                else:
                    # Include all attributes
                    result[entity_id] = {
                        "state": state.state,
                        "attributes": dict(state.attributes)
                    }
        
        return result

    async def add_shopping_list_item(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        """Add an item to the shopping list.
        
        Args:
            hass: The Home Assistant instance.
            function: The function configuration.
            arguments: Arguments passed to the function (should contain 'item' key).
            user_input: The conversation input.
            exposed_entities: List of entities exposed to the conversation.
            
        Returns:
            A dictionary indicating success or error.
        """
        try:
            item_text = arguments.get("item", "")
            if not item_text:
                return {"error": "No item specified"}
                
            _LOGGER.debug("Original item text: %s", item_text)
            
            # Using a completely different approach
            # 1. Split by common separators
            separators = [",", " and ", " & "]
            items = []
            
            # Handle special case for "milk and eggs" in a more direct way
            remaining_text = item_text
            for separator in separators:
                if separator in remaining_text:
                    parts = remaining_text.split(separator)
                    # Add all parts except the last one
                    for part in parts[:-1]:
                        if part.strip():
                            items.append(part.strip())
                    # Keep the last part for further processing
                    remaining_text = parts[-1].strip()
            
            # Add the remaining text if it exists
            if remaining_text:
                items.append(remaining_text)
            
            _LOGGER.debug("Parsed items: %s", items)
            
            # If we didn't find any separators but have multiple words that don't match common phrases, 
            # split by words
            if len(items) == 1 and len(items[0].split()) > 2:
                # List of common multi-word items that should not be split
                common_phrases = [
                    "peanut butter", "ice cream", "olive oil", "toilet paper",
                    "paper towels", "dish soap", "laundry detergent", "orange juice"
                ]
                
                # Check if our item matches any common phrases
                is_common_phrase = False
                for phrase in common_phrases:
                    if phrase in items[0].lower():
                        is_common_phrase = True
                        break
                        
                if not is_common_phrase:
                    # Split the single item by words
                    words = items[0].split()
                    items = []
                    for word in words:
                        if word.strip():
                            items.append(word.strip())
            
            if not items:
                return {"error": "No valid items specified"}
            
            _LOGGER.debug("Final items to add: %s", items)
            
            added_items = []
            
            # Add each item using the Home Assistant service
            for item_name in items:
                await hass.services.async_call(
                    domain="shopping_list",
                    service="add_item",
                    service_data={"name": item_name}
                )
                added_items.append(item_name)
                _LOGGER.debug("Added item to shopping list: %s", item_name)
            
            if len(added_items) == 1:
                return {"success": True, "added": added_items[0]}
            else:
                return {"success": True, "added": added_items}
                
        except Exception as err:
            _LOGGER.warning("Error adding shopping list item: %s", err)
            return {"error": "Failed to add item"}

    async def complete_shopping_list_item(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        """Complete an item in the shopping list.
        
        Args:
            hass: The Home Assistant instance.
            function: The function configuration.
            arguments: Arguments passed to the function (should contain 'item' key).
            user_input: The conversation input.
            exposed_entities: List of entities exposed to the conversation.
            
        Returns:
            A dictionary indicating success or error.
        """
        try:
            item_name = arguments.get("item", "").lower()
            if not item_name:
                return {"error": "No item specified"}
                
            # Get the shopping list path to read current items
            shopping_list_path = hass.config.path('.shopping_list.json')
            
            # Helper functions to perform file I/O in executor
            def _read_shopping_list(path):
                if not os.path.exists(path):
                    return None
                with open(path, 'r') as f:
                    return json.loads(f.read())
                    
            def _write_shopping_list(path, data):
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            # Run file read in executor
            items = await hass.async_add_executor_job(_read_shopping_list, shopping_list_path)
            
            if items is None:
                _LOGGER.warning("Shopping list file not found at %s", shopping_list_path)
                return {"error": "unavailable"}
                
            # Find items that match (case insensitive)
            found = False
            completed_item = None
            
            for item in items:
                if item["name"].lower() == item_name and not item.get("complete", False):
                    # Mark as completed directly in the items list
                    item["complete"] = True
                    found = True
                    completed_item = item["name"]
                    break
            
            if not found:
                return {"error": f"Item '{item_name}' not found or already completed"}
            
            # Write the updated list back (in executor)
            await hass.async_add_executor_job(_write_shopping_list, shopping_list_path, items)
                
            # Try to use the Home Assistant service as a backup
            try:
                await hass.services.async_call(
                    domain="shopping_list",
                    service="complete_item",
                    service_data={"name": completed_item}
                )
            except Exception as service_err:
                _LOGGER.warning("Could not use service to complete item: %s", service_err)
                # Continue anyway since we've already updated the file
            
            # Force reload
            await hass.services.async_call(
                domain="homeassistant",
                service="reload_config_entry",
                service_data={"domain": "shopping_list"}
            )
            
            _LOGGER.info("Marked item as complete: %s", completed_item)
            return {"success": True, "completed": completed_item}
                
        except Exception as err:
            _LOGGER.warning("Error completing shopping list item: %s", err)
            return {"error": f"Failed to complete item: {str(err)}"}

    async def complete_all_shopping_list_items(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        """Complete all items in the shopping list.
        
        Args:
            hass: The Home Assistant instance.
            function: The function configuration.
            arguments: Arguments passed to the function.
            user_input: The conversation input.
            exposed_entities: List of entities exposed to the conversation.
            
        Returns:
            A dictionary indicating success or error.
        """
        try:
            # Get the shopping list path
            shopping_list_path = hass.config.path('.shopping_list.json')
            
            # Helper functions to perform file I/O in executor
            def _read_shopping_list(path):
                if not os.path.exists(path):
                    return None
                with open(path, 'r') as f:
                    return json.loads(f.read())
                    
            def _write_shopping_list(path, data):
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
                    f.flush()  # Ensure data is written to disk
            
            # Run file read in executor
            items = await hass.async_add_executor_job(_read_shopping_list, shopping_list_path)
            
            if items is None:
                _LOGGER.warning("Shopping list file not found at %s", shopping_list_path)
                return {"error": "unavailable"}
            
            # Count active items and mark them as complete
            active_items = []
            modified = False
            
            for item in items:
                if not item.get("complete", False):
                    item["complete"] = True
                    active_items.append(item["name"])
                    modified = True
            
            count = len(active_items)
            
            if count == 0:
                return {"success": True, "count": 0, "message": "No active items to complete in shopping list"}
            
            # Write back to the file if we modified anything
            if modified:
                await hass.async_add_executor_job(_write_shopping_list, shopping_list_path, items)
            
            # Try to use the Home Assistant services as a backup
            success = True
            try:
                # Mark all items as completed using the service
                for item_name in active_items:
                    try:
                        await hass.services.async_call(
                            domain="shopping_list",
                            service="complete_item",
                            service_data={"name": item_name}
                        )
                    except Exception as e:
                        _LOGGER.debug("Error completing item %s via service: %s", item_name, e)
                        success = False
                
                # Force a reload if any service calls failed
                if not success:
                    try:
                        await hass.services.async_call(
                            domain="homeassistant",
                            service="reload_config_entry",
                            service_data={"domain": "shopping_list"}
                        )
                    except Exception as reload_err:
                        _LOGGER.debug("Failed to reload shopping list: %s", reload_err)
            except Exception as e:
                _LOGGER.debug("Error using shopping_list services: %s", e)
            
            # If all else fails, try a refresh via event
            if not success:
                try:
                    hass.bus.async_fire("shopping_list_updated")
                except Exception:
                    pass
            
            _LOGGER.info("Completed all shopping list items: %d items", count)
            return {"success": True, "count": count, "completed": active_items}
            
        except Exception as err:
            _LOGGER.warning("Error completing all shopping list items: %s", err)
            return {"error": f"Failed to complete all shopping list items: {str(err)}"}

    async def remove_shopping_list_item(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        """Remove an item from the shopping list permanently.
        
        Args:
            hass: The Home Assistant instance.
            function: The function configuration.
            arguments: Arguments passed to the function (should contain 'item' key).
            user_input: The conversation input.
            exposed_entities: List of entities exposed to the conversation.
            
        Returns:
            A dictionary indicating success or error.
        """
        try:
            item_name = arguments.get("item", "").lower()
            if not item_name:
                return {"error": "No item specified"}
            
            # First, check if the item exists in the shopping list
            shopping_list_path = hass.config.path('.shopping_list.json')
            
            # Helper functions to perform file I/O in executor
            def _read_shopping_list(path):
                if not os.path.exists(path):
                    return None
                with open(path, 'r') as f:
                    return json.loads(f.read())
                    
            def _write_shopping_list(path, data):
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            # Run file read in executor
            items = await hass.async_add_executor_job(_read_shopping_list, shopping_list_path)
            
            if items is None:
                _LOGGER.warning("Shopping list file not found at %s", shopping_list_path)
                return {"error": "unavailable"}
            
            # Find matching items
            items_to_remove = []
            found = False
            
            for item in items:
                if item["name"].lower() == item_name:
                    items_to_remove.append(item)
                    found = True
            
            if not found:
                return {"error": f"Item '{item_name}' not found in shopping list"}
            
            # Now call Home Assistant's services to remove the item
            # First, remove it from the list in memory
            updated_items = [item for item in items if item["name"].lower() != item_name]
            
            # Write the updated list back to the file (in executor)
            await hass.async_add_executor_job(_write_shopping_list, shopping_list_path, updated_items)
            
            # Force a reload of the shopping list component
            await hass.services.async_call(
                domain="homeassistant",
                service="reload_config_entry",
                service_data={"domain": "shopping_list"}
            )
            
            _LOGGER.info("Removed shopping list item: %s", item_name)
            return {"success": True, "removed": item_name}
            
        except Exception as err:
            _LOGGER.warning("Error removing shopping list item: %s", err)
            return {"error": f"Failed to remove item: {str(err)}"}

    async def clear_shopping_list(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        """Clear all items from the shopping list permanently.
        
        Args:
            hass: The Home Assistant instance.
            function: The function configuration.
            arguments: Arguments passed to the function.
            user_input: The conversation input.
            exposed_entities: List of entities exposed to the conversation.
            
        Returns:
            A dictionary indicating success or error.
        """
        try:
            # Get the shopping list path
            shopping_list_path = hass.config.path('.shopping_list.json')
            
            # Helper functions to perform file I/O in executor
            def _read_shopping_list(path):
                if not os.path.exists(path):
                    return None
                with open(path, 'r') as f:
                    return json.loads(f.read())
                    
            def _write_shopping_list(path, data):
                with open(path, 'w') as f:
                    f.write("[]")  # Write directly to avoid any potential JSON serialization issues
                    f.flush()  # Ensure data is written to disk
            
            # Run file read in executor
            items = await hass.async_add_executor_job(_read_shopping_list, shopping_list_path)
            
            if items is None:
                _LOGGER.warning("Shopping list file not found at %s", shopping_list_path)
                return {"error": "unavailable"}
            
            # Count active items
            active_items = [item for item in items if not item.get("complete", False)]
            count = len(active_items)
            
            if count == 0:
                return {"success": True, "count": 0, "message": "Shopping list is already empty"}
            
            # Approach 1: Try the official Home Assistant services first
            success = False
            
            try:
                # First mark all items as completed
                for item in active_items:
                    try:
                        await hass.services.async_call(
                            domain="shopping_list",
                            service="complete_item",
                            service_data={"name": item["name"]}
                        )
                    except Exception as e:
                        _LOGGER.debug("Error completing item %s: %s", item["name"], e)
                        continue
                
                # Then clear completed items
                await hass.services.async_call(
                    domain="shopping_list",
                    service="clear_completed_items"
                )
                
                # Check if the list is now empty by reading it again
                updated_items = await hass.async_add_executor_job(_read_shopping_list, shopping_list_path)
                if not updated_items or all(item.get("complete", False) for item in updated_items):
                    success = True
            except Exception as e:
                _LOGGER.debug("Error using shopping_list services: %s", e)
            
            # Approach 2: If service calls failed, modify the file directly and reload
            if not success:
                try:
                    # Save an empty array to the file (in executor)
                    await hass.async_add_executor_job(_write_shopping_list, shopping_list_path, None)
                    
                    # Try to reload the component
                    try:
                        # Try various reload methods
                        methods = [
                            {"domain": "shopping_list", "service": "reload"},
                            {"domain": "shopping_list", "service": "refresh"},
                        ]
                        
                        for method in methods:
                            try:
                                await hass.services.async_call(
                                    domain=method["domain"],
                                    service=method["service"]
                                )
                                success = True
                                break
                            except Exception:
                                continue
                    except Exception as reload_err:
                        _LOGGER.debug("Failed to reload shopping list: %s", reload_err)
                except Exception as write_err:
                    _LOGGER.warning("Failed to write empty shopping list file: %s", write_err)
            
            # If all else fails, try a restart of the shopping_list integration
            if not success:
                try:
                    _LOGGER.info("Attempting to restart shopping_list integration")
                    # Fire an event that might trigger a refresh of the integration
                    hass.bus.async_fire("shopping_list_updated")
                except Exception:
                    pass
            
            _LOGGER.info("Cleared shopping list, removed %d items", count)
            return {"success": True, "count": count}
            
        except Exception as err:
            _LOGGER.warning("Error clearing shopping list: %s", err)
            return {"error": f"Failed to clear shopping list: {str(err)}"}

    def as_utc(self, value: str, default_value, parse_error_message: str):
        if value is None:
            return default_value

        parsed_datetime = dt_util.parse_datetime(value)
        if parsed_datetime is None:
            raise HomeAssistantError(parse_error_message)

        return dt_util.as_utc(parsed_datetime)

    def as_dict(self, state: State | dict[str, Any]):
        if isinstance(state, State):
            return state.as_dict()
        return state


class ScriptFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize script function"""
        super().__init__(SCRIPT_ENTITY_SCHEMA)

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        script = Script(
            hass,
            function["sequence"],
            "extended_openai_conversation",
            DOMAIN,
            running_description="[extended_openai_conversation] function",
            logger=_LOGGER,
        )

        result = await script.async_run(
            run_variables=arguments, context=user_input.context
        )
        return result.variables.get("_function_result", "Success")


class TemplateFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize template function"""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("value_template"): cv.template,
                    vol.Optional("parse_result"): bool,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        return function["value_template"].async_render(
            arguments,
            parse_result=function.get("parse_result", False),
        )


class RestFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize Rest function"""
        super().__init__(
            vol.Schema(rest.RESOURCE_SCHEMA).extend(
                {
                    vol.Optional("value_template"): cv.template,
                    vol.Optional("payload_template"): cv.template,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        config = function
        rest_data = _get_rest_data(hass, config, arguments)

        await rest_data.async_update()
        value = rest_data.data_without_xml()
        value_template = config.get(CONF_VALUE_TEMPLATE)

        if value is not None and value_template is not None:
            value = value_template.async_render_with_possible_json_value(
                value, None, arguments
            )

        return value


class ScrapeFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize Scrape function"""
        super().__init__(
            scrape.COMBINED_SCHEMA.extend(
                {
                    vol.Optional("value_template"): cv.template,
                    vol.Optional("payload_template"): cv.template,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        config = function
        rest_data = _get_rest_data(hass, config, arguments)
        coordinator = scrape.coordinator.ScrapeCoordinator(
            hass,
            rest_data,
            scrape.const.DEFAULT_SCAN_INTERVAL,
        )
        await coordinator.async_config_entry_first_refresh()

        new_arguments = dict(arguments)

        for sensor_config in config["sensor"]:
            name: Template = sensor_config.get(CONF_NAME)
            value = self._async_update_from_rest_data(
                coordinator.data, sensor_config, arguments
            )
            new_arguments["value"] = value
            if name:
                new_arguments[name.async_render()] = value

        result = new_arguments["value"]
        value_template = config.get(CONF_VALUE_TEMPLATE)

        if value_template is not None:
            result = value_template.async_render_with_possible_json_value(
                result, None, new_arguments
            )

        return result

    def _async_update_from_rest_data(
        self,
        data: BeautifulSoup,
        sensor_config: dict[str, Any],
        arguments: dict[str, Any],
    ) -> None:
        """Update state from the rest data."""
        value = self._extract_value(data, sensor_config)
        value_template = sensor_config.get(CONF_VALUE_TEMPLATE)

        if value_template is not None:
            value = value_template.async_render_with_possible_json_value(
                value, None, arguments
            )

        return value

    def _extract_value(self, data: BeautifulSoup, sensor_config: dict[str, Any]) -> Any:
        """Parse the html extraction in the executor."""
        value: str | list[str] | None
        select = sensor_config[scrape.const.CONF_SELECT]
        index = sensor_config.get(scrape.const.CONF_INDEX, 0)
        attr = sensor_config.get(CONF_ATTRIBUTE)
        try:
            if attr is not None:
                value = data.select(select)[index][attr]
            else:
                tag = data.select(select)[index]
                if tag.name in ("style", "script", "template"):
                    value = tag.string
                else:
                    value = tag.text
        except IndexError:
            _LOGGER.warning("Index '%s' not found", index)
            value = None
        except KeyError:
            _LOGGER.warning("Attribute '%s' not found", attr)
            value = None
        _LOGGER.debug("Parsed value: %s", value)
        return value


class CompositeFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize composite function"""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("sequence"): vol.All(
                        cv.ensure_list, [self.function_schema]
                    )
                }
            )
        )

    def function_schema(self, value: Any) -> dict:
        """Validate a composite function schema."""
        if not isinstance(value, dict):
            raise vol.Invalid("expected dictionary")

        composite_schema = {
            vol.Optional("response_variable"): str,
            vol.Optional("arguments"): dict,
        }
        function_executor = get_function_executor(value["type"])

        return function_executor.data_schema.extend(composite_schema)(value)

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        config = function
        sequence = config["sequence"]

        for executor_config in sequence:
            function_executor = get_function_executor(executor_config["type"])
            
            # Merge executor_config arguments with the function arguments if provided
            executor_arguments = dict(arguments)
            if "arguments" in executor_config:
                for key, value in executor_config["arguments"].items():
                    if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                        # This is a template, evaluate it
                        template = Template(value, hass)
                        executor_arguments[key] = template.async_render(arguments)
                    else:
                        executor_arguments[key] = value
            
            result = await function_executor.execute(
                hass, executor_config, executor_arguments, user_input, exposed_entities
            )

            response_variable = executor_config.get("response_variable")
            if response_variable:
                arguments[response_variable] = result

        return result


class SqliteFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize sqlite function"""
        super().__init__(
            vol.Schema(
                {
                    vol.Optional("query"): str,
                    vol.Optional("db_url"): str,
                    vol.Optional("single"): bool,
                }
            )
        )

    def is_exposed(self, entity_id, exposed_entities) -> bool:
        return any(
            exposed_entity["entity_id"] == entity_id
            for exposed_entity in exposed_entities
        )

    def is_exposed_entity_in_query(self, query: str, exposed_entities) -> bool:
        exposed_entity_ids = list(
            map(lambda e: f"'{e['entity_id']}'", exposed_entities)
        )
        return any(
            exposed_entity_id in query for exposed_entity_id in exposed_entity_ids
        )

    def raise_error(self, msg="Unexpected error occurred."):
        raise HomeAssistantError(msg)

    def get_default_db_url(self, hass: HomeAssistant) -> str:
        db_file_path = os.path.join(hass.config.config_dir, recorder.DEFAULT_DB_FILE)
        return f"file:{db_file_path}?mode=ro"

    def set_url_read_only(self, url: str) -> str:
        scheme, netloc, path, query_string, fragment = parse.urlsplit(url)
        query_params = parse.parse_qs(query_string)

        query_params["mode"] = ["ro"]
        new_query_string = parse.urlencode(query_params, doseq=True)

        return parse.urlunsplit((scheme, netloc, path, new_query_string, fragment))

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ):
        db_url = self.set_url_read_only(
            function.get("db_url", self.get_default_db_url(hass))
        )
        query = function.get("query", "{{query}}")

        template_arguments = {
            "is_exposed": lambda e: self.is_exposed(e, exposed_entities),
            "is_exposed_entity_in_query": lambda q: self.is_exposed_entity_in_query(
                q, exposed_entities
            ),
            "exposed_entities": exposed_entities,
            "raise": self.raise_error,
        }
        template_arguments.update(arguments)

        q = Template(query, hass).async_render(template_arguments)
        _LOGGER.info("Rendered query: %s", q)

        with sqlite3.connect(db_url, uri=True) as conn:
            cursor = conn.cursor().execute(q)
            names = [description[0] for description in cursor.description]

            if function.get("single") is True:
                row = cursor.fetchone()
                return {name: val for name, val in zip(names, row)}

            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({name: val for name, val in zip(names, row)})
            return result


FUNCTION_EXECUTORS: dict[str, FunctionExecutor] = {
    "native": NativeFunctionExecutor(),
    "script": ScriptFunctionExecutor(),
    "template": TemplateFunctionExecutor(),
    "rest": RestFunctionExecutor(),
    "scrape": ScrapeFunctionExecutor(),
    "composite": CompositeFunctionExecutor(),
    "sqlite": SqliteFunctionExecutor(),
}

def log_openai_interaction(hass: HomeAssistant, request_data, response_data):
    """Log the OpenAI API request and response to a temporary file.
    
    Args:
        hass: The Home Assistant instance.
        request_data: The request data sent to OpenAI.
        response_data: The response data received from OpenAI.
    """
    try:
        log_dir = hass.config.path("tmp")
        log_file = os.path.join(log_dir, "openai_last_interaction.json")
        
        # Create tmp directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create log entry with timestamp
        # Sanitize data for JSON serialization
        def sanitize_for_json(obj):
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items() 
                        if not asyncio.iscoroutine(v)}
            if asyncio.iscoroutine(obj):
                return "<coroutine>"
            # For other types, convert to string
            return str(obj)
            
        # Create sanitized log entry
        log_entry = {
            "timestamp": dt_util.now().isoformat(),
            "request": sanitize_for_json(request_data),
            "response": sanitize_for_json(response_data)
        }
        
        # Make sure we can write to a file from the event loop
        def _write_log_file():
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2)
                
        # Run the file operation in the executor to prevent blocking
        hass.async_add_executor_job(_write_log_file)
        
        _LOGGER.debug("OpenAI interaction logged to %s", log_file)
    except Exception as err:
        _LOGGER.error("Failed to log OpenAI interaction: %s", err)

# Add our own implementation of sanitize_filename
def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing problematic characters."""
    import re
    import unicodedata
    
    # Remove diacritics
    filename = unicodedata.normalize('NFKD', filename)
    filename = ''.join([c for c in filename if not unicodedata.combining(c)])
    
    # Replace spaces and punctuation with underscores
    filename = re.sub(r'[^\w\s-]', '_', filename)
    filename = re.sub(r'[-\s]+', '_', filename)
    
    # Ensure the filename is not too long
    if len(filename) > 255:
        filename = filename[:255]
        
    return filename
