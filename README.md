# Extended OpenAI Conversation

A Home Assistant custom component enhancing the standard OpenAI Conversation integration with advanced features for smarter home control.

Derived from [OpenAI Conversation](https://www.home-assistant.io/integrations/openai_conversation/) with significant enhancements that provide more context-aware interactions.

## Additional Features

- Ability to call services of Home Assistant
- Ability to create automations dynamically
- Ability to get data from external API or web pages
- Ability to retrieve state history of entities
- Ability to retrieve attributes of multiple entities in a single call
- Domain-aware entity attributes for smart contextual responses
- Option to pass the current user's name to OpenAI via the user message context
- Area support: entity areas are included in the prompt, along with a list of all available areas

## Area Support

The prompt includes area information for each entity and a comprehensive list of all areas in your Home Assistant instance. This allows the AI to:

1. See which area each entity belongs to in the entity listing
2. Access a full list of all areas with their IDs, names, and aliases
3. Make better context-aware decisions about controlling devices based on their location

## Domain-Aware Attributes

The integration automatically includes relevant entity attributes for specific domains based on keywords in the user's query. When you ask about "lights" or mention "brightness," the integration automatically includes detailed attributes of all light entities to provide more context to the AI.

This allows for more intelligent responses without having to explicitly request entity attributes each time. The AI will have access to additional information such as:
- Light brightness levels, color modes, and supported features
- Climate settings like temperature, target temperature, and modes
- Media player states and media information
- And more...

You can configure which domains and keywords should trigger attribute inclusion through the integration options.

## How it works

Extended OpenAI Conversation uses OpenAI API's feature of [function calling](https://platform.openai.com/docs/guides/function-calling) to call services within Home Assistant.

Since OpenAI models understand how to call Home Assistant services in general, you just need to expose your entities to let the model know what devices you have available.

## Installation

### Method 1: HACS (Recommended)

1. Open HACS in your Home Assistant instance
2. Go to "Integrations" tab
3. Click the three dots in the top right corner and select "Custom repositories"
4. Add repository as a new custom repository:
   - Category: Integration
   - Repository: `https://github.com/simonrasmussendk/extended_openai_conversation`
5. Click "ADD"
6. Search for "Extended OpenAI Conversation" in the Integrations tab
7. Click "Download"
8. Restart Home Assistant

### Method 2: Manual Installation

1. Copy the `extended_openai_conversation` folder into your `<config directory>/custom_components` directory
2. Restart Home Assistant

### Configuration

After installation (via either method):

1. Go to Settings > Devices & Services
2. Click the Add Integration button in the bottom right corner
3. Search for and select "Extended OpenAI Conversation"
4. Follow the on-screen instructions to complete setup (API Key required)
   - [How to generate an OpenAI API Key](https://www.home-assistant.io/integrations/openai_conversation/#generate-an-api-key)
   - Specify "Base Url" if using compatible servers like Azure OpenAI, LocalAI, etc.
5. Go to Settings > [Voice Assistants](https://my.home-assistant.io/redirect/voice_assistants/)
6. Click to edit the Assistant (named "Home Assistant" by default)
7. Select "Extended OpenAI Conversation" from the "Conversation agent" tab

## Preparation

After installation, you need to expose the entities you want to control via the conversation interface:
1. Navigate to "http://{your-home-assistant}/config/voice-assistants/expose"
2. Select the entities you wish to make available to the conversation agent

## Configuration
### Options

You can customize the integration's behavior through the Options section. Click the Edit Assist button and configure:

- `Attach Username`: Pass the active user's name to OpenAI via the message payload (for UI/REST API conversations)

- `Maximum Function Calls Per Conversation`: Limit function calls in a single conversation to prevent potential infinite loops

- `Domain Keywords`: Define domain-specific keywords that automatically include attributes for all entities in that domain when detected in user queries

  Example configuration:
  ```yaml
  - domain: light
    keywords:
      - brightness
      - color
      - dim
  - domain: climate
    keywords:
      - temperature
      - thermostat
      - heat
  ```

- `Functions`: A list of mappings between function specifications and implementations

### Supported Function Types

- `native`: Built-in functions provided by the integration:
  - `execute_service`: Call Home Assistant services
  - `get_entity_attributes`: Retrieve attributes of multiple entities
  - `get_domain_entity_attributes`: Get attributes for all entities in a domain
  - `add_automation`: Create automations dynamically
  - `get_history`: Retrieve historical entity data

- `script`: Execute a sequence of Home Assistant services
- `template`: Return templated values
- `rest`: Fetch data from REST API endpoints
- `scrape`: Extract information from websites
- `composite`: Execute a sequence of functions
- `sqlite`: Run queries against the Home Assistant database

Default function configuration example:

```yaml
- spec:
    name: execute_services
    description: Use this function to execute service of devices in Home Assistant.
    parameters:
      type: object
      properties:
        list:
          type: array
          items:
            type: object
            properties:
              domain:
                type: string
                description: The domain of the service
              service:
                type: string
                description: The service to be called
              service_data:
                type: object
                description: The service data object to indicate what to control.
                properties:
                  entity_id:
                    type: string
                    description: The entity_id retrieved from available devices. It must start with domain, followed by dot character.
                required:
                - entity_id
            required:
            - domain
            - service
            - service_data
  function:
    type: native
    name: execute_service
```

## Function Examples

Below are examples of how to configure different types of functions:

### 1. Template Functions

For simple data transformations or fixed responses:

```yaml
- spec:
    name: get_current_weather
    description: Get the current weather in a given location
    parameters:
      type: object
      properties:
        location:
          type: string
          description: The city and state, e.g. San Francisco, CA
        unit:
          type: string
          enum:
          - celcius
          - farenheit
      required:
      - location
  function:
    type: template
    value_template: The temperature in {{ location }} is 25 {{unit}}
```

### 2. Script Functions

For executing Home Assistant services:

```yaml
- spec:
    name: add_item_to_shopping_cart
    description: Add item to shopping cart
    parameters:
      type: object
      properties:
        item:
          type: string
          description: The item to be added to cart
      required:
      - item
  function:
    type: script
    sequence:
    - service: shopping_list.add_item
      data:
        name: '{{item}}'
```

### 3. Native Functions

Get detailed entity attributes:

```yaml
- spec:
    name: get_entity_attributes
    description: Get attributes of multiple Home Assistant entities in one call
    parameters:
      type: object
      properties:
        entity_ids:
          type: array
          items:
            type: string
          description: List of entity IDs to get attributes for
        attributes:
          type: array
          items:
            type: string
          description: Optional list of specific attribute names to retrieve. If not provided, all attributes will be returned.
      required:
      - entity_ids
  function:
    type: native
    name: get_entity_attributes
```

This function is particularly useful for getting detailed capabilities from entities:
- Temperature ranges, modes, and current settings from climate devices
- Color modes and supported features from lights
- Available input sources for media players
- Coordinates and other details from sensors

### 4. Database Functions

Query historical data from Home Assistant's database:

```yaml
- spec:
    name: get_last_updated_time_of_entity
    description: >
      Use this function to get last updated time of entity
    parameters:
      type: object
      properties:
        entity_id:
          type: string
          description: The target entity
  function:
    type: sqlite
    query: >-
      {%- if is_exposed(entity_id) -%}
        SELECT datetime(s.last_updated_ts, 'unixepoch', 'localtime') as last_updated_ts
        FROM states s
          INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id
          INNER JOIN states old ON s.old_state_id = old.state_id
        WHERE sm.entity_id = '{{entity_id}}' AND s.state != old.state ORDER BY s.last_updated_ts DESC LIMIT 1
      {%- else -%}
        {{ raise("entity_id should be exposed.") }}
      {%- endif -%}
```

### Database FAQ

1. Can GPT modify or delete data?
   > No, connections are created in read-only mode, data is only used for fetching.

2. Can GPT query data from unexposed entities?
   > Yes, it's difficult to validate whether a query only uses exposed entities.

3. How do I adjust timezone in queries?
   > Set the "TZ" environment variable to your [region](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) (e.g., `Asia/Seoul`) or use offset adjustments in queries (e.g., `datetime(s.last_updated_ts, 'unixepoch', '+9 hours')`).

## Logging

To monitor API requests and responses, add this to your `configuration.yaml`:

```yaml
logger:
  logs:
    custom_components.extended_openai_conversation: info
```
