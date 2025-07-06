"""Temporary file to hold the corrected query method."""

async def query(
    self,
    user_input: conversation.ConversationInput,
    messages,
    exposed_entities,
    n_requests,
) -> OpenAIQueryResponse:
    """Process a sentence."""
    model = self.chat_model
    max_tokens = self.max_tokens
    top_p = self.top_p
    temperature = self.temperature
    use_tools = self.entry.options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS)
    context_threshold = self.context_threshold
    functions = list(map(lambda s: s["spec"], self.get_functions()))

    function_call = "auto"
    if n_requests == self.max_function_calls_per_conversation:
        function_call = "none"

    tool_kwargs = {"functions": functions, "function_call": function_call}
    if use_tools:
        tool_kwargs = {
            "tools": [{"type": "function", "function": func} for func in functions],
            "tool_choice": function_call,
        }

    if len(functions) == 0:
        tool_kwargs = {}

    # Use a safer logging approach to avoid serialization issues
    try:
        # Create a sanitized copy for logging
        safe_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                safe_msg = {}
                for k, v in msg.items():
                    if asyncio.iscoroutine(v):
                        safe_msg[k] = "<coroutine>"  # Replace coroutines with a placeholder
                    else:
                        safe_msg[k] = v
                safe_messages.append(safe_msg)
            else:
                safe_messages.append(str(msg))
        
        _LOGGER.info("Prompt for %s: %s", model, json.dumps(safe_messages))
    except Exception as e:
        _LOGGER.error("Error logging messages: %s", e)

    # Sanitize messages to remove any non-serializable objects
    # Must maintain proper OpenAI message format with role and content
    sanitized_messages = []
    try:
        for msg in messages:
            if isinstance(msg, dict):
                # Ensure each message has required fields
                if 'role' not in msg:
                    _LOGGER.warning("Skipping message without 'role' field: %s", msg)
                    continue
                    
                sanitized_msg = {
                    'role': msg['role']  # Role is required
                }
                
                # Process content field
                if 'content' in msg:
                    if isinstance(msg['content'], (str, type(None))):
                        sanitized_msg['content'] = msg['content']
                    elif asyncio.iscoroutine(msg['content']):
                        _LOGGER.warning("Skipping coroutine in 'content' field")
                        sanitized_msg['content'] = "Content unavailable"
                    else:
                        # Try to convert other content types to string
                        try:
                            sanitized_msg['content'] = str(msg['content'])
                        except Exception:
                            sanitized_msg['content'] = "Content unavailable"
                else:
                    # Content is required by the API
                    sanitized_msg['content'] = ""
                
                # Copy other fields if they're serializable
                for k, v in msg.items():
                    if k not in ['role', 'content']:
                        if asyncio.iscoroutine(v):
                            continue
                        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                            sanitized_msg[k] = v
                        else:
                            # Skip non-serializable values
                            _LOGGER.debug("Skipping non-serializable value for field '%s'", k)
                            
                sanitized_messages.append(sanitized_msg)
            else:
                # Skip non-dict messages - all messages must be objects
                _LOGGER.warning("Skipping non-dict message: %s", msg)
        
        # Ensure we have at least one valid message
        if not sanitized_messages:
            # Add a default system message if no valid messages found
            sanitized_messages.append({
                "role": "system",
                "content": "You are a helpful assistant."
            })
    except Exception as e:
        _LOGGER.error("Error sanitizing messages: %s", e)
        # Fallback to a safe default message
        sanitized_messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]
    
    # Create request data for logging - ensure it's sanitized for JSON serialization
    request_data = {
        "model": model,
        "messages": sanitized_messages,  # Use sanitized messages to avoid serialization issues
        "max_tokens": max_tokens,
        "top_p": top_p,
        "temperature": temperature,
        "user": user_input.conversation_id,
        **tool_kwargs
    }
    
    try:
        # Move the OpenAI API call to an executor to prevent blocking I/O operations
        def execute_openai_request():
            # Create a new event loop for the executor thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Run the async API call in the new event loop
                return loop.run_until_complete(
                    self.client.chat.completions.create(
                        model=model,
                        messages=sanitized_messages,  # Use sanitized messages
                        max_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature,
                        user=user_input.conversation_id,
                        **tool_kwargs,
                    )
                )
            finally:
                loop.close()
        
        # Run the OpenAI API call in an executor to prevent blocking I/O
        response: ChatCompletion = await self.hass.async_add_executor_job(execute_openai_request)
        
        # Log the API interaction to a temp file
        response_dict = response.model_dump(exclude_none=True)
        log_openai_interaction(self.hass, request_data, response_dict)
        
    except OpenAIError as err:
        _LOGGER.error("OpenAI API error: %s", err)
        raise

    _LOGGER.info("Response %s", json.dumps(response.model_dump(exclude_none=True)))

    if response.usage.total_tokens > context_threshold:
        await self.truncate_message_history(messages, exposed_entities, user_input)

    choice: Choice = response.choices[0]
    message = choice.message

    if choice.finish_reason == "function_call":
        _LOGGER.info("Function call detected: %s", message.function_call.name)
        return await self.execute_function_call(
            user_input, messages, message, exposed_entities, n_requests + 1
        )
    if choice.finish_reason == "tool_calls":
        tool_call_names = [tool.function.name for tool in message.tool_calls]
        _LOGGER.info("Tool calls detected: %s", ", ".join(tool_call_names))
        return await self.execute_tool_calls(
            user_input, messages, message, exposed_entities, n_requests + 1
        )
    if choice.finish_reason == "length":
        raise TokenLengthExceededError(response.usage.completion_tokens)

    return OpenAIQueryResponse(response=response, message=message)
