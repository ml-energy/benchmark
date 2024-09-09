- Added `chat_template` that uses the chat format recommended by phi-2's HuggingFace hub README.

### Chat template

```jinja
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content']  + ' ' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 %}
        {% set content = system_message + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {{ (message['role'] | title) + ': ' + content + '\n' }}
{% endfor %}
{% if add_generation_prompt %}
    {{ 'Assistant:' }}
{% endif %}
```
