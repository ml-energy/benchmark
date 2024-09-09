- Added empty `bos_token` because TGI v1.4.2 fails to load the local tokenizer configuration file if it's missing.
- Added `chat_template`.

### Chat template

```jinja
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 %}
        {{ system_message }}
    {% endif %}
    {{ '\n\n' + (message['role'] | title) + ': ' + message['content'] }}
{% endfor %}
{% if add_generation_prompt %}
    {{ '\n\nAssistant:' }}
{% endif %}
```
