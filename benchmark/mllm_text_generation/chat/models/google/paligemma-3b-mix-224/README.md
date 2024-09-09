- Added `chat_template` that just concatenates the system and user prompts.

```jinja
{%- for message in messages -%}
    {{ message['content'] + ' ' }}
{%- endfor -%}
```
