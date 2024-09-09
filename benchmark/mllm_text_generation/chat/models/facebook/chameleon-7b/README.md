- Added a simple chat template that assumes two messages come in: The first one as the system prompt and the second one as the user's real prompt.

```jinja
{{ bos_token }}
{{'USER: ' + messages[0]['content'] + '\n' + messages[1]['content'] + '\nASSISTANT:'}}
```
