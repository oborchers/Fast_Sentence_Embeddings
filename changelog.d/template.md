{{ "\n" -}}
{%- for section, categories in sections.items() -%}
{%- for category, definition in definitions.items() if category in categories -%}
{{ "### " ~ definition["name"] ~ "\n\n" -}}
{%- for text, issues in categories[category].items() -%}
{{ "- " ~ text ~ "\n\n" -}}
{%- endfor -%}
{%- endfor -%}
{%- endfor -%}
