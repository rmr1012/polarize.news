from django import template
register = template.Library()


@register.filter
def tohslh(value):
    return 100-value*100
