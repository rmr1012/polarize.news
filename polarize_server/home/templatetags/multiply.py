from django import template
register = template.Library()

@register.filter
def multiply(value, arg):
    return value*arg

@register.filter
def multiplystr(value, arg):
    return float("%.1f" % (value*arg))

@register.filter
def mapbias(value):
    if value>0.9:
        return "Extreame"
    elif value>0.7:
        return "Highly Biased"
    elif value>0.5:
        return "Biased"
    elif value>0.3:
        return "Slightly Biased"
    elif value>0.15:
        return "Fair"
    else:
        return "Neutral"

@register.filter
def sluggy(value):
    return value.replace(" ","-")
