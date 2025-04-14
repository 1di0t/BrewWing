"""
WSGI config for brewing project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

import os

try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    # torch 혹은 torch._dynamo가 없으면 무시
    pass

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'brewing.settings')

application = get_wsgi_application()
