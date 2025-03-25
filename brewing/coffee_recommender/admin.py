from django.contrib import admin
from .models import CoffeeBean, UserPreference

@admin.register(CoffeeBean)
class CoffeeBeanAdmin(admin.ModelAdmin):
    list_display = ('name', 'origin', 'roast_level')
    search_fields = ('name', 'origin')

@admin.register(UserPreference)
class UserPreferenceAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'roast_preference', 'brewing_method', 'created_at')
    search_fields = ('session_id',)
