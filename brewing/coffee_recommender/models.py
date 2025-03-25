from django.db import models

class CoffeeBean(models.Model):
    name = models.CharField(max_length=100)
    origin = models.CharField(max_length=100)
    flavor_profile = models.TextField()
    roast_level = models.CharField(max_length=50)
    description = models.TextField()
    
    def __str__(self):
        return self.name

class UserPreference(models.Model):
    session_id = models.CharField(max_length=100)
    acidity_preference = models.IntegerField(null=True, blank=True)
    bitterness_preference = models.IntegerField(null=True, blank=True)
    roast_preference = models.CharField(max_length=50, null=True, blank=True)
    brewing_method = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Preference {self.id} - {self.session_id}"
