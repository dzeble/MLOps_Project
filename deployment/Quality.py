from pydantic import BaseModel

class WineQuality(BaseModel):
   
   fixed_acidity:         float
   volatile_acidity:      float
   citric_acid:           float
   residual_sugar:        float
   chlorides:             float
   free_sulfur_dioxide:   float
   total_sulfur_dioxide:  float
   density:               float
   pH:                    float
   sulphates:             float
   alcohol:               float
   red_wine:              float