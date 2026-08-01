"""
Field & sensor metrics that CANNOT be derived from an RGB photo.

Plant height, LAI, biomass, chlorophyll/nitrogen content, growth stage,
disease incidence/severity, pest infestation level, yield, grain weight,
fruit count, soil moisture/temperature/pH/EC/organic matter, air
temperature, humidity, rainfall, solar radiation, wind speed, elevation,
and slope all require dedicated instruments (rulers, ceptometers,
lab assays, soil probes, weather stations, GPS/DEM) or destructive/manual
ground-truth measurement. This pipeline NEVER fabricates values for these
-- it only stores what the user explicitly enters, clearly labelled as
user-supplied field data, separate from anything the image analysis
computes.
"""
from __future__ import annotations

#: Field name -> (display label, unit) for the manual-entry table.
FIELD_METRICS = [
    ("plant_height", "Plant Height", "cm"),
    ("lai", "Leaf Area Index (LAI)", "m²/m²"),
    ("biomass", "Biomass", "g/m²"),
    ("chlorophyll_content", "Chlorophyll Content", "SPAD"),
    ("nitrogen_content", "Nitrogen Content", "%"),
    ("water_content", "Water Content", "%"),
    ("growth_stage", "Crop Growth Stage", "BBCH/stage"),
    ("disease_incidence", "Disease Incidence", "%"),
    ("disease_severity", "Disease Severity", "0-5 scale"),
    ("pest_infestation", "Pest Infestation Level", "0-5 scale"),
    ("yield_", "Yield", "t/ha"),
    ("grain_weight", "Grain Weight", "g/1000 grains"),
    ("fruit_count", "Fruit Count", "count"),
    ("soil_moisture", "Soil Moisture", "%"),
    ("soil_temperature", "Soil Temperature", "°C"),
    ("air_temperature", "Air Temperature", "°C"),
    ("relative_humidity", "Relative Humidity", "%"),
    ("rainfall", "Rainfall", "mm"),
    ("solar_radiation", "Solar Radiation", "MJ/m²/day"),
    ("wind_speed", "Wind Speed", "m/s"),
    ("soil_ph", "Soil pH", "pH"),
    ("electrical_conductivity", "Electrical Conductivity", "dS/m"),
    ("organic_matter", "Organic Matter", "%"),
    ("elevation", "Elevation", "m"),
    ("slope", "Slope", "%"),
    ("group_label", "Group / Crop Label (for ML metrics)", "text"),
]

FIELD_METRIC_LABELS = [label for _key, label, _unit in FIELD_METRICS]
FIELD_METRIC_KEYS = [key for key, _label, _unit in FIELD_METRICS]


def empty_row() -> dict:
    """A row of field data with every value unset (never pre-filled with guesses)."""
    return {key: "" for key in FIELD_METRIC_KEYS}
