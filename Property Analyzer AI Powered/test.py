"""
Property Analyzer: AI-Powered Real Estate Assessment Tool

This is a modified version that works without requiring Google Earth Engine
authentication. It uses simulated data for demonstration purposes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import random

import requests
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.geocoders import Nominatim, GoogleV3, ArcGIS
import time
import os
from dotenv import load_dotenv

load_dotenv()

class PropertyAnalyzer:
    def __init__(self):
        """Initialize the PropertyAnalyzer with required components."""
        try:
            # Initialize geocoder
            self.geolocator = Nominatim(user_agent="property_analyzer")
            
            # Initialize ML model for property valuation
            self.value_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            print("Property Analyzer initialized successfully")
            
        except Exception as e:
            print(f"Initialization error: {e}")
    
    def geocode_address(self, address):
        """Convert address to coordinates."""
        try:
            location = self.geolocator.geocode(address)
            if location:
                return {
                    'lat': location.latitude,
                    'lon': location.longitude,
                    'address': location.address
                }
            else:
                # For demo purposes, generate random coordinates if geocoding fails
                print(f"Could not geocode address: {address}, using simulated location data")
                return {
                    'lat': random.uniform(30, 45),
                    'lon': random.uniform(-120, -70),
                    'address': address
                }
        except Exception as e:
            print(f"Geocoding error: {e}")
            # For demo purposes, generate random coordinates
            print("Using simulated location data")
            return {
                'lat': random.uniform(30, 45),
                'lon': random.uniform(-120, -70),
                'address': address
            }

    def get_property_coordinates(self, address=None, lat=None, lon=None):
        """Get coordinates either from address or direct lat/lon input."""
        if address:
            return self.geocode_address(address)
        elif lat is not None and lon is not None:
            return {'lat': lat, 'lon': lon, 'address': f"Coordinates: {lat}, {lon}"}
        else:
            raise ValueError("Either address or lat/lon coordinates must be provided")

    def analyze_flood_risk(self, location_data, radius_meters=1000):
        """Simulate flood risk based on location data."""
        try:
            # Get coordinates
            lat = location_data['lat']
            lon = location_data['lon']
            
            # For demo purposes, generate simulated data
            # Use latitude to influence the flood risk (more northern = less risk in this simulation)
            lat_factor = (lat - 30) / 15  # Normalize latitude between 30-45 to 0-1
            lat_factor = max(0, min(1, lat_factor))  # Clamp between 0-1
            
            # Use longitude as a random seed for some variability
            random.seed(int((lon + 180) * 100))
            
            # Simulated elevation (higher = lower flood risk)
            mean_elevation = 50 + random.uniform(-20, 100) + lat_factor * 50
            
            # Simulated distance to water (higher = lower flood risk)
            min_distance = 100 + random.uniform(0, 900) - lat_factor * 200
            
            # Simulated water presence
            water_presence = max(0, min(100, 30 - lat_factor * 20 + random.uniform(-10, 10)))
            
            # Calculate risk score
            # Low elevation + close to water = higher risk
            elev_factor = max(0, min(1, mean_elevation / 100))  # 0-100m scale
            dist_factor = max(0, min(1, min_distance / radius_meters))
            
            # Combined risk (0-100 scale, higher is riskier)
            risk_score = int(100 * (1 - (elev_factor * 0.7 + dist_factor * 0.3)))
            
            # Risk level categories
            if risk_score < 20:
                risk_level = "Very Low"
            elif risk_score < 40:
                risk_level = "Low"
            elif risk_score < 60:
                risk_level = "Moderate"
            elif risk_score < 80:
                risk_level = "High"
            else:
                risk_level = "Very High"
            
            return {
                "elevation_m": mean_elevation,
                "distance_to_water_m": min_distance,
                "flood_risk_score": risk_score,
                "flood_risk_level": risk_level,
                "water_presence_pct": water_presence
            }
            
        except Exception as e:
            print(f"Flood risk analysis error: {e}")
            return {"error": str(e)}

    def analyze_sun_exposure(self, location_data):
        """Simulate sun exposure throughout the year."""
        try:
            # Get coordinates
            lat = location_data['lat']
            lon = location_data['lon']
            
            # Use latitude to influence sun exposure (higher latitudes get less sun)
            lat_factor = 1 - abs(lat - 35) / 15  # Best sun around latitude 35, decreasing as you move away
            lat_factor = max(0.5, min(1, lat_factor))  # Clamp between 0.5-1
            
            # Use longitude as a random seed for some variability
            random.seed(int((lon + 180) * 100) + 1)  # Different seed than flood risk
            
            # Sun angles for different times of year
            sun_angles = [
                {"season": "Winter", "time": "Morning", "azimuth": 120, "elevation": 20},
                {"season": "Winter", "time": "Noon", "azimuth": 180, "elevation": 30},
                {"season": "Winter", "time": "Afternoon", "azimuth": 240, "elevation": 20},
                {"season": "Spring/Fall", "time": "Morning", "azimuth": 90, "elevation": 30},
                {"season": "Spring/Fall", "time": "Noon", "azimuth": 180, "elevation": 60},
                {"season": "Spring/Fall", "time": "Afternoon", "azimuth": 270, "elevation": 30},
                {"season": "Summer", "time": "Morning", "azimuth": 60, "elevation": 40},
                {"season": "Summer", "time": "Noon", "azimuth": 180, "elevation": 80},
                {"season": "Summer", "time": "Afternoon", "azimuth": 300, "elevation": 40}
            ]
            
            results = []
            base_score = int(70 * lat_factor + random.uniform(-5, 15))
            
            for angle in sun_angles:
                # Simulate hillshade values
                season_factor = 1.0
                if angle["season"] == "Winter":
                    season_factor = 0.7
                elif angle["season"] == "Summer":
                    season_factor = 1.2
                
                time_factor = 1.0
                if angle["time"] == "Morning":
                    time_factor = 0.9
                elif angle["time"] == "Afternoon":
                    time_factor = 0.95
                
                sun_score = int(base_score * season_factor * time_factor)
                sun_score = max(0, min(100, sun_score + random.uniform(-10, 10)))
                
                results.append({
                    "season": angle["season"],
                    "time_of_day": angle["time"],
                    "sun_exposure_score": sun_score
                })
            
            # Calculate average sun exposure
            valid_scores = [r["sun_exposure_score"] for r in results if r["sun_exposure_score"] is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            
            # Determine overall rating
            if avg_score is not None:
                if avg_score >= 80:
                    rating = "Excellent"
                elif avg_score >= 60:
                    rating = "Good"
                elif avg_score >= 40:
                    rating = "Moderate"
                else:
                    rating = "Poor"
            else:
                rating = "Unknown"
            
            return {
                "detailed_sun_exposure": results,
                "average_sun_score": avg_score,
                "sun_exposure_rating": rating
            }
            
        except Exception as e:
            print(f"Sun exposure analysis error: {e}")
            return {"error": str(e)}

    def analyze_neighborhood_development(self, location_data, years_back=10, radius_km=2):
        """Simulate development trends in the neighborhood."""
        try:
            # Get coordinates
            lat = location_data['lat']
            lon = location_data['lon']
            
            # For demo purposes, generate simulated data
            # Use longitude to influence development (more eastern areas develop faster in this simulation)
            lon_factor = (lon + 120) / 50  # Normalize longitude
            lon_factor = max(0, min(1, lon_factor))  # Clamp between 0-1
            
            # Use latitude as a random seed for some variability
            random.seed(int(lat * 100) + 2)  # Different seed than previous analyses
            
            # Set up time periods to analyze
            current_year = datetime.now().year
            start_year = current_year - years_back
            
            # Generate development trend data
            development_trend = []
            
            # Base growth rate (% per year), influenced by location
            base_growth = 1.5 + lon_factor * 2 + random.uniform(-0.5, 1.5)
            
            # Starting development percentage
            start_pct = 25 + random.uniform(-10, 40)
            
            # Generate data for available years
            available_years = [y for y in range(start_year, current_year + 1, 2)]
            
            for i, year in enumerate(available_years):
                # Calculate built-up percentage with some random variation
                year_factor = i / (len(available_years) - 1) if len(available_years) > 1 else 0.5
                pct = start_pct * (1 + base_growth/100) ** (i * 2)  # Compound growth
                pct = pct + random.uniform(-2, 2)  # Add small random variations
                pct = max(0, min(100, pct))  # Clamp between 0-100
                
                development_trend.append({
                    "year": year,
                    "built_up_percentage": pct
                })
            
            # Calculate development growth rate
            if len(development_trend) >= 2:
                first = development_trend[0]["built_up_percentage"]
                last = development_trend[-1]["built_up_percentage"]
                years_diff = development_trend[-1]["year"] - development_trend[0]["year"]
                
                if years_diff > 0 and first is not None and last is not None and first > 0:
                    annual_growth_rate = ((last / first) ** (1 / years_diff) - 1) * 100
                else:
                    annual_growth_rate = base_growth
            else:
                annual_growth_rate = base_growth
            
            # Determine growth category
            if annual_growth_rate > 5:
                growth_category = "Rapid Growth"
            elif annual_growth_rate > 2:
                growth_category = "Moderate Growth"
            elif annual_growth_rate > 0.5:
                growth_category = "Stable Growth"
            elif annual_growth_rate > -0.5:
                growth_category = "Stable"
            else:
                growth_category = "Declining"
            
            return {
                "development_trend": development_trend,
                "analysis_period": f"{start_year}-{current_year}",
                "annual_growth_rate_pct": annual_growth_rate,
                "development_category": growth_category
            }
            
        except Exception as e:
            print(f"Neighborhood development analysis error: {e}")
            return {"error": str(e)}

    def analyze_traffic_noise(self, location_data, radius_meters=500):
        """Simulate traffic and noise levels."""
        try:
            # Get coordinates
            lat = location_data['lat']
            lon = location_data['lon']
            
            # For demo purposes, generate simulated data
            # Use location to affect noise levels (lower latitudes = higher noise in this simulation)
            # This simulates the idea that more southern areas might be more urban/populated
            lat_factor = 1 - (lat - 30) / 15  # Normalize latitude between 30-45 to 1-0
            lat_factor = max(0, min(1, lat_factor))  # Clamp between 0-1
            
            # Use longitude as a random seed
            random.seed(int((lon + 180) * 100) + 3)  # Different seed than previous analyses
            
            # Simulated distances to different road types
            # Lower latitude = more likely to be close to highways
            highway_factor = lat_factor * 0.7 + random.random() * 0.3
            highway_distance = None if random.random() > highway_factor else int(200 + random.uniform(0, 1500) * (1 - highway_factor))
            
            major_road_factor = lat_factor * 0.5 + 0.3
            major_road_distance = int(100 + random.uniform(0, 500) * (1 - major_road_factor))
            
            minor_road_distance = int(10 + random.uniform(0, 200))
            
            # Simulated road density based on latitude (higher in more populated areas)
            road_density = max(0.1, min(5, lat_factor * 3 + random.uniform(-0.5, 1.5)))
            
            # Calculate noise score
            # Highway factor (0-100, higher means more noise)
            if highway_distance is None or highway_distance > 1000:
                highway_factor = 0
            else:
                highway_factor = max(0, 100 - (highway_distance / 10))
            
            # Major road factor
            if major_road_distance > 500:
                major_road_factor = 0
            else:
                major_road_factor = max(0, 100 - (major_road_distance / 5))
            
            # Minor road factor
            if minor_road_distance > 200:
                minor_road_factor = 0
            else:
                minor_road_factor = max(0, 100 - (minor_road_distance / 2))
            
            # Density factor
            density_factor = min(100, road_density * 20)
            
            # Combined noise score (0-100)
            noise_score = int(0.4 * highway_factor + 0.3 * major_road_factor +
                              0.2 * minor_road_factor + 0.1 * density_factor)
            
            # Determine noise level
            if noise_score < 20:
                noise_level = "Very Low"
            elif noise_score < 40:
                noise_level = "Low"
            elif noise_score < 60:
                noise_level = "Moderate"
            elif noise_score < 80:
                noise_level = "High"
            else:
                noise_level = "Very High"
            
            return {
                "distance_to_highway_m": highway_distance,
                "distance_to_major_road_m": major_road_distance,
                "distance_to_minor_road_m": minor_road_distance,
                "road_density_km_per_km2": road_density,
                "estimated_noise_score": noise_score,
                "estimated_noise_level": noise_level
            }
            
        except Exception as e:
            print(f"Traffic and noise analysis error: {e}")
            return {"error": str(e)}

    def predict_property_value(self, location_data, current_value=None, property_size=None, bedrooms=None):
        """
        Predict future property value based on location data, current market trends,
        and historical development patterns.
        
        Note: This is a simplified prediction model for demonstration purposes.
        """
        try:
            # Gather all our analyzed data
            flood_risk = self.analyze_flood_risk(location_data)
            sun_exposure = self.analyze_sun_exposure(location_data)
            development = self.analyze_neighborhood_development(location_data)
            traffic = self.analyze_traffic_noise(location_data)
            
            # Create features for our model
            features = {
                'lat': location_data['lat'],
                'lon': location_data['lon'],
                'flood_risk_score': flood_risk.get('flood_risk_score', 50),
                'avg_sun_score': sun_exposure.get('average_sun_score', 50),
                'development_rate': development.get('annual_growth_rate_pct', 1),
                'noise_score': traffic.get('estimated_noise_score', 50)
            }
            
            if property_size is not None:
                features['property_size'] = property_size
            
            if bedrooms is not None:
                features['bedrooms'] = bedrooms
            
            # For demonstration, we'll use a simplified formula instead of an actual trained model
            
            if current_value is None:
                # If no current value provided, we can't make a percentage prediction
                return {
                    "message": "Current property value required for prediction",
                    "recommendation": "For value prediction, please provide the current property value"
                }
            
            # Factors that positively affect value:
            # - Higher development rate
            # - Higher sun exposure
            # - Lower flood risk
            # - Lower noise
            
            # Calculate a simple growth factor
            development_factor = min(5, max(0, features['development_rate']))
            
            quality_factor = (
                (100 - features['flood_risk_score']) / 100 * 0.3 +  # Lower flood risk is better
                features['avg_sun_score'] / 100 * 0.3 +             # Higher sun exposure is better
                (100 - features['noise_score']) / 100 * 0.4         # Lower noise is better
            )
            
            # Base annual appreciation
            base_appreciation = 2.0  # 2% base annual appreciation
            
            # Adjusted appreciation rate
            adjusted_rate = base_appreciation + (development_factor * 0.5) + (quality_factor * 2 - 1)
            adjusted_rate = max(0, min(10, adjusted_rate))  # Cap between 0-10%
            
            # 5-year prediction
            five_year_value = current_value * ((1 + (adjusted_rate / 100)) ** 5)
            
            # Value prediction outcomes
            value_prediction = {
                "current_value": current_value,
                "estimated_annual_appreciation_pct": round(adjusted_rate, 2),
                "five_year_projected_value": int(five_year_value),
                "projected_gain_pct": round(((five_year_value / current_value) - 1) * 100, 2)
            }
            
            # Add investment rating
            if adjusted_rate > 6:
                value_prediction["investment_potential"] = "Excellent"
            elif adjusted_rate > 4:
                value_prediction["investment_potential"] = "Good"
            elif adjusted_rate > 2:
                value_prediction["investment_potential"] = "Average"
            else:
                value_prediction["investment_potential"] = "Below Average"
                
            return value_prediction
            
        except Exception as e:
            print(f"Property value prediction error: {e}")
            return {"error": str(e)}

    def generate_comprehensive_report(self, address=None, lat=None, lon=None,
                                     current_value=None, property_size=None, bedrooms=None):
        """Generate a comprehensive property analysis report."""
        try:
            # Get location data
            location_data = self.get_property_coordinates(address, lat, lon)
            if not location_data:
                return {"error": "Could not geocode the provided address or coordinates"}
            
            print(f"Analyzing property at: {location_data['address']}")
            
            # Run all analyses
            flood_risk = self.analyze_flood_risk(location_data)
            sun_exposure = self.analyze_sun_exposure(location_data)
            development = self.analyze_neighborhood_development(location_data)
            traffic = self.analyze_traffic_noise(location_data)
            
            value_prediction = None
            if current_value is not None:
                value_prediction = self.predict_property_value(
                    location_data, current_value, property_size, bedrooms
                )
            
            # Generate overall property score (0-100)
            scores = []
            
            # Flood risk (lower is better)
            if 'flood_risk_score' in flood_risk and flood_risk['flood_risk_score'] is not None:
                flood_score = 100 - flood_risk['flood_risk_score']
                scores.append(('Flood Safety', flood_score))
            
            # Sun exposure (higher is better)
            if 'average_sun_score' in sun_exposure and sun_exposure['average_sun_score'] is not None:
                scores.append(('Sun Exposure', sun_exposure['average_sun_score']))
            
            # Development (growth rate converted to 0-100 scale)
            if 'annual_growth_rate_pct' in development and development['annual_growth_rate_pct'] is not None:
                # Convert growth rate (-5% to +5%) to 0-100 scale
                growth_score = min(100, max(0, (development['annual_growth_rate_pct'] + 5) * 10))
                scores.append(('Development Potential', growth_score))
            
            # Noise (lower is better)
            if 'estimated_noise_score' in traffic and traffic['estimated_noise_score'] is not None:
                quiet_score = 100 - traffic['estimated_noise_score']
                scores.append(('Quietness', quiet_score))
            
            # Calculate overall score
            if scores:
                overall_score = int(sum(score for _, score in scores) / len(scores))
            else:
                overall_score = None
            
            # Determine overall rating
            if overall_score is not None:
                if overall_score >= 80:
                    overall_rating = "Excellent"
                elif overall_score >= 70:
                    overall_rating = "Very Good"
                elif overall_score >= 60:
                    overall_rating = "Good"
                elif overall_score >= 50:
                    overall_rating = "Average"
                elif overall_score >= 40:
                    overall_rating = "Below Average"
                else:
                    overall_rating = "Poor"
            else:
                overall_rating = "Unknown"
            
            # Compile report
            report = {
                "property_location": location_data,
                "overall_score": overall_score,
                "overall_rating": overall_rating,
                "category_scores": {name: score for name, score in scores},
                "flood_risk_analysis": flood_risk,
                "sun_exposure_analysis": sun_exposure,
                "neighborhood_development": development,
                "traffic_noise_analysis": traffic
            }
            
            if value_prediction:
                report["property_value_prediction"] = value_prediction
            
            return report
            
        except Exception as e:
            print(f"Report generation error: {e}")
            return {"error": str(e)}
        
        
#########################################################################
#########################################################################


    def enhanced_geocode_address(self, address):
        """Enhanced geocoding with multiple providers and retry logic."""
        geocoders = [
            (Nominatim(user_agent="property_analyzer"), "Nominatim"),
            (ArcGIS(), "ArcGIS"),
        ]

        # Try each geocoder in order
        for geocoder, name in geocoders:
            try:
                print(f"Trying geocoder: {name}")
                location = geocoder.geocode(address, timeout=10)
                if location:
                    print(f"Successfully geocoded with {name}")
                    return {
                        'lat': location.latitude,
                        'lon': location.longitude,
                        'address': location.address,
                        'provider': name
                    }
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                print(f"Error with {name} geocoder: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error with {name} geocoder: {e}")
                continue

        # Fall back to the original simulation approach if all geocoders fail
        print(f"All geocoders failed for address: {address}, using simulated location data")
        return {
            'lat': random.uniform(30, 45),
            'lon': random.uniform(-120, -70),
            'address': address,
            'provider': 'simulation'
        }

    def get_property_details(self, address, api_key=None):
        """
        Retrieve property details from a real estate API.
        """
        if api_key is None:
            api_key = os.getenv("REAL_ESTATE_API_KEY")

        if not api_key:
            print("No API key available for real estate data")
            return None

        try:
            # Sample API call to a real estate data provider
            base_url = "https://api.realestatedata.example/v1/property"
            params = {
                "address": address,
                "api_key": api_key
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                return {
                    "property_size": data.get("size_sqft"),
                    "bedrooms": data.get("bedrooms"),
                    "bathrooms": data.get("bathrooms"),
                    "year_built": data.get("year_built"),
                    "lot_size": data.get("lot_size_sqft"),
                    "last_sale_price": data.get("last_sale_price"),
                    "last_sale_date": data.get("last_sale_date"),
                    "zoning": data.get("zoning_code"),
                    "property_type": data.get("property_type")
                }
            else:
                print(f"API error: Status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching property details: {e}")
            return None

    def get_historical_sales(self, address=None, lat=None, lon=None):
        """
        Retrieve historical sales data for a property.
        """
        api_key = os.getenv("REAL_ESTATE_API_KEY")

        if not api_key:
            print("No API key available for sales history data")
            return self._simulate_sales_history()

        try:
            # Determine query parameter - address or coordinates
            params = {"api_key": api_key}

            if address:
                params["address"] = address
            elif lat is not None and lon is not None:
                params["lat"] = lat
                params["lon"] = lon
            else:
                return self._simulate_sales_history()

            # Make API request
            base_url = "https://api.realestatedata.example/v1/sales_history"
            response = requests.get(base_url, params=params)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"API error: Status code {response.status_code}")
                return self._simulate_sales_history()
        except Exception as e:
            print(f"Error fetching sales history: {e}")
            return self._simulate_sales_history()

    def _simulate_sales_history(self):
        """
        Generate simulated sales history when API data isn't available.
        """
        current_year = datetime.now().year
        current_price = random.uniform(200000, 800000)

        history = []

        # Generate 3-5 past sales
        num_sales = random.randint(3, 5)

        for i in range(num_sales):
            years_back = random.randint(2, 8) * (i + 1)
            sale_year = current_year - years_back

            # Each previous sale is some percentage lower
            discount_factor = random.uniform(0.70, 0.90)
            previous_price = current_price * discount_factor

            history.append({
                "sale_date": f"{sale_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "sale_price": int(previous_price),
                "source": "simulation"
            })

            current_price = previous_price

        # Sort by date, most recent first
        history.sort(key=lambda x: x["sale_date"], reverse=True)

        return {
            "sales_history": history,
            "average_annual_appreciation": round(random.uniform(2.0, 8.0), 2),
            "data_source": "simulation"
        }

    def analyze_school_district(self, location_data):
        """
        Analyze school district quality and nearby schools.
        """
        api_key = os.getenv("EDUCATION_API_KEY")

        if not api_key:
            return self._simulate_school_data(location_data)

        try:
            lat = location_data['lat']
            lon = location_data['lon']

            base_url = "https://api.education.example/v1/schools/nearby"
            params = {
                "lat": lat,
                "lon": lon,
                "radius": 5,  # 5 miles
                "api_key": api_key
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return self._simulate_school_data(location_data)
        except Exception as e:
            print(f"Error fetching school data: {e}")
            return self._simulate_school_data(location_data)

    def _simulate_school_data(self, location_data):
        """
        Generate simulated school district data.
        """
        # Use location to influence school quality
        lat = location_data['lat']
        lon = location_data['lon']

        # Use latitude as a factor in determining school quality
        lat_factor = (lat - 30) / 15  # Normalize between 0-1 for latitudes 30-45
        lat_factor = max(0, min(1, lat_factor))

        # Use longitude as a random seed
        random.seed(int((lon + 180) * 100) + 5)

        # Create simulated elementary schools
        elementary_schools = []
        for i in range(random.randint(2, 4)):
            quality_base = random.uniform(0.3, 0.9) + lat_factor * 0.3
            quality_score = min(10, max(1, quality_score_to_rating(quality_base * 10)))

            elementary_schools.append({
                "name": f"Elementary School #{i+1}",
                "distance_miles": round(random.uniform(0.2, 3.0), 1),
                "rating": quality_score,
                "grades": "K-5"
            })

        # Create simulated middle schools
        middle_schools = []
        for i in range(random.randint(1, 3)):
            quality_base = random.uniform(0.3, 0.9) + lat_factor * 0.3
            quality_score = min(10, max(1, quality_score_to_rating(quality_base * 10)))

            middle_schools.append({
                "name": f"Middle School #{i+1}",
                "distance_miles": round(random.uniform(0.5, 4.0), 1),
                "rating": quality_score,
                "grades": "6-8"
            })

        # Create simulated high schools
        high_schools = []
        for i in range(random.randint(1, 2)):
            quality_base = random.uniform(0.3, 0.9) + lat_factor * 0.3
            quality_score = min(10, max(1, quality_score_to_rating(quality_base * 10)))

            high_schools.append({
                "name": f"High School #{i+1}",
                "distance_miles": round(random.uniform(0.8, 5.0), 1),
                "rating": quality_score,
                "grades": "9-12"
            })

        # Calculate district average rating
        all_schools = elementary_schools + middle_schools + high_schools
        avg_rating = sum(school["rating"] for school in all_schools) / len(all_schools)

        # Determine overall quality category
        if avg_rating >= 8.5:
            quality_category = "Excellent"
        elif avg_rating >= 7.0:
            quality_category = "Very Good"
        elif avg_rating >= 5.5:
            quality_category = "Good"
        elif avg_rating >= 4.0:
            quality_category = "Average"
        else:
            quality_category = "Below Average"

        return {
            "district_name": "Simulated School District",
            "district_rating": round(avg_rating, 1),
            "district_quality": quality_category,
            "elementary_schools": elementary_schools,
            "middle_schools": middle_schools,
            "high_schools": high_schools,
            "data_source": "simulation"
        }

    def quality_score_to_rating(score):
        """Convert a 0-10 quality score to a 1-10 rating."""
        return round(score)

    def analyze_crime_rates(self, location_data):
        """
        Analyze crime rates in the area.
        """
        api_key = os.getenv("CRIME_API_KEY")

        if not api_key:
            return self._simulate_crime_data(location_data)

        try:
            lat = location_data['lat']
            lon = location_data['lon']

            base_url = "https://api.crimedata.example/v1/stats"
            params = {
                "lat": lat,
                "lon": lon,
                "radius": 2,  # 2 miles
                "api_key": api_key
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return self._simulate_crime_data(location_data)
        except Exception as e:
            print(f"Error fetching crime data: {e}")
            return self._simulate_crime_data(location_data)

    def _simulate_crime_data(self, location_data):
        """
        Generate simulated crime data.
        """
        # Use location to influence crime rates
        lat = location_data['lat']
        lon = location_data['lon']

        # Use latitude as a factor (lower latitude = higher crime in this simulation)
        lat_factor = 1 - (lat - 30) / 15  # Normalize between 0-1 for latitudes 30-45
        lat_factor = max(0, min(1, lat_factor))

        # Use longitude as a random seed
        random.seed(int((lon + 180) * 100) + 6)

        # Base crime rate influenced by latitude
        base_crime_rate = lat_factor * 60 + random.uniform(-20, 20)
        base_crime_rate = max(0, min(100, base_crime_rate))

        # Crime categories and their rates
        crime_categories = {
            "violent": base_crime_rate * random.uniform(0.1, 0.3),
            "property": base_crime_rate * random.uniform(0.4, 0.7),
            "other": base_crime_rate * random.uniform(0.2, 0.4)
        }

        # National averages (simulated)
        national_avg = {
            "violent": 30.0,
            "property": 45.0,
            "other": 25.0
        }

        # Calculate percentages compared to national average
        comparison = {}
        for category, rate in crime_categories.items():
            if national_avg[category] > 0:
                comparison[category] = (rate / national_avg[category]) * 100
            else:
                comparison[category] = 100

        # Overall safety score (100 = safest)
        safety_score = 100 - base_crime_rate

        # Safety level
        if safety_score >= 80:
            safety_level = "Very Safe"
        elif safety_score >= 60:
            safety_level = "Safe"
        elif safety_score >= 40:
            safety_level = "Moderate"
        elif safety_score >= 20:
            safety_level = "Concerning"
        else:
            safety_level = "High Crime Area"

        return {
            "safety_score": int(safety_score),
            "safety_level": safety_level,
            "crime_rates": {
                "violent": round(crime_categories["violent"], 1),
                "property": round(crime_categories["property"], 1),
                "other": round(crime_categories["other"], 1)
            },
            "comparison_to_national_avg": {
                "violent": round(comparison["violent"], 1),
                "property": round(comparison["property"], 1),
                "other": round(comparison["other"], 1)
            },
            "data_source": "simulation"
        }

    def analyze_amenities_proximity(self, location_data, radius_km=2):
        """
        Analyze proximity to amenities like shopping, parks, etc.
        """
        api_key = os.getenv("AMENITIES_API_KEY")

        if not api_key:
            return self._simulate_amenities_data(location_data, radius_km)

        try:
            lat = location_data['lat']
            lon = location_data['lon']

            base_url = "https://api.amenities.example/v1/nearby"
            params = {
                "lat": lat,
                "lon": lon,
                "radius": radius_km,
                "api_key": api_key
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return self._simulate_amenities_data(location_data, radius_km)
        except Exception as e:
            print(f"Error fetching amenities data: {e}")
            return self._simulate_amenities_data(location_data, radius_km)

    def _simulate_amenities_data(self, location_data, radius_km=2):
        """
        Generate simulated amenities data.
        """
        # Use location to influence amenity density
        lat = location_data['lat']
        lon = location_data['lon']

        # Use latitude as a factor (lower latitude = more urban/more amenities)
        lat_factor = 1 - (lat - 30) / 15  # Normalize between 0-1 for latitudes 30-45
        lat_factor = max(0, min(1, lat_factor))

        # Use longitude as a random seed
        random.seed(int((lon + 180) * 100) + 7)

        # Define amenity categories and counts
        amenity_categories = {
            "restaurants": int(lat_factor * 20 + random.uniform(1, 10)),
            "grocery_stores": int(lat_factor * 5 + random.uniform(0, 3)),
            "shopping": int(lat_factor * 8 + random.uniform(0, 5)),
            "parks": int((1 - lat_factor) * 6 + lat_factor * 3 + random.uniform(0, 3)),
            "gyms": int(lat_factor * 4 + random.uniform(0, 2)),
            "schools": int(3 + random.uniform(0, 3)),
            "medical": int(lat_factor * 6 + random.uniform(0, 3))
        }

        # Calculate closest amenities
        closest_amenities = {}
        for category in amenity_categories:
            if amenity_categories[category] > 0:
                closest_amenities[category] = round(random.uniform(0.1, radius_km), 1)
            else:
                closest_amenities[category] = None

        # Calculate convenience score (0-100)
        if sum(amenity_categories.values()) > 0:
            convenience_base = min(100, lat_factor * 70 + sum(amenity_categories.values()) * 2)
            convenience_score = int(convenience_base + random.uniform(-10, 10))
            convenience_score = max(0, min(100, convenience_score))
        else:
            convenience_score = int(lat_factor * 30 + random.uniform(0, 20))

        # Determine walkability rating
        if convenience_score >= 80:
            walkability = "Excellent"
        elif convenience_score >= 60:
            walkability = "Very Good"
        elif convenience_score >= 40:
            walkability = "Good"
        elif convenience_score >= 20:
            walkability = "Limited"
        else:
            walkability = "Poor"

        return {
            "total_amenities": sum(amenity_categories.values()),
            "amenity_counts": amenity_categories,
            "closest_amenities_km": closest_amenities,
            "convenience_score": convenience_score,
            "walkability_rating": walkability,
            "data_source": "simulation"
        }

    def generate_enhanced_comprehensive_report(self, address=None, lat=None, lon=None,
                                           current_value=None, property_size=None, bedrooms=None):
        """Generate an enhanced comprehensive property analysis report with additional data sources."""
        try:
            # Get location data using the enhanced geocoder
            location_data = self.get_property_coordinates(address, lat, lon)
            if not location_data:
                return {"error": "Could not geocode the provided address or coordinates"}

            print(f"Analyzing property at: {location_data['address']}")

            # Try to get real property details if available
            property_details = None
            if address:
                property_details = self.get_property_details(address)

            # If we got property details, use those values instead of passed parameters
            if property_details:
                if not property_size and 'property_size' in property_details:
                    property_size = property_details['property_size']
                if not bedrooms and 'bedrooms' in property_details:
                    bedrooms = property_details['bedrooms']
                if not current_value and 'last_sale_price' in property_details:
                    current_value = property_details['last_sale_price']

            # Run all original analyses
            flood_risk = self.analyze_flood_risk(location_data)
            sun_exposure = self.analyze_sun_exposure(location_data)
            development = self.analyze_neighborhood_development(location_data)
            traffic = self.analyze_traffic_noise(location_data)

            # Run new enhanced analyses
            sales_history = self.get_historical_sales(address, location_data['lat'], location_data['lon'])
            school_data = self.analyze_school_district(location_data)
            crime_data = self.analyze_crime_rates(location_data)
            amenities_data = self.analyze_amenities_proximity(location_data)

            value_prediction = None
            if current_value is not None:
                value_prediction = self.predict_property_value(
                    location_data, current_value, property_size, bedrooms
                )

            # Generate overall property score (0-100) including new factors
            scores = []

            # Original scores
            if 'flood_risk_score' in flood_risk and flood_risk['flood_risk_score'] is not None:
                flood_score = 100 - flood_risk['flood_risk_score']
                scores.append(('Flood Safety', flood_score))

            if 'average_sun_score' in sun_exposure and sun_exposure['average_sun_score'] is not None:
                scores.append(('Sun Exposure', sun_exposure['average_sun_score']))

            if 'annual_growth_rate_pct' in development and development['annual_growth_rate_pct'] is not None:
                growth_score = min(100, max(0, (development['annual_growth_rate_pct'] + 5) * 10))
                scores.append(('Development Potential', growth_score))

            if 'estimated_noise_score' in traffic and traffic['estimated_noise_score'] is not None:
                quiet_score = 100 - traffic['estimated_noise_score']
                scores.append(('Quietness', quiet_score))

            # Add new factors to scores
            if school_data and 'district_rating' in school_data:
                school_score = school_data['district_rating'] * 10  # Convert 0-10 to 0-100
                scores.append(('School Quality', school_score))

            if crime_data and 'safety_score' in crime_data:
                scores.append(('Safety', crime_data['safety_score']))

            if amenities_data and 'convenience_score' in amenities_data:
                scores.append(('Convenience', amenities_data['convenience_score']))

            # Calculate overall score
            if scores:
                overall_score = int(sum(score for _, score in scores) / len(scores))
            else:
                overall_score = None

            # Determine overall rating
            if overall_score is not None:
                if overall_score >= 80:
                    overall_rating = "Excellent"
                elif overall_score >= 70:
                    overall_rating = "Very Good"
                elif overall_score >= 60:
                    overall_rating = "Good"
                elif overall_score >= 50:
                    overall_rating = "Average"
                elif overall_score >= 40:
                    overall_rating = "Below Average"
                else:
                    overall_rating = "Poor"
            else:
                overall_rating = "Unknown"

            # Compile enhanced report
            report = {
                "property_location": location_data,
                "property_details": property_details,
                "overall_score": overall_score,
                "overall_rating": overall_rating,
                "category_scores": {name: score for name, score in scores},
                "flood_risk_analysis": flood_risk,
                "sun_exposure_analysis": sun_exposure,
                "neighborhood_development": development,
                "traffic_noise_analysis": traffic,
                "sales_history": sales_history,
                "school_district_analysis": school_data,
                "crime_analysis": crime_data,
                "amenities_analysis": amenities_data
            }

            if value_prediction:
                report["property_value_prediction"] = value_prediction

            return report

        except Exception as e:
            print(f"Enhanced report generation error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


############################################################################
############################################################################
    
    
    def visualize_report(self, report, output_file=None):
        """Generate enhanced visualizations for the property analysis report with new data sources."""
        if "error" in report:
            print(f"Cannot visualize report with error: {report['error']}")
            return

        try:
            # Create figure with more subplots for the new data
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle(f"Enhanced Property Analysis: {report['property_location']['address']}", fontsize=16)

            # Original plots (modified to fit new layout)
            # Plot 1: Category Scores Spider Chart (Polar)
            if "category_scores" in report and report["category_scores"]:
                categories = list(report["category_scores"].keys())
                values = list(report["category_scores"].values())

                # Complete the loop for the spider chart
                categories.append(categories[0])
                values.append(values[0])

                # Convert to radians for plotting
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)

                # Create a polar subplot
                ax1 = fig.add_subplot(3, 3, 1, polar=True)
                ax1.plot(angles, values, 'o-', linewidth=2)
                ax1.fill(angles, values, alpha=0.25)
                ax1.set_xticks(angles[:-1])
                ax1.set_xticklabels(categories[:-1], fontsize=8)
                ax1.set_ylim(0, 100)
                ax1.set_title("Property Quality Factors")
                ax1.grid(True)

            # Plot 2: Neighborhood Development Trend
            ax2 = fig.add_subplot(3, 3, 2)
            if ("neighborhood_development" in report and
                "development_trend" in report["neighborhood_development"] and
                report["neighborhood_development"]["development_trend"]):

                trend = report["neighborhood_development"]["development_trend"]
                years = [item["year"] for item in trend]
                values = [item["built_up_percentage"] for item in trend]

                ax2.plot(years, values, 'o-', color='green')
                ax2.set_title("Neighborhood Development Trend")
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Built-up Area (%)")
                ax2.grid(True)

            # Plot 3: Sun Exposure by Season and Time
            ax3 = fig.add_subplot(3, 3, 3)
            if ("sun_exposure_analysis" in report and
                "detailed_sun_exposure" in report["sun_exposure_analysis"]):

                sun_data = report["sun_exposure_analysis"]["detailed_sun_exposure"]
                seasons = sorted(set(item["season"] for item in sun_data))
                times = sorted(set(item["time_of_day"] for item in sun_data))

                data = np.zeros((len(seasons), len(times)))
                season_idx = {season: i for i, season in enumerate(seasons)}
                time_idx = {time: j for j, time in enumerate(times)}

                for item in sun_data:
                    i = season_idx[item["season"]]
                    j = time_idx[item["time_of_day"]]
                    data[i, j] = item["sun_exposure_score"]

                im = ax3.imshow(data, cmap='YlOrRd')

                # Set ticks and labels
                ax3.set_xticks(np.arange(len(times)))
                ax3.set_yticks(np.arange(len(seasons)))
                ax3.set_xticklabels(times)
                ax3.set_yticklabels(seasons)

                # Rotate the tick labels
                plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax3)
                cbar.set_label('Sun Exposure Score')

                ax3.set_title("Sun Exposure by Season and Time")

                # Add text annotations in the cells
                for i in range(len(seasons)):
                    for j in range(len(times)):
                        text = ax3.text(j, i, int(data[i, j]),
                                    ha="center", va="center", color="black")
                        
            
                   # NEW PLOT 4: Historical Sales Trend
            ax4 = fig.add_subplot(3, 3, 4)
            if "sales_history" in report and "sales_history" in report["sales_history"]:
                sales = report["sales_history"]["sales_history"]
                if sales:
                    # Extract dates and prices
                    dates = [sale["sale_date"] for sale in sales]
                    prices = [sale["sale_price"] for sale in sales]

                    # Sort by date
                    date_price = sorted(zip(dates, prices), key=lambda x: x[0])
                    dates = [dp[0] for dp in date_price]
                    prices = [dp[1] for dp in date_price]

                    # Format dates for better display
                    dates = [d.split("-")[0] for d in dates]  # Just show the year

                    ax4.plot(dates, prices, 'o-', color='blue')
                    ax4.set_title("Historical Sales Prices")
                    ax4.set_xlabel("Year")
                    ax4.set_ylabel("Sale Price ($)")

                    # Format y-axis to show thousands/millions
                    ax4.get_yaxis().set_major_formatter(
                        plt.FuncFormatter(lambda x, loc: "${:,}".format(int(x)))
                    )

                    # Rotate x-axis labels for better readability
                    plt.setp(ax4.get_xticklabels(), rotation=45)
                    ax4.grid(True)

            
            # Plot 4: Overall Score Gauge
            ax4 = fig.add_subplot(2, 2, 4)
            if "overall_score" in report and report["overall_score"] is not None:
                # Create a simple gauge
                score = report["overall_score"]
                
                # Create gauge
                gauge_colors = ['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9']
                bounds = [0, 20, 40, 60, 80, 100]
                norm = plt.Normalize(0, 100)
                
                # Draw gauge background
                for i in range(len(bounds)-1):
                    ax4.barh(0, bounds[i+1]-bounds[i], left=bounds[i], height=0.5,
                        color=gauge_colors[i], alpha=0.7)
                
                # Draw score indicator
                ax4.barh(0, 2, left=score-1, height=0.7, color='black')
                
                # Add score text
                ax4.text(50, -0.5, f"Overall Score: {score}/100",
                    ha='center', va='center', fontsize=14, fontweight='bold')
                ax4.text(50, 0.7, f"Rating: {report['overall_rating']}",
                    ha='center', va='center', fontsize=12)
                
                # Clean up plot
                ax4.set_xlim(0, 100)
                ax4.set_ylim(-1, 1)
                ax4.set_title("Property Score")
                ax4.axis('off')
            
            # Add overall recommendations and key insights
            recommendations = []
            
            # Flood risk recommendation
            if "flood_risk_analysis" in report and "flood_risk_level" in report["flood_risk_analysis"]:
                risk_level = report["flood_risk_analysis"]["flood_risk_level"]
                if risk_level in ["High", "Very High"]:
                    recommendations.append("⚠️ High flood risk detected. Consider flood insurance and mitigation measures.")
                elif risk_level == "Moderate":
                    recommendations.append("⚠️ Moderate flood risk. Verify if flood insurance is recommended.")
            
            # Sun exposure recommendation
            if "sun_exposure_analysis" in report and "sun_exposure_rating" in report["sun_exposure_analysis"]:
                sun_rating = report["sun_exposure_analysis"]["sun_exposure_rating"]
                if sun_rating in ["Poor", "Moderate"]:
                    recommendations.append("☀️ Limited sun exposure may increase heating/lighting costs and affect garden potential.")
                elif sun_rating == "Excellent":
                    recommendations.append("☀️ Excellent sun exposure - good potential for solar panels and gardening.")
            
            # Development recommendation
            if "neighborhood_development" in report and "development_category" in report["neighborhood_development"]:
                dev_category = report["neighborhood_development"]["development_category"]
                if dev_category in ["Rapid Growth", "Moderate Growth"]:
                    recommendations.append("📈 Area shows strong development - potential for value appreciation but watch for increasing density.")
                elif dev_category == "Declining":
                    recommendations.append("📉 Area shows declining development - may affect long-term property values.")
            
            # Noise recommendation
            if "traffic_noise_analysis" in report and "estimated_noise_level" in report["traffic_noise_analysis"]:
                noise_level = report["traffic_noise_analysis"]["estimated_noise_level"]
                if noise_level in ["High", "Very High"]:
                    recommendations.append("🔊 High noise levels detected. Consider sound insulation if purchasing.")
            
            # Value recommendation
            if "property_value_prediction" in report and "investment_potential" in report["property_value_prediction"]:
                potential = report["property_value_prediction"]["investment_potential"]
                if potential in ["Excellent", "Good"]:
                    recommendations.append("💰 Good investment potential with projected value appreciation.")
                elif potential == "Below Average":
                    recommendations.append("💰 Below average investment potential - consider negotiating price or exploring other options.")
            
            # Add recommendations to the plot
            fig.text(0.5, 0.01, "Key Insights & Recommendations:", ha='center', fontsize=14, fontweight='bold')
            for i, rec in enumerate(recommendations[:5]):  # Limit to top 5 recommendations
                fig.text(0.5, -0.02 - i*0.02, f"• {rec}", ha='center', fontsize=10)
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            if output_file:
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                print(f"Report visualization saved to {output_file}")
            
            plt.show()
            return fig
            
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main application function for Property Analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Property Analyzer: AI-powered real estate assessment tool')
    parser.add_argument('--address', type=str, help='Property address to analyze')
    parser.add_argument('--lat', type=float, help='Property latitude (alternative to address)')
    parser.add_argument('--lon', type=float, help='Property longitude (alternative to address)')
    parser.add_argument('--value', type=float, help='Current property value for investment prediction')
    parser.add_argument('--size', type=float, help='Property size in square feet/meters')
    parser.add_argument('--bedrooms', type=int, help='Number of bedrooms')
    parser.add_argument('--output', type=str, help='Output file for visualization (e.g., report.png)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PropertyAnalyzer()
    
    if args.interactive:
        print("=" * 80)
        print("Property Analyzer: AI-Powered Real Estate Assessment Tool")
        print("=" * 80)
        
        # Get property information
        address = input("Enter property address (or press Enter to use coordinates): ")
        
        lat, lon = None, None
        if not address:
            try:
                lat = float(input("Enter latitude: "))
                lon = float(input("Enter longitude: "))
            except ValueError:
                print("Error: Invalid coordinates")
                return
        
        # Get optional property details
        value = None
        try:
            value_input = input("Enter current property value (optional, press Enter to skip): ")
            if value_input:
                value = float(value_input)
        except ValueError:
            print("Warning: Invalid value, skipping investment prediction")
        
        size = None
        try:
            size_input = input("Enter property size in sq ft/m (optional, press Enter to skip): ")
            if size_input:
                size = float(size_input)
        except ValueError:
            print("Warning: Invalid size")
        
        bedrooms = None
        try:
            bedrooms_input = input("Enter number of bedrooms (optional, press Enter to skip): ")
            if bedrooms_input:
                bedrooms = int(bedrooms_input)
        except ValueError:
            print("Warning: Invalid number of bedrooms")
        
        output_file = input("Enter output file name for visualization (optional, press Enter to skip): ")
        if not output_file:
            output_file = None
            
        print("\nAnalyzing property... (this may take a minute)")
        
    else:
        # Use command line arguments
        address = args.address
        lat = args.lat
        lon = args.lon
        value = args.value
        size = args.size
        bedrooms = args.bedrooms
        output_file = args.output
        
        if not address and (lat is None or lon is None):
            print("Error: Either address or coordinates (lat/lon) must be provided")
            return
    
    # Generate report
    report = analyzer.generate_comprehensive_report(
        address=address, 
        lat=lat, 
        lon=lon,
        current_value=value,
        property_size=size,
        bedrooms=bedrooms
    )
    
    if "error" in report:
        print(f"Error generating report: {report['error']}")
        return
    
    # Display text report
    print("\n" + "=" * 80)
    print(f"PROPERTY ANALYSIS REPORT")
    print("=" * 80)
    print(f"Location: {report['property_location']['address']}")
    print(f"Coordinates: {report['property_location']['lat']}, {report['property_location']['lon']}")
    print(f"Overall Score: {report['overall_score']}/100 ({report['overall_rating']})")
    print("\nCATEGORY SCORES:")
    for category, score in report.get('category_scores', {}).items():
        print(f"- {category}: {score}/100")
    
    print("\nFLOOD RISK ASSESSMENT:")
    flood = report.get('flood_risk_analysis', {})
    print(f"- Risk Level: {flood.get('flood_risk_level', 'Unknown')}")
    print(f"- Risk Score: {flood.get('flood_risk_score', 'Unknown')}/100")
    print(f"- Elevation: {flood.get('elevation_m', 'Unknown')} meters")
    print(f"- Distance to Water: {flood.get('distance_to_water_m', 'Unknown')} meters")
    
    print("\nSUN EXPOSURE ASSESSMENT:")
    sun = report.get('sun_exposure_analysis', {})
    print(f"- Rating: {sun.get('sun_exposure_rating', 'Unknown')}")
    print(f"- Average Score: {sun.get('average_sun_score', 'Unknown')}/100")
    
    print("\nNEIGHBORHOOD DEVELOPMENT:")
    dev = report.get('neighborhood_development', {})
    print(f"- Category: {dev.get('development_category', 'Unknown')}")
    print(f"- Annual Growth Rate: {dev.get('annual_growth_rate_pct', 'Unknown')}%")
    print(f"- Analysis Period: {dev.get('analysis_period', 'Unknown')}")
    
    print("\nTRAFFIC & NOISE ASSESSMENT:")
    noise = report.get('traffic_noise_analysis', {})
    print(f"- Noise Level: {noise.get('estimated_noise_level', 'Unknown')}")
    print(f"- Noise Score: {noise.get('estimated_noise_score', 'Unknown')}/100")
    print(f"- Distance to Highway: {noise.get('distance_to_highway_m', 'Unknown')} meters")
    print(f"- Distance to Major Road: {noise.get('distance_to_major_road_m', 'Unknown')} meters")
    
    if 'property_value_prediction' in report:
        print("\nPROPERTY VALUE PREDICTION:")
        value_pred = report['property_value_prediction']
        print(f"- Current Value: ${value_pred.get('current_value', 'Unknown'):,.2f}")
        print(f"- Estimated Annual Appreciation: {value_pred.get('estimated_annual_appreciation_pct', 'Unknown')}%")
        print(f"- 5-Year Projected Value: ${value_pred.get('five_year_projected_value', 'Unknown'):,.2f}")
        print(f"- Projected Gain: {value_pred.get('projected_gain_pct', 'Unknown')}%")
        print(f"- Investment Potential: {value_pred.get('investment_potential', 'Unknown')}")
    
    # Generate visualization
    analyzer.visualize_report(report, output_file)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()