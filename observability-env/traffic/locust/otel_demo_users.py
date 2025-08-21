"""
Locust user classes specifically designed for OpenTelemetry Demo application.
Implements realistic user behavior patterns matching the OTel Demo API.
"""

import random
import json
import os
from locust import HttpUser, task, between
from locust.exception import RescheduleTask
import logging

logger = logging.getLogger(__name__)

class OTelDemoUser(HttpUser):
    """
    User class for OpenTelemetry Demo application.
    Implements realistic e-commerce behavior using OTel Demo's API endpoints.
    """
    
    wait_time = between(1, 5)
    
    def on_start(self):
        """Initialize user session for OTel Demo."""
        self.session_id = f"otel_session_{random.randint(1000, 9999)}"
        self.user_id = f"otel_user_{random.randint(100, 999)}"
        self.currency = random.choice(["USD", "EUR", "CAD", "JPY"])
        
        # OTel Demo specific product catalog
        self.products = [
            "0PUK6V6EV0",  # Vintage Typewriter
            "1YMWWN1N4O",  # Home Barista Kit  
            "2ZYFJ3GM2N",  # Road Bike
            "66VCHSJNUP",  # Stainless Steel Travel Mug
            "6E92ZMYYFZ",  # Vintage Record Player
            "9SIQT8TOJO",  # City Bike
            "L9ECAV7KIM",  # Air Plant
            "LS4PSXUNUM",  # Film Camera
            "OLJCESPC7Z"   # Vintage Camera Lens
        ]
        
        logger.info(f"OTel Demo user started: {self.user_id} with currency {self.currency}")
    
    @task(50)  # 50% - Browse products
    def browse_products(self):
        """Browse product catalog and view product details."""
        try:
            # Get product catalog
            with self.client.get("/api/products", 
                                catch_response=True, 
                                name="get_products") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Get products failed: {response.status_code}")
            
            # View random product details
            product_id = random.choice(self.products)
            with self.client.get(f"/api/products/{product_id}",
                                catch_response=True,
                                name="get_product_detail") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Get product detail failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Browse products failed: {e}")
    
    @task(20)  # 20% - Add items to cart
    def add_to_cart(self):
        """Add items to shopping cart."""
        try:
            product_id = random.choice(self.products)
            quantity = random.randint(1, 3)
            
            cart_item = {
                "product_id": product_id,
                "quantity": quantity
            }
            
            with self.client.post("/api/cart",
                                 json=cart_item,
                                 catch_response=True,
                                 name="add_to_cart") as response:
                if response.status_code in [200, 201]:
                    response.success()
                else:
                    response.failure(f"Add to cart failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Add to cart failed: {e}")
    
    @task(15)  # 15% - View cart
    def view_cart(self):
        """View current cart contents."""
        try:
            with self.client.get("/api/cart",
                                catch_response=True,
                                name="view_cart") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"View cart failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"View cart failed: {e}")
    
    @task(10)  # 10% - Get shipping quote
    def get_shipping_quote(self):
        """Get shipping cost estimate."""
        try:
            # First get cart to have items for shipping
            cart_response = self.client.get("/api/cart")
            
            if cart_response.status_code == 200:
                shipping_request = {
                    "address": {
                        "street_address": "1600 Amphitheatre Parkway",
                        "city": "Mountain View",
                        "state": "CA",
                        "country": "USA",
                        "zip_code": "94043"
                    },
                    "items": []  # Would normally include cart items
                }
                
                with self.client.post("/api/cart/checkout",
                                     json=shipping_request,
                                     catch_response=True,
                                     name="get_shipping_quote") as response:
                    if response.status_code in [200, 400]:  # 400 might be expected for empty cart
                        response.success()
                    else:
                        response.failure(f"Get shipping quote failed: {response.status_code}")
                        
        except Exception as e:
            logger.error(f"Get shipping quote failed: {e}")
    
    @task(8)  # 8% - Complete checkout
    def complete_checkout(self):
        """Complete the checkout process."""
        try:
            # Add item to cart first
            product_id = random.choice(self.products)
            cart_item = {"product_id": product_id, "quantity": 1}
            self.client.post("/api/cart", json=cart_item)
            
            # Complete checkout
            checkout_request = {
                "user_id": self.user_id,
                "user_currency": self.currency,
                "address": {
                    "street_address": "1600 Amphitheatre Parkway",
                    "city": "Mountain View", 
                    "state": "CA",
                    "country": "USA",
                    "zip_code": "94043"
                },
                "email": f"{self.user_id}@example.com",
                "credit_card": {
                    "credit_card_number": "4432-8015-6152-0454",
                    "credit_card_cvv": 672,
                    "credit_card_expiration_year": 2025,
                    "credit_card_expiration_month": 1
                }
            }
            
            with self.client.post("/api/checkout",
                                 json=checkout_request,
                                 catch_response=True,
                                 name="complete_checkout") as response:
                if response.status_code in [200, 201]:
                    response.success()
                    # Clear cart after successful checkout
                    self.client.post("/api/cart/empty")
                else:
                    response.failure(f"Complete checkout failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Complete checkout failed: {e}")
    
    @task(5)  # 5% - Get recommendations
    def get_recommendations(self):
        """Get product recommendations."""
        try:
            product_ids = random.sample(self.products, min(3, len(self.products)))
            
            with self.client.get(f"/api/recommendations?productIds={','.join(product_ids)}",
                                catch_response=True,
                                name="get_recommendations") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Get recommendations failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Get recommendations failed: {e}")
    
    @task(3)  # 3% - Currency conversion
    def convert_currency(self):
        """Test currency conversion service."""
        try:
            from_currency = "USD"
            to_currency = self.currency
            
            with self.client.get(f"/api/currency/convert?from={from_currency}&to={to_currency}&amount=100",
                                catch_response=True,
                                name="convert_currency") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Convert currency failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Convert currency failed: {e}")
    
    @task(2)  # 2% - Error scenarios
    def error_scenarios(self):
        """Generate error scenarios for testing."""
        try:
            error_actions = [
                self.invalid_product_request,
                self.malformed_cart_request,
                self.invalid_checkout_request
            ]
            
            action = random.choice(error_actions)
            action()
            
        except Exception as e:
            logger.error(f"Error scenario failed: {e}")
    
    def invalid_product_request(self):
        """Request invalid product to trigger 404."""
        invalid_id = "INVALID_PRODUCT_ID"
        with self.client.get(f"/api/products/{invalid_id}",
                            catch_response=True,
                            name="invalid_product") as response:
            # Any response is acceptable for error testing
            response.success()
    
    def malformed_cart_request(self):
        """Send malformed cart request."""
        malformed_data = {"invalid": "data", "missing_required": "fields"}
        with self.client.post("/api/cart",
                             json=malformed_data,
                             catch_response=True,
                             name="malformed_cart") as response:
            response.success()
    
    def invalid_checkout_request(self):
        """Send invalid checkout request."""
        invalid_checkout = {"incomplete": "data"}
        with self.client.post("/api/checkout",
                             json=invalid_checkout,
                             catch_response=True,
                             name="invalid_checkout") as response:
            response.success()


class OTelDemoHeavyUser(OTelDemoUser):
    """Heavy load user for stress testing OTel Demo."""
    wait_time = between(0.1, 1)
    
    @task(80)  # Increased browsing for heavy load
    def heavy_browse(self):
        """Intensive browsing behavior."""
        super().browse_products()
        # Additional product views
        for _ in range(random.randint(1, 3)):
            product_id = random.choice(self.products)
            self.client.get(f"/api/products/{product_id}", name="heavy_browse_product")


class OTelDemoErrorUser(OTelDemoUser):
    """User focused on generating errors for testing."""
    wait_time = between(0.5, 2)
    
    @task(60)  # High error rate
    def generate_errors(self):
        """Focus on error generation."""
        super().error_scenarios()
        
        # Additional error scenarios
        self.client.get("/api/nonexistent", name="404_error")
        self.client.post("/api/products", json={"invalid": "post"}, name="method_error")
        self.client.get("/api/products/", name="empty_path_error")