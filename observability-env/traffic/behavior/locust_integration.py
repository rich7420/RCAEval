"""
Locust integration for User Behavior Simulator.

This module provides Locust user classes that use the behavior simulator
to generate realistic e-commerce traffic patterns with configurable behavior.
"""

import random
import time
import json
import os
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask
from user_behavior_simulator import (
    UserBehaviorSimulator, BehaviorType, ApplicationType, 
    BehaviorConfig, UserJourney
)
from config_loader import BehaviorConfigLoader
import logging

logger = logging.getLogger(__name__)

class BehaviorDrivenUser(HttpUser):
    """
    Locust user class that uses the behavior simulator for realistic traffic patterns.
    
    This class integrates the behavior simulator with Locust to generate
    traffic that follows the configured behavior patterns (70% browse, 20% cart, 8% checkout, 2% error).
    """
    
    # Dynamic wait time based on behavior simulator
    wait_time = between(1, 10)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior_simulator = None
        self.current_journey = None
        self.behavior_config_profile = os.environ.get('BEHAVIOR_PROFILE', 'default')
        self.setup_behavior_simulator()
    
    def setup_behavior_simulator(self):
        """Initialize the behavior simulator with configuration."""
        try:
            config_loader = BehaviorConfigLoader()
            self.behavior_simulator = config_loader.create_simulator(self.behavior_config_profile)
            logger.info(f"Initialized behavior simulator with profile: {self.behavior_config_profile}")
        except Exception as e:
            logger.error(f"Failed to initialize behavior simulator: {e}")
            # Fallback to default configuration
            config = BehaviorConfig()
            self.behavior_simulator = UserBehaviorSimulator(config)
    
    def on_start(self):
        """Initialize user session when Locust user starts."""
        self.current_journey = self.behavior_simulator.create_user_session()
        logger.info(f"Started Locust user session: {self.current_journey.session_id}")
    
    def on_stop(self):
        """Finalize user session when Locust user stops."""
        if self.current_journey:
            # Mark journey as completed
            self.current_journey.total_duration = time.time() - self.current_journey.start_time.timestamp()
            
            with self.behavior_simulator.session_lock:
                if self.current_journey.session_id in self.behavior_simulator.active_sessions:
                    del self.behavior_simulator.active_sessions[self.current_journey.session_id]
                self.behavior_simulator.completed_journeys.append(self.current_journey)
            
            logger.info(f"Completed Locust user session: {self.current_journey.session_id}")
    
    @task
    def execute_behavior_pattern(self):
        """
        Main task that executes behavior patterns based on the simulator.
        
        This method uses the behavior simulator to determine the next action
        and executes it through Locust HTTP requests.
        """
        if not self.current_journey:
            return
        
        # Select next behavior using the simulator's logic
        behavior_type = self.behavior_simulator.select_next_behavior(self.current_journey)
        
        # Calculate realistic think time
        think_time = self.behavior_simulator.calculate_think_time(behavior_type, self.current_journey)
        
        # Execute the behavior through HTTP requests
        try:
            if behavior_type == BehaviorType.BROWSE:
                self.execute_browse_requests()
            elif behavior_type == BehaviorType.CART:
                self.execute_cart_requests()
            elif behavior_type == BehaviorType.CHECKOUT:
                self.execute_checkout_requests()
            elif behavior_type == BehaviorType.ERROR:
                self.execute_error_requests()
            
            # Record behavior in journey
            self.current_journey.behaviors.append(behavior_type)
            
            # Apply think time for next action
            time.sleep(think_time)
            
        except Exception as e:
            logger.error(f"Error executing {behavior_type.value} behavior: {e}")
            self.current_journey.error_count += 1
    
    def execute_browse_requests(self):
        """Execute HTTP requests for browsing behavior."""
        app_type = self.behavior_simulator.config.application_type
        
        if app_type == ApplicationType.OTEL_DEMO:
            self.execute_otel_demo_browse()
        elif app_type == ApplicationType.ONLINE_BOUTIQUE:
            self.execute_online_boutique_browse()
        else:
            self.execute_generic_browse()
    
    def execute_otel_demo_browse(self):
        """Execute OTel Demo specific browsing requests."""
        browse_actions = [
            self.otel_get_products,
            self.otel_get_product_detail,
            self.otel_get_recommendations
        ]
        
        action = random.choice(browse_actions)
        action()
    
    def execute_online_boutique_browse(self):
        """Execute Online Boutique specific browsing requests."""
        browse_actions = [
            self.boutique_browse_homepage,
            self.boutique_view_product,
            self.boutique_search_products
        ]
        
        action = random.choice(browse_actions)
        action()
    
    def execute_generic_browse(self):
        """Execute generic browsing requests."""
        with self.client.get("/", catch_response=True, name="homepage") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Homepage failed: {response.status_code}")
    
    def execute_cart_requests(self):
        """Execute HTTP requests for cart behavior."""
        app_type = self.behavior_simulator.config.application_type
        
        if app_type == ApplicationType.OTEL_DEMO:
            self.execute_otel_demo_cart()
        elif app_type == ApplicationType.ONLINE_BOUTIQUE:
            self.execute_online_boutique_cart()
        else:
            self.execute_generic_cart()
    
    def execute_otel_demo_cart(self):
        """Execute OTel Demo specific cart requests."""
        cart_actions = [
            self.otel_add_to_cart,
            self.otel_view_cart
        ]
        
        action = random.choice(cart_actions)
        action()
    
    def execute_online_boutique_cart(self):
        """Execute Online Boutique specific cart requests."""
        cart_actions = [
            self.boutique_add_to_cart,
            self.boutique_view_cart
        ]
        
        action = random.choice(cart_actions)
        action()
    
    def execute_generic_cart(self):
        """Execute generic cart requests."""
        product_id = random.choice(self.behavior_simulator.product_catalog)
        cart_data = {"product_id": product_id, "quantity": 1}
        
        with self.client.post("/cart", json=cart_data, catch_response=True, name="add_to_cart") as response:
            if response.status_code in [200, 201]:
                response.success()
                if product_id not in [item['product_id'] for item in self.current_journey.cart_items]:
                    self.current_journey.cart_items.append(cart_data)
            else:
                response.failure(f"Add to cart failed: {response.status_code}")
    
    def execute_checkout_requests(self):
        """Execute HTTP requests for checkout behavior."""
        app_type = self.behavior_simulator.config.application_type
        
        if app_type == ApplicationType.OTEL_DEMO:
            self.execute_otel_demo_checkout()
        elif app_type == ApplicationType.ONLINE_BOUTIQUE:
            self.execute_online_boutique_checkout()
        else:
            self.execute_generic_checkout()
    
    def execute_otel_demo_checkout(self):
        """Execute OTel Demo specific checkout requests."""
        # Ensure cart has items
        if not self.current_journey.cart_items:
            self.otel_add_to_cart()
        
        checkout_data = {
            "user_id": self.current_journey.user_id,
            "user_currency": "USD",
            "address": {
                "street_address": "1600 Amphitheatre Parkway",
                "city": "Mountain View",
                "state": "CA",
                "country": "USA",
                "zip_code": "94043"
            },
            "email": f"{self.current_journey.user_id}@example.com",
            "credit_card": {
                "credit_card_number": "4432-8015-6152-0454",
                "credit_card_cvv": 672,
                "credit_card_expiration_year": 2025,
                "credit_card_expiration_month": 1
            }
        }
        
        with self.client.post("/api/checkout", json=checkout_data, catch_response=True, name="otel_checkout") as response:
            if response.status_code in [200, 201]:
                response.success()
                self.current_journey.completed_checkout = True
                self.current_journey.cart_items.clear()
            else:
                response.failure(f"OTel checkout failed: {response.status_code}")
    
    def execute_online_boutique_checkout(self):
        """Execute Online Boutique specific checkout requests."""
        # Ensure cart has items
        if not self.current_journey.cart_items:
            self.boutique_add_to_cart()
        
        checkout_data = {
            "email": f"{self.current_journey.user_id}@example.com",
            "street_address": "1600 Amphitheatre Parkway",
            "zip_code": "94043",
            "city": "Mountain View",
            "state": "CA",
            "country": "United States",
            "credit_card_number": "4432-8015-6152-0454",
            "credit_card_expiration_month": "1",
            "credit_card_expiration_year": "2025",
            "credit_card_cvv": "672"
        }
        
        with self.client.post("/cart/checkout", data=checkout_data, catch_response=True, name="boutique_checkout") as response:
            if response.status_code in [200, 302]:
                response.success()
                self.current_journey.completed_checkout = True
                self.current_journey.cart_items.clear()
            else:
                response.failure(f"Boutique checkout failed: {response.status_code}")
    
    def execute_generic_checkout(self):
        """Execute generic checkout requests."""
        with self.client.post("/checkout", catch_response=True, name="generic_checkout") as response:
            if response.status_code in [200, 201, 302]:
                response.success()
                self.current_journey.completed_checkout = True
                self.current_journey.cart_items.clear()
            else:
                response.failure(f"Generic checkout failed: {response.status_code}")
    
    def execute_error_requests(self):
        """Execute HTTP requests that are designed to generate errors."""
        error_scenarios = [
            self.generate_404_error,
            self.generate_400_error,
            self.generate_timeout_error
        ]
        
        scenario = random.choice(error_scenarios)
        scenario()
    
    # OTel Demo specific methods
    def otel_get_products(self):
        """Get product catalog from OTel Demo."""
        with self.client.get("/api/products", catch_response=True, name="otel_get_products") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Get products failed: {response.status_code}")
    
    def otel_get_product_detail(self):
        """Get product details from OTel Demo."""
        product_id = random.choice(self.behavior_simulator.product_catalog)
        
        with self.client.get(f"/api/products/{product_id}", catch_response=True, name="otel_product_detail") as response:
            if response.status_code == 200:
                response.success()
                if product_id not in self.current_journey.products_viewed:
                    self.current_journey.products_viewed.append(product_id)
            else:
                response.failure(f"Product detail failed: {response.status_code}")
    
    def otel_get_recommendations(self):
        """Get product recommendations from OTel Demo."""
        product_ids = random.sample(self.behavior_simulator.product_catalog, min(3, len(self.behavior_simulator.product_catalog)))
        
        with self.client.get(f"/api/recommendations?productIds={','.join(product_ids)}", 
                           catch_response=True, name="otel_recommendations") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Recommendations failed: {response.status_code}")
    
    def otel_add_to_cart(self):
        """Add item to cart in OTel Demo."""
        product_id = random.choice(self.behavior_simulator.product_catalog)
        cart_item = {"product_id": product_id, "quantity": random.randint(1, 3)}
        
        with self.client.post("/api/cart", json=cart_item, catch_response=True, name="otel_add_cart") as response:
            if response.status_code in [200, 201]:
                response.success()
                self.current_journey.cart_items.append(cart_item)
            else:
                response.failure(f"Add to cart failed: {response.status_code}")
    
    def otel_view_cart(self):
        """View cart contents in OTel Demo."""
        with self.client.get("/api/cart", catch_response=True, name="otel_view_cart") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"View cart failed: {response.status_code}")
    
    # Online Boutique specific methods
    def boutique_browse_homepage(self):
        """Browse homepage in Online Boutique."""
        with self.client.get("/", catch_response=True, name="boutique_homepage") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Homepage failed: {response.status_code}")
    
    def boutique_view_product(self):
        """View product in Online Boutique."""
        product_id = random.choice(self.behavior_simulator.product_catalog)
        
        with self.client.get(f"/product/{product_id}", catch_response=True, name="boutique_product") as response:
            if response.status_code == 200:
                response.success()
                if product_id not in self.current_journey.products_viewed:
                    self.current_journey.products_viewed.append(product_id)
            else:
                response.failure(f"Product view failed: {response.status_code}")
    
    def boutique_search_products(self):
        """Search products in Online Boutique."""
        search_terms = ["vintage", "bike", "camera", "plant", "mug"]
        search_term = random.choice(search_terms)
        
        with self.client.get(f"/?q={search_term}", catch_response=True, name="boutique_search") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Search failed: {response.status_code}")
    
    def boutique_add_to_cart(self):
        """Add item to cart in Online Boutique."""
        product_id = random.choice(self.behavior_simulator.product_catalog)
        cart_data = {"product_id": product_id, "quantity": random.randint(1, 3)}
        
        with self.client.post("/cart", data=cart_data, catch_response=True, name="boutique_add_cart") as response:
            if response.status_code in [200, 302]:
                response.success()
                self.current_journey.cart_items.append(cart_data)
            else:
                response.failure(f"Add to cart failed: {response.status_code}")
    
    def boutique_view_cart(self):
        """View cart in Online Boutique."""
        with self.client.get("/cart", catch_response=True, name="boutique_view_cart") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"View cart failed: {response.status_code}")
    
    # Error generation methods
    def generate_404_error(self):
        """Generate 404 errors by accessing non-existent resources."""
        invalid_paths = [
            "/nonexistent",
            "/product/INVALID_ID",
            "/api/invalid_endpoint",
            "/missing_page"
        ]
        
        path = random.choice(invalid_paths)
        with self.client.get(path, catch_response=True, name="error_404") as response:
            # Any response is acceptable for error testing
            response.success()
    
    def generate_400_error(self):
        """Generate 400 errors by sending malformed requests."""
        malformed_data = [
            {"invalid": "json", "structure": None},
            "not_json_at_all",
            {"missing_required_fields": True}
        ]
        
        data = random.choice(malformed_data)
        with self.client.post("/api/cart", json=data, catch_response=True, name="error_400") as response:
            response.success()
    
    def generate_timeout_error(self):
        """Generate potential timeout errors."""
        # Request with very long parameter to potentially cause processing delays
        long_param = "x" * 1000
        with self.client.get(f"/search?q={long_param}", catch_response=True, name="error_timeout") as response:
            response.success()


# Specialized user classes for different behavior profiles
class MobileBehaviorUser(BehaviorDrivenUser):
    """User class with mobile behavior patterns."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior_config_profile = "mobile"
        self.setup_behavior_simulator()


class HighValueBehaviorUser(BehaviorDrivenUser):
    """User class with high-value customer behavior patterns."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior_config_profile = "high_value"
        self.setup_behavior_simulator()


class ErrorTestingUser(BehaviorDrivenUser):
    """User class focused on generating errors for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior_config_profile = "error_testing"
        self.setup_behavior_simulator()


# Event handlers for statistics collection
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start with behavior configuration."""
    profile = os.environ.get('BEHAVIOR_PROFILE', 'default')
    logger.info(f"Starting Locust test with behavior profile: {profile}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test completion and behavior statistics."""
    logger.info("Locust test completed")
    
    # Try to get statistics from any active user
    for user in environment.runner.user_greenlets:
        if hasattr(user, 'user') and hasattr(user.user, 'behavior_simulator'):
            stats = user.user.behavior_simulator.get_session_statistics()
            logger.info(f"Behavior statistics: {json.dumps(stats, indent=2)}")
            break