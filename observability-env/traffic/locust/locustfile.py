"""
Locust traffic generator for observability data collection.
Simulates realistic e-commerce user behavior with configurable patterns.
"""

import random
import time
from locust import HttpUser, task, between
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import json
import os
import logging

# Setup logging
setup_logging("INFO", None)
logger = logging.getLogger(__name__)

class ECommerceUser(HttpUser):
    """
    Simulates realistic e-commerce user behavior patterns.
    
    Traffic distribution:
    - 70% browsing behavior
    - 20% cart operations  
    - 8% checkout process
    - 2% error scenarios
    """
    
    # Realistic think time between actions (1-10 seconds)
    wait_time = between(1, 10)
    
    def on_start(self):
        """Initialize user session and set up context."""
        self.session_id = f"session_{random.randint(1000, 9999)}"
        self.user_id = f"user_{random.randint(100, 999)}"
        self.cart_items = []
        
        # Load configuration if available
        self.config = self.load_config()
        
        logger.info(f"Starting session {self.session_id} for user {self.user_id}")
    
    def load_config(self):
        """Load traffic generation configuration."""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        default_config = {
            "browse_weight": 70,
            "cart_weight": 20, 
            "checkout_weight": 8,
            "error_weight": 2,
            "think_time_min": 1,
            "think_time_max": 10,
            "products": ["product1", "product2", "product3", "product4", "product5"]
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    default_config.update(config)
            return default_config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return default_config
    
    @task(70)  # 70% of traffic - browsing behavior
    def browse_products(self):
        """Simulate product browsing behavior."""
        try:
            # Browse homepage
            with self.client.get("/", catch_response=True, name="homepage") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Homepage failed with status {response.status_code}")
            
            # Browse product categories
            categories = ["electronics", "clothing", "books", "home"]
            category = random.choice(categories)
            
            with self.client.get(f"/category/{category}", catch_response=True, name="browse_category") as response:
                if response.status_code in [200, 404]:  # 404 might be expected for some categories
                    response.success()
                else:
                    response.failure(f"Category browse failed with status {response.status_code}")
            
            # View specific products
            product_id = random.choice(self.config["products"])
            with self.client.get(f"/product/{product_id}", catch_response=True, name="view_product") as response:
                if response.status_code in [200, 404]:
                    response.success()
                else:
                    response.failure(f"Product view failed with status {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Browse products failed: {e}")
    
    @task(20)  # 20% of traffic - cart operations
    def cart_operations(self):
        """Simulate add to cart and cart management."""
        try:
            product_id = random.choice(self.config["products"])
            quantity = random.randint(1, 3)
            
            # Add item to cart
            cart_data = {
                "product_id": product_id,
                "quantity": quantity,
                "user_id": self.user_id
            }
            
            with self.client.post("/cart/add", json=cart_data, catch_response=True, name="add_to_cart") as response:
                if response.status_code in [200, 201]:
                    self.cart_items.append(product_id)
                    response.success()
                else:
                    response.failure(f"Add to cart failed with status {response.status_code}")
            
            # View cart occasionally
            if random.random() < 0.3:  # 30% chance to view cart
                with self.client.get("/cart", catch_response=True, name="view_cart") as response:
                    if response.status_code == 200:
                        response.success()
                    else:
                        response.failure(f"View cart failed with status {response.status_code}")
                        
        except Exception as e:
            logger.error(f"Cart operations failed: {e}")
    
    @task(8)  # 8% of traffic - checkout process
    def checkout_process(self):
        """Simulate complete checkout process."""
        try:
            if not self.cart_items:
                # Add an item first if cart is empty
                self.cart_operations()
            
            # Start checkout
            with self.client.get("/checkout", catch_response=True, name="start_checkout") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Start checkout failed with status {response.status_code}")
            
            # Add shipping information
            shipping_data = {
                "address": "123 Test St",
                "city": "Test City", 
                "zip": "12345",
                "user_id": self.user_id
            }
            
            with self.client.post("/checkout/shipping", json=shipping_data, catch_response=True, name="add_shipping") as response:
                if response.status_code in [200, 201]:
                    response.success()
                else:
                    response.failure(f"Add shipping failed with status {response.status_code}")
            
            # Process payment
            payment_data = {
                "card_number": "4111111111111111",
                "expiry": "12/25",
                "cvv": "123",
                "user_id": self.user_id
            }
            
            with self.client.post("/checkout/payment", json=payment_data, catch_response=True, name="process_payment") as response:
                if response.status_code in [200, 201]:
                    self.cart_items.clear()  # Clear cart after successful payment
                    response.success()
                else:
                    response.failure(f"Process payment failed with status {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Checkout process failed: {e}")
    
    @task(2)  # 2% of traffic - error scenarios
    def error_scenarios(self):
        """Simulate error-inducing user behavior."""
        try:
            error_scenarios = [
                self.invalid_product_access,
                self.malformed_requests,
                self.timeout_scenarios
            ]
            
            scenario = random.choice(error_scenarios)
            scenario()
            
        except Exception as e:
            logger.error(f"Error scenario failed: {e}")
    
    def invalid_product_access(self):
        """Access invalid or non-existent products."""
        invalid_ids = ["invalid", "999999", "null", ""]
        product_id = random.choice(invalid_ids)
        
        with self.client.get(f"/product/{product_id}", catch_response=True, name="invalid_product") as response:
            # We expect this to fail, so any response is acceptable
            response.success()
    
    def malformed_requests(self):
        """Send malformed requests to trigger errors."""
        malformed_data = [
            {"invalid": "json", "structure": None},
            "not_json_at_all",
            {"missing_required_fields": True}
        ]
        
        data = random.choice(malformed_data)
        
        with self.client.post("/cart/add", json=data, catch_response=True, name="malformed_request") as response:
            # Any response is acceptable for error scenarios
            response.success()
    
    def timeout_scenarios(self):
        """Trigger potential timeout scenarios."""
        # Request with very long product ID to potentially cause processing delays
        long_id = "x" * 1000
        
        with self.client.get(f"/product/{long_id}", catch_response=True, name="timeout_scenario") as response:
            response.success()


class DistributedECommerceUser(ECommerceUser):
    """
    Extended user class for distributed load generation.
    Includes additional coordination and reporting capabilities.
    """
    
    def on_start(self):
        """Enhanced initialization for distributed testing."""
        super().on_start()
        
        # Add distributed testing context
        self.worker_id = os.environ.get('LOCUST_WORKER_ID', 'master')
        self.test_id = os.environ.get('LOCUST_TEST_ID', 'default')
        
        logger.info(f"Distributed user started - Worker: {self.worker_id}, Test: {self.test_id}")
    
    def on_stop(self):
        """Cleanup for distributed testing."""
        logger.info(f"Distributed user stopped - Session: {self.session_id}")


# Configuration for different load patterns
class LightLoadUser(ECommerceUser):
    """Light load pattern - fewer requests, longer think times."""
    wait_time = between(5, 15)


class HeavyLoadUser(ECommerceUser):
    """Heavy load pattern - more requests, shorter think times."""
    wait_time = between(0.5, 3)


class BurstLoadUser(ECommerceUser):
    """Burst load pattern - alternating between high and low activity."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.burst_mode = False
        self.burst_start = time.time()
    
    def wait_time_func(self):
        """Dynamic wait time based on burst pattern."""
        current_time = time.time()
        
        # Switch burst mode every 30 seconds
        if current_time - self.burst_start > 30:
            self.burst_mode = not self.burst_mode
            self.burst_start = current_time
        
        if self.burst_mode:
            return random.uniform(0.1, 1)  # High activity
        else:
            return random.uniform(3, 8)    # Low activity
    
    wait_time = wait_time_func