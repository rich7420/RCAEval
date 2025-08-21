"""
Locust user classes specifically designed for Google's Online Boutique application.
Implements realistic user behavior patterns matching the Online Boutique frontend.
"""

import random
import json
import os
from locust import HttpUser, task, between
import logging

logger = logging.getLogger(__name__)

class OnlineBoutiqueUser(HttpUser):
    """
    User class for Google's Online Boutique application.
    Simulates realistic shopping behavior through the web frontend.
    """
    
    wait_time = between(2, 8)
    
    def on_start(self):
        """Initialize user session for Online Boutique."""
        self.session_id = f"boutique_session_{random.randint(1000, 9999)}"
        self.user_id = f"boutique_user_{random.randint(100, 999)}"
        self.currency = random.choice(["USD", "EUR", "CAD", "JPY"])
        
        # Online Boutique product catalog (based on actual demo products)
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
        
        logger.info(f"Online Boutique user started: {self.user_id} with currency {self.currency}")
    
    @task(40)  # 40% - Browse homepage and products
    def browse_homepage(self):
        """Browse the main homepage and product listings."""
        try:
            # Load homepage
            with self.client.get("/", 
                                catch_response=True,
                                name="homepage") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Homepage failed: {response.status_code}")
            
            # Browse products with currency
            with self.client.get(f"/?currency_code={self.currency}",
                                catch_response=True,
                                name="homepage_with_currency") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Homepage with currency failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Browse homepage failed: {e}")
    
    @task(30)  # 30% - View product details
    def view_product_details(self):
        """View detailed product information."""
        try:
            product_id = random.choice(self.products)
            
            with self.client.get(f"/product/{product_id}",
                                catch_response=True,
                                name="product_details") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Product details failed: {response.status_code}")
            
            # View product with specific currency
            with self.client.get(f"/product/{product_id}?currency_code={self.currency}",
                                catch_response=True,
                                name="product_details_currency") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Product details with currency failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"View product details failed: {e}")
    
    @task(15)  # 15% - Add to cart
    def add_to_cart(self):
        """Add products to shopping cart."""
        try:
            product_id = random.choice(self.products)
            quantity = random.randint(1, 5)
            
            # Add to cart via POST request
            cart_data = {
                "product_id": product_id,
                "quantity": quantity
            }
            
            with self.client.post("/cart",
                                 data=cart_data,
                                 catch_response=True,
                                 name="add_to_cart") as response:
                if response.status_code in [200, 302]:  # 302 for redirect after add
                    response.success()
                else:
                    response.failure(f"Add to cart failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Add to cart failed: {e}")
    
    @task(10)  # 10% - View cart
    def view_cart(self):
        """View shopping cart contents."""
        try:
            with self.client.get("/cart",
                                catch_response=True,
                                name="view_cart") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"View cart failed: {response.status_code}")
            
            # View cart with currency
            with self.client.get(f"/cart?currency_code={self.currency}",
                                catch_response=True,
                                name="view_cart_currency") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"View cart with currency failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"View cart failed: {e}")
    
    @task(8)  # 8% - Checkout process
    def checkout_process(self):
        """Complete checkout process."""
        try:
            # First add an item to cart
            product_id = random.choice(self.products)
            self.client.post("/cart", data={"product_id": product_id, "quantity": 1})
            
            # Start checkout
            with self.client.get("/cart/checkout",
                                catch_response=True,
                                name="start_checkout") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Start checkout failed: {response.status_code}")
            
            # Complete checkout with shipping and payment info
            checkout_data = {
                "email": f"{self.user_id}@example.com",
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
            
            with self.client.post("/cart/checkout",
                                 data=checkout_data,
                                 catch_response=True,
                                 name="complete_checkout") as response:
                if response.status_code in [200, 302]:  # 302 for redirect after checkout
                    response.success()
                else:
                    response.failure(f"Complete checkout failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Checkout process failed: {e}")
    
    @task(5)  # 5% - Search functionality
    def search_products(self):
        """Search for products using search functionality."""
        try:
            search_terms = ["vintage", "bike", "camera", "plant", "mug", "typewriter"]
            search_term = random.choice(search_terms)
            
            with self.client.get(f"/?q={search_term}",
                                catch_response=True,
                                name="search_products") as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Search products failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Search products failed: {e}")
    
    @task(3)  # 3% - Currency switching
    def switch_currency(self):
        """Test currency switching functionality."""
        try:
            currencies = ["USD", "EUR", "CAD", "JPY", "GBP"]
            new_currency = random.choice(currencies)
            
            with self.client.post("/setCurrency",
                                 data={"currency_code": new_currency},
                                 catch_response=True,
                                 name="switch_currency") as response:
                if response.status_code in [200, 302]:
                    self.currency = new_currency
                    response.success()
                else:
                    response.failure(f"Switch currency failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Switch currency failed: {e}")
    
    @task(2)  # 2% - Error scenarios
    def error_scenarios(self):
        """Generate various error scenarios."""
        try:
            error_actions = [
                self.invalid_product_page,
                self.empty_cart_checkout,
                self.invalid_search,
                self.malformed_cart_request
            ]
            
            action = random.choice(error_actions)
            action()
            
        except Exception as e:
            logger.error(f"Error scenario failed: {e}")
    
    def invalid_product_page(self):
        """Access non-existent product page."""
        invalid_id = "NONEXISTENT_PRODUCT"
        with self.client.get(f"/product/{invalid_id}",
                            catch_response=True,
                            name="invalid_product_page") as response:
            response.success()  # Any response is acceptable for error testing
    
    def empty_cart_checkout(self):
        """Try to checkout with empty cart."""
        # Clear cart first
        self.client.post("/cart/empty")
        
        with self.client.get("/cart/checkout",
                            catch_response=True,
                            name="empty_cart_checkout") as response:
            response.success()
    
    def invalid_search(self):
        """Search with invalid or problematic terms."""
        invalid_terms = ["", "   ", "!@#$%", "x" * 1000]
        term = random.choice(invalid_terms)
        
        with self.client.get(f"/?q={term}",
                            catch_response=True,
                            name="invalid_search") as response:
            response.success()
    
    def malformed_cart_request(self):
        """Send malformed add to cart request."""
        malformed_data = {
            "product_id": "",
            "quantity": "invalid"
        }
        
        with self.client.post("/cart",
                             data=malformed_data,
                             catch_response=True,
                             name="malformed_cart_request") as response:
            response.success()


class OnlineBoutiqueMobileUser(OnlineBoutiqueUser):
    """
    Mobile user simulation with different behavior patterns.
    Shorter sessions, more browsing, less checkout completion.
    """
    
    wait_time = between(1, 4)  # Faster interactions on mobile
    
    @task(60)  # Higher browsing rate on mobile
    def mobile_browse(self):
        """Mobile-optimized browsing behavior."""
        super().browse_homepage()
        super().view_product_details()
    
    @task(25)  # Mobile users add to cart but don't always checkout
    def mobile_cart_behavior(self):
        """Mobile cart behavior - add items but often abandon."""
        super().add_to_cart()
        if random.random() < 0.3:  # Only 30% proceed to view cart
            super().view_cart()
    
    @task(3)  # Much lower checkout completion on mobile
    def mobile_checkout(self):
        """Reduced checkout completion rate for mobile."""
        super().checkout_process()


class OnlineBoutiqueHighValueUser(OnlineBoutiqueUser):
    """
    High-value customer simulation with different purchasing patterns.
    More items per cart, higher checkout completion rate.
    """
    
    wait_time = between(3, 12)  # More deliberate shopping
    
    @task(20)  # Less browsing, more focused shopping
    def focused_browse(self):
        """Focused browsing for high-value customers."""
        super().browse_homepage()
    
    @task(40)  # More product detail viewing
    def detailed_product_research(self):
        """Thorough product research."""
        super().view_product_details()
        # View multiple products
        for _ in range(random.randint(1, 3)):
            product_id = random.choice(self.products)
            self.client.get(f"/product/{product_id}")
    
    @task(25)  # Higher add to cart rate
    def high_value_cart(self):
        """Add multiple items to cart."""
        for _ in range(random.randint(1, 4)):  # Add multiple items
            super().add_to_cart()
        super().view_cart()
    
    @task(15)  # Higher checkout completion rate
    def high_value_checkout(self):
        """Higher likelihood of completing purchase."""
        super().checkout_process()