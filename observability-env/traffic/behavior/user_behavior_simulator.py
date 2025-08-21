"""
User Behavior Simulator for Observability Data Collection

This module implements configurable behavior patterns for e-commerce scenarios
with realistic traffic distribution and user journey flows.

Traffic Distribution:
- 70% browse behavior
- 20% cart operations  
- 8% checkout process
- 2% error scenarios

Features:
- Configurable behavior patterns
- Realistic delays and user journey flows
- Support for both OTel Demo and Online Boutique applications
- Dynamic traffic distribution
- User session management
- Comprehensive error scenario generation
"""

import random
import time
import json
import os
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ApplicationType(Enum):
    """Supported application types."""
    OTEL_DEMO = "otel_demo"
    ONLINE_BOUTIQUE = "online_boutique"

class BehaviorType(Enum):
    """User behavior types."""
    BROWSE = "browse"
    CART = "cart"
    CHECKOUT = "checkout"
    ERROR = "error"

@dataclass
class UserJourney:
    """Represents a complete user journey through the application."""
    journey_id: str
    user_id: str
    session_id: str
    start_time: datetime
    behaviors: List[BehaviorType] = field(default_factory=list)
    products_viewed: List[str] = field(default_factory=list)
    cart_items: List[Dict] = field(default_factory=list)
    completed_checkout: bool = False
    total_duration: float = 0.0
    error_count: int = 0

@dataclass
class BehaviorConfig:
    """Configuration for user behavior patterns."""
    # Traffic distribution (must sum to 100)
    browse_percentage: float = 70.0
    cart_percentage: float = 20.0
    checkout_percentage: float = 8.0
    error_percentage: float = 2.0
    
    # Timing configuration (seconds)
    min_think_time: float = 1.0
    max_think_time: float = 10.0
    session_duration_min: float = 60.0
    session_duration_max: float = 600.0
    
    # Journey configuration
    max_products_per_session: int = 10
    max_cart_items: int = 5
    checkout_completion_rate: float = 0.6  # 60% of checkout attempts complete
    
    # Error configuration
    error_retry_attempts: int = 2
    error_scenarios_enabled: bool = True
    
    # Application-specific settings
    application_type: ApplicationType = ApplicationType.OTEL_DEMO
    base_url: str = "http://localhost:8080"
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        total_percentage = (self.browse_percentage + self.cart_percentage + 
                          self.checkout_percentage + self.error_percentage)
        
        if abs(total_percentage - 100.0) > 0.1:
            logger.error(f"Traffic distribution must sum to 100%, got {total_percentage}%")
            return False
            
        if self.min_think_time >= self.max_think_time:
            logger.error("min_think_time must be less than max_think_time")
            return False
            
        if self.session_duration_min >= self.session_duration_max:
            logger.error("session_duration_min must be less than session_duration_max")
            return False
            
        return True

class UserBehaviorSimulator:
    """
    Main class for simulating realistic user behavior patterns.
    
    Implements configurable e-commerce user journeys with realistic timing,
    traffic distribution, and error scenarios.
    """
    
    def __init__(self, config: BehaviorConfig):
        """Initialize the behavior simulator."""
        self.config = config
        if not self.config.validate():
            raise ValueError("Invalid configuration provided")
            
        self.active_sessions: Dict[str, UserJourney] = {}
        self.completed_journeys: List[UserJourney] = []
        self.session_lock = threading.Lock()
        
        # Load application-specific product catalogs
        self.product_catalog = self._load_product_catalog()
        
        # Behavior weights for random selection
        self.behavior_weights = [
            (BehaviorType.BROWSE, self.config.browse_percentage),
            (BehaviorType.CART, self.config.cart_percentage),
            (BehaviorType.CHECKOUT, self.config.checkout_percentage),
            (BehaviorType.ERROR, self.config.error_percentage)
        ]
        
        logger.info(f"User Behavior Simulator initialized for {self.config.application_type.value}")
        logger.info(f"Traffic distribution: Browse {self.config.browse_percentage}%, "
                   f"Cart {self.config.cart_percentage}%, "
                   f"Checkout {self.config.checkout_percentage}%, "
                   f"Error {self.config.error_percentage}%")
    
    def _load_product_catalog(self) -> List[str]:
        """Load product catalog based on application type."""
        if self.config.application_type == ApplicationType.OTEL_DEMO:
            return [
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
        elif self.config.application_type == ApplicationType.ONLINE_BOUTIQUE:
            return [
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
        else:
            return ["product1", "product2", "product3", "product4", "product5"]
    
    def create_user_session(self, user_id: Optional[str] = None) -> UserJourney:
        """Create a new user session and journey."""
        if user_id is None:
            user_id = f"user_{random.randint(1000, 9999)}"
            
        session_id = f"session_{random.randint(10000, 99999)}_{int(time.time())}"
        journey_id = f"journey_{session_id}"
        
        journey = UserJourney(
            journey_id=journey_id,
            user_id=user_id,
            session_id=session_id,
            start_time=datetime.now()
        )
        
        with self.session_lock:
            self.active_sessions[session_id] = journey
            
        logger.info(f"Created new user session: {session_id} for user: {user_id}")
        return journey
    
    def select_next_behavior(self, journey: UserJourney) -> BehaviorType:
        """
        Select the next behavior based on traffic distribution and user journey context.
        
        Implements intelligent behavior selection that considers:
        - Overall traffic distribution requirements (70/20/8/2)
        - User's current journey state
        - Realistic user flow patterns
        """
        # Use a hybrid approach: 70% based on target distribution, 30% based on context
        use_target_distribution = random.random() < 0.7
        
        if use_target_distribution or len(journey.behaviors) == 0:
            # Use target distribution weights
            weights = [
                (BehaviorType.BROWSE, self.config.browse_percentage),
                (BehaviorType.CART, self.config.cart_percentage),
                (BehaviorType.CHECKOUT, self.config.checkout_percentage),
                (BehaviorType.ERROR, self.config.error_percentage)
            ]
        else:
            # Use contextual weights based on last behavior
            last_behavior = journey.behaviors[-1]
            
            if last_behavior == BehaviorType.BROWSE:
                # After browsing, slightly favor continuing to browse or adding to cart
                weights = [
                    (BehaviorType.BROWSE, 60.0),
                    (BehaviorType.CART, 30.0),
                    (BehaviorType.CHECKOUT, 7.0),
                    (BehaviorType.ERROR, 3.0)
                ]
            elif last_behavior == BehaviorType.CART:
                # After cart operations, favor checkout or more browsing
                weights = [
                    (BehaviorType.BROWSE, 50.0),
                    (BehaviorType.CART, 20.0),
                    (BehaviorType.CHECKOUT, 25.0),
                    (BehaviorType.ERROR, 5.0)
                ]
            elif last_behavior == BehaviorType.CHECKOUT:
                # After checkout, usually browse more or end session
                if journey.completed_checkout:
                    weights = [
                        (BehaviorType.BROWSE, 80.0),
                        (BehaviorType.CART, 10.0),
                        (BehaviorType.CHECKOUT, 5.0),
                        (BehaviorType.ERROR, 5.0)
                    ]
                else:
                    # Failed checkout, might retry or browse
                    weights = [
                        (BehaviorType.BROWSE, 60.0),
                        (BehaviorType.CART, 15.0),
                        (BehaviorType.CHECKOUT, 20.0),
                        (BehaviorType.ERROR, 5.0)
                    ]
            else:  # ERROR
                # After error, usually browse or retry
                weights = [
                    (BehaviorType.BROWSE, 70.0),
                    (BehaviorType.CART, 15.0),
                    (BehaviorType.CHECKOUT, 10.0),
                    (BehaviorType.ERROR, 5.0)
                ]
        
        # Select behavior based on weights
        behaviors, probabilities = zip(*weights)
        return random.choices(behaviors, weights=probabilities)[0]
    
    def calculate_think_time(self, behavior_type: BehaviorType, journey: UserJourney) -> float:
        """
        Calculate realistic think time based on behavior type and user context.
        
        Different behaviors have different natural timing patterns:
        - Browsing: Variable timing as users read and consider
        - Cart operations: Usually quicker decisions
        - Checkout: Longer as users enter information
        - Errors: Often followed by quick retry or longer consideration
        """
        base_min = self.config.min_think_time
        base_max = self.config.max_think_time
        
        # Adjust timing based on behavior type
        if behavior_type == BehaviorType.BROWSE:
            # Browsing can be quick (scanning) or slow (detailed reading)
            min_time = base_min
            max_time = base_max * 1.5
        elif behavior_type == BehaviorType.CART:
            # Cart operations are usually quick decisions
            min_time = base_min * 0.5
            max_time = base_max * 0.8
        elif behavior_type == BehaviorType.CHECKOUT:
            # Checkout requires form filling and consideration
            min_time = base_min * 2
            max_time = base_max * 2
        elif behavior_type == BehaviorType.ERROR:
            # Errors might cause quick retry or longer consideration
            min_time = base_min * 0.3
            max_time = base_max * 1.2
        else:
            min_time = base_min
            max_time = base_max
        
        # Add session-based variation (users get faster as session progresses)
        session_progress = len(journey.behaviors) / 20.0  # Normalize to 0-1
        speed_factor = 1.0 - (session_progress * 0.3)  # Up to 30% faster
        
        min_time *= speed_factor
        max_time *= speed_factor
        
        return random.uniform(max(0.1, min_time), max(0.5, max_time))
    
    def execute_browse_behavior(self, journey: UserJourney) -> Dict[str, Any]:
        """
        Execute browsing behavior patterns.
        
        Browsing includes:
        - Homepage visits
        - Category browsing
        - Product detail views
        - Search functionality
        """
        actions = []
        
        # Select browsing pattern
        browse_patterns = [
            self._browse_homepage,
            self._browse_category,
            self._view_product_details,
            self._search_products
        ]
        
        # Execute 1-3 browsing actions
        num_actions = random.randint(1, 3)
        for _ in range(num_actions):
            pattern = random.choice(browse_patterns)
            action_result = pattern(journey)
            actions.append(action_result)
            
            # Add realistic delay between actions
            time.sleep(random.uniform(0.5, 2.0))
        
        journey.behaviors.append(BehaviorType.BROWSE)
        
        return {
            "behavior_type": "browse",
            "actions": actions,
            "products_viewed": len(journey.products_viewed),
            "timestamp": datetime.now().isoformat()
        }
    
    def _browse_homepage(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate homepage browsing."""
        logger.info(f"User {journey.user_id} browsing homepage")
        
        return {
            "action": "browse_homepage",
            "url": "/",
            "duration": random.uniform(2, 8),
            "success": True
        }
    
    def _browse_category(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate category browsing."""
        categories = ["electronics", "clothing", "books", "home", "sports"]
        category = random.choice(categories)
        
        logger.info(f"User {journey.user_id} browsing category: {category}")
        
        return {
            "action": "browse_category",
            "category": category,
            "url": f"/category/{category}",
            "duration": random.uniform(3, 12),
            "success": True
        }
    
    def _view_product_details(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate product detail viewing."""
        product_id = random.choice(self.product_catalog)
        
        if product_id not in journey.products_viewed:
            journey.products_viewed.append(product_id)
        
        logger.info(f"User {journey.user_id} viewing product: {product_id}")
        
        return {
            "action": "view_product",
            "product_id": product_id,
            "url": f"/product/{product_id}",
            "duration": random.uniform(5, 20),
            "success": True
        }
    
    def _search_products(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate product search."""
        search_terms = ["vintage", "bike", "camera", "plant", "mug", "typewriter", "kit"]
        search_term = random.choice(search_terms)
        
        logger.info(f"User {journey.user_id} searching for: {search_term}")
        
        return {
            "action": "search_products",
            "search_term": search_term,
            "url": f"/search?q={search_term}",
            "duration": random.uniform(2, 6),
            "success": True
        }
    
    def execute_cart_behavior(self, journey: UserJourney) -> Dict[str, Any]:
        """
        Execute cart-related behavior patterns.
        
        Cart behaviors include:
        - Adding items to cart
        - Viewing cart contents
        - Modifying cart quantities
        - Removing items from cart
        """
        actions = []
        
        # Ensure user has viewed some products before adding to cart
        if not journey.products_viewed:
            self._view_product_details(journey)
        
        cart_patterns = [
            self._add_to_cart,
            self._view_cart,
            self._modify_cart_quantity,
            self._remove_from_cart
        ]
        
        # Weight patterns based on current cart state
        if not journey.cart_items:
            # Empty cart - focus on adding items
            pattern = self._add_to_cart
        elif len(journey.cart_items) >= self.config.max_cart_items:
            # Full cart - focus on viewing or modifying
            pattern = random.choice([self._view_cart, self._modify_cart_quantity, self._remove_from_cart])
        else:
            # Normal cart - any action is reasonable
            pattern = random.choice(cart_patterns)
        
        action_result = pattern(journey)
        actions.append(action_result)
        
        journey.behaviors.append(BehaviorType.CART)
        
        return {
            "behavior_type": "cart",
            "actions": actions,
            "cart_items": len(journey.cart_items),
            "timestamp": datetime.now().isoformat()
        }
    
    def _add_to_cart(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate adding item to cart."""
        if len(journey.cart_items) >= self.config.max_cart_items:
            return {"action": "add_to_cart", "success": False, "reason": "cart_full"}
        
        # Select product from viewed products or random
        if journey.products_viewed:
            product_id = random.choice(journey.products_viewed)
        else:
            product_id = random.choice(self.product_catalog)
            journey.products_viewed.append(product_id)
        
        quantity = random.randint(1, 3)
        
        cart_item = {
            "product_id": product_id,
            "quantity": quantity,
            "added_at": datetime.now().isoformat()
        }
        
        journey.cart_items.append(cart_item)
        
        logger.info(f"User {journey.user_id} added {quantity}x {product_id} to cart")
        
        return {
            "action": "add_to_cart",
            "product_id": product_id,
            "quantity": quantity,
            "success": True,
            "duration": random.uniform(1, 4)
        }
    
    def _view_cart(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate viewing cart contents."""
        logger.info(f"User {journey.user_id} viewing cart ({len(journey.cart_items)} items)")
        
        return {
            "action": "view_cart",
            "cart_size": len(journey.cart_items),
            "success": True,
            "duration": random.uniform(2, 8)
        }
    
    def _modify_cart_quantity(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate modifying cart item quantity."""
        if not journey.cart_items:
            return {"action": "modify_cart", "success": False, "reason": "empty_cart"}
        
        item = random.choice(journey.cart_items)
        old_quantity = item["quantity"]
        item["quantity"] = random.randint(1, 5)
        
        logger.info(f"User {journey.user_id} changed quantity of {item['product_id']} "
                   f"from {old_quantity} to {item['quantity']}")
        
        return {
            "action": "modify_cart_quantity",
            "product_id": item["product_id"],
            "old_quantity": old_quantity,
            "new_quantity": item["quantity"],
            "success": True,
            "duration": random.uniform(1, 3)
        }
    
    def _remove_from_cart(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate removing item from cart."""
        if not journey.cart_items:
            return {"action": "remove_from_cart", "success": False, "reason": "empty_cart"}
        
        item = journey.cart_items.pop(random.randint(0, len(journey.cart_items) - 1))
        
        logger.info(f"User {journey.user_id} removed {item['product_id']} from cart")
        
        return {
            "action": "remove_from_cart",
            "product_id": item["product_id"],
            "success": True,
            "duration": random.uniform(1, 2)
        }
    
    def execute_checkout_behavior(self, journey: UserJourney) -> Dict[str, Any]:
        """
        Execute checkout behavior patterns.
        
        Checkout process includes:
        - Starting checkout process
        - Entering shipping information
        - Selecting payment method
        - Completing or abandoning checkout
        """
        actions = []
        
        # Ensure cart has items for checkout
        if not journey.cart_items:
            # Add item to cart first
            self._add_to_cart(journey)
        
        checkout_success = random.random() < self.config.checkout_completion_rate
        
        # Start checkout process
        actions.append(self._start_checkout(journey))
        time.sleep(random.uniform(1, 3))
        
        # Enter shipping information
        actions.append(self._enter_shipping_info(journey))
        time.sleep(random.uniform(2, 5))
        
        if checkout_success:
            # Complete payment
            actions.append(self._complete_payment(journey))
            journey.completed_checkout = True
            journey.cart_items.clear()  # Clear cart after successful checkout
        else:
            # Abandon checkout
            actions.append(self._abandon_checkout(journey))
        
        journey.behaviors.append(BehaviorType.CHECKOUT)
        
        return {
            "behavior_type": "checkout",
            "actions": actions,
            "completed": checkout_success,
            "timestamp": datetime.now().isoformat()
        }
    
    def _start_checkout(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate starting checkout process."""
        logger.info(f"User {journey.user_id} starting checkout with {len(journey.cart_items)} items")
        
        return {
            "action": "start_checkout",
            "cart_items": len(journey.cart_items),
            "success": True,
            "duration": random.uniform(2, 5)
        }
    
    def _enter_shipping_info(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate entering shipping information."""
        logger.info(f"User {journey.user_id} entering shipping information")
        
        # Simulate form filling time
        form_fill_time = random.uniform(15, 45)
        
        return {
            "action": "enter_shipping_info",
            "success": True,
            "duration": form_fill_time
        }
    
    def _complete_payment(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate completing payment."""
        logger.info(f"User {journey.user_id} completing payment")
        
        # Simulate payment processing time
        payment_time = random.uniform(5, 15)
        
        return {
            "action": "complete_payment",
            "success": True,
            "duration": payment_time
        }
    
    def _abandon_checkout(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate abandoning checkout."""
        logger.info(f"User {journey.user_id} abandoning checkout")
        
        return {
            "action": "abandon_checkout",
            "success": True,
            "duration": random.uniform(1, 3)
        }
    
    def execute_error_behavior(self, journey: UserJourney) -> Dict[str, Any]:
        """
        Execute error-inducing behavior patterns.
        
        Error scenarios include:
        - Invalid product access
        - Malformed requests
        - Timeout scenarios
        - Network simulation errors
        """
        if not self.config.error_scenarios_enabled:
            return {"behavior_type": "error", "skipped": True, "reason": "disabled"}
        
        error_scenarios = [
            self._invalid_product_access,
            self._malformed_request,
            self._timeout_scenario,
            self._invalid_cart_operation,
            self._invalid_checkout_data
        ]
        
        scenario = random.choice(error_scenarios)
        action_result = scenario(journey)
        
        journey.behaviors.append(BehaviorType.ERROR)
        journey.error_count += 1
        
        return {
            "behavior_type": "error",
            "action": action_result,
            "error_count": journey.error_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def _invalid_product_access(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate accessing invalid or non-existent products."""
        invalid_ids = ["invalid", "999999", "null", "", "nonexistent_product", "test123"]
        product_id = random.choice(invalid_ids)
        
        logger.info(f"User {journey.user_id} accessing invalid product: {product_id}")
        
        return {
            "action": "invalid_product_access",
            "product_id": product_id,
            "expected_error": "404_not_found",
            "duration": random.uniform(1, 3)
        }
    
    def _malformed_request(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate sending malformed requests."""
        malformed_types = [
            "invalid_json",
            "missing_required_fields",
            "wrong_data_types",
            "oversized_payload",
            "special_characters"
        ]
        
        error_type = random.choice(malformed_types)
        
        logger.info(f"User {journey.user_id} sending malformed request: {error_type}")
        
        return {
            "action": "malformed_request",
            "error_type": error_type,
            "expected_error": "400_bad_request",
            "duration": random.uniform(0.5, 2)
        }
    
    def _timeout_scenario(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate scenarios that might cause timeouts."""
        timeout_scenarios = [
            "very_long_product_id",
            "complex_search_query",
            "large_cart_operation",
            "slow_network_simulation"
        ]
        
        scenario = random.choice(timeout_scenarios)
        
        logger.info(f"User {journey.user_id} triggering timeout scenario: {scenario}")
        
        return {
            "action": "timeout_scenario",
            "scenario_type": scenario,
            "expected_error": "timeout",
            "duration": random.uniform(10, 30)  # Longer duration for timeout
        }
    
    def _invalid_cart_operation(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate invalid cart operations."""
        operations = [
            "add_invalid_product",
            "negative_quantity",
            "excessive_quantity",
            "empty_cart_checkout"
        ]
        
        operation = random.choice(operations)
        
        logger.info(f"User {journey.user_id} performing invalid cart operation: {operation}")
        
        return {
            "action": "invalid_cart_operation",
            "operation": operation,
            "expected_error": "400_bad_request",
            "duration": random.uniform(1, 4)
        }
    
    def _invalid_checkout_data(self, journey: UserJourney) -> Dict[str, Any]:
        """Simulate invalid checkout data submission."""
        invalid_data_types = [
            "invalid_email",
            "invalid_credit_card",
            "missing_address",
            "invalid_zip_code",
            "expired_card"
        ]
        
        data_type = random.choice(invalid_data_types)
        
        logger.info(f"User {journey.user_id} submitting invalid checkout data: {data_type}")
        
        return {
            "action": "invalid_checkout_data",
            "data_type": data_type,
            "expected_error": "422_validation_error",
            "duration": random.uniform(2, 8)
        }
    
    def simulate_user_session(self, duration_seconds: Optional[float] = None, user_id: Optional[str] = None) -> UserJourney:
        """
        Simulate a complete user session with realistic behavior patterns.
        
        Args:
            duration_seconds: Optional session duration override
            
        Returns:
            Completed UserJourney object with all behaviors and metrics
        """
        journey = self.create_user_session(user_id)
        
        if duration_seconds is None:
            duration_seconds = random.uniform(
                self.config.session_duration_min,
                self.config.session_duration_max
            )
        
        session_start = time.time()
        session_end = session_start + duration_seconds
        
        logger.info(f"Starting user session {journey.session_id} for {duration_seconds:.1f} seconds")
        
        behavior_results = []
        
        while time.time() < session_end:
            # Select next behavior based on traffic distribution and journey context
            behavior_type = self.select_next_behavior(journey)
            
            # Calculate realistic think time
            think_time = self.calculate_think_time(behavior_type, journey)
            
            logger.debug(f"User {journey.user_id} thinking for {think_time:.1f}s before {behavior_type.value}")
            time.sleep(think_time)
            
            # Execute the selected behavior
            try:
                if behavior_type == BehaviorType.BROWSE:
                    result = self.execute_browse_behavior(journey)
                elif behavior_type == BehaviorType.CART:
                    result = self.execute_cart_behavior(journey)
                elif behavior_type == BehaviorType.CHECKOUT:
                    result = self.execute_checkout_behavior(journey)
                elif behavior_type == BehaviorType.ERROR:
                    result = self.execute_error_behavior(journey)
                else:
                    logger.warning(f"Unknown behavior type: {behavior_type}")
                    continue
                
                behavior_results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing {behavior_type.value} behavior: {e}")
                journey.error_count += 1
            
            # Check if session should end early (e.g., after successful checkout)
            if journey.completed_checkout and random.random() < 0.3:
                logger.info(f"User {journey.user_id} ending session after successful checkout")
                break
        
        # Finalize journey
        journey.total_duration = time.time() - session_start
        
        with self.session_lock:
            if journey.session_id in self.active_sessions:
                del self.active_sessions[journey.session_id]
            self.completed_journeys.append(journey)
        
        logger.info(f"Completed user session {journey.session_id}: "
                   f"{len(journey.behaviors)} behaviors, "
                   f"{len(journey.products_viewed)} products viewed, "
                   f"{journey.error_count} errors, "
                   f"checkout: {journey.completed_checkout}")
        
        return journey
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all user sessions."""
        all_journeys = list(self.active_sessions.values()) + self.completed_journeys
        
        if not all_journeys:
            return {"total_sessions": 0}
        
        # Calculate behavior distribution
        behavior_counts = {behavior.value: 0 for behavior in BehaviorType}
        total_behaviors = 0
        
        for journey in all_journeys:
            for behavior in journey.behaviors:
                behavior_counts[behavior.value] += 1
                total_behaviors += 1
        
        # Calculate behavior percentages
        behavior_percentages = {}
        if total_behaviors > 0:
            for behavior, count in behavior_counts.items():
                behavior_percentages[behavior] = (count / total_behaviors) * 100
        
        # Calculate other metrics
        completed_sessions = len(self.completed_journeys)
        total_checkouts = sum(1 for j in all_journeys if j.completed_checkout)
        total_errors = sum(j.error_count for j in all_journeys)
        
        avg_session_duration = 0
        if self.completed_journeys:
            avg_session_duration = sum(j.total_duration for j in self.completed_journeys) / len(self.completed_journeys)
        
        return {
            "total_sessions": len(all_journeys),
            "active_sessions": len(self.active_sessions),
            "completed_sessions": completed_sessions,
            "total_behaviors": total_behaviors,
            "behavior_distribution": behavior_percentages,
            "target_distribution": {
                "browse": self.config.browse_percentage,
                "cart": self.config.cart_percentage,
                "checkout": self.config.checkout_percentage,
                "error": self.config.error_percentage
            },
            "checkout_completion_rate": (total_checkouts / completed_sessions * 100) if completed_sessions > 0 else 0,
            "total_errors": total_errors,
            "avg_session_duration": avg_session_duration,
            "avg_behaviors_per_session": total_behaviors / len(all_journeys) if all_journeys else 0
        }
    
    def export_journey_data(self, filepath: str) -> None:
        """Export all journey data to JSON file for analysis."""
        export_data = {
            "config": {
                "browse_percentage": self.config.browse_percentage,
                "cart_percentage": self.config.cart_percentage,
                "checkout_percentage": self.config.checkout_percentage,
                "error_percentage": self.config.error_percentage,
                "application_type": self.config.application_type.value,
                "session_duration_range": [self.config.session_duration_min, self.config.session_duration_max],
                "think_time_range": [self.config.min_think_time, self.config.max_think_time]
            },
            "statistics": self.get_session_statistics(),
            "journeys": []
        }
        
        # Export journey data
        for journey in self.completed_journeys:
            journey_data = {
                "journey_id": journey.journey_id,
                "user_id": journey.user_id,
                "session_id": journey.session_id,
                "start_time": journey.start_time.isoformat(),
                "total_duration": journey.total_duration,
                "behaviors": [b.value for b in journey.behaviors],
                "products_viewed": journey.products_viewed,
                "cart_items": journey.cart_items,
                "completed_checkout": journey.completed_checkout,
                "error_count": journey.error_count
            }
            export_data["journeys"].append(journey_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported journey data to {filepath}")


class BehaviorSimulatorFactory:
    """Factory class for creating behavior simulators with different configurations."""
    
    @staticmethod
    def create_default_simulator(app_type: ApplicationType = ApplicationType.OTEL_DEMO) -> UserBehaviorSimulator:
        """Create simulator with default e-commerce behavior configuration."""
        config = BehaviorConfig(application_type=app_type)
        return UserBehaviorSimulator(config)
    
    @staticmethod
    def create_mobile_simulator(app_type: ApplicationType = ApplicationType.OTEL_DEMO) -> UserBehaviorSimulator:
        """Create simulator optimized for mobile user behavior patterns."""
        config = BehaviorConfig(
            browse_percentage=80.0,  # Higher browsing on mobile
            cart_percentage=15.0,    # Lower cart conversion
            checkout_percentage=3.0,  # Much lower checkout completion
            error_percentage=2.0,
            min_think_time=0.5,      # Faster interactions
            max_think_time=5.0,
            session_duration_min=30.0,  # Shorter sessions
            session_duration_max=300.0,
            checkout_completion_rate=0.3,  # Lower completion rate
            application_type=app_type
        )
        return UserBehaviorSimulator(config)
    
    @staticmethod
    def create_high_value_simulator(app_type: ApplicationType = ApplicationType.OTEL_DEMO) -> UserBehaviorSimulator:
        """Create simulator for high-value customers with different behavior patterns."""
        config = BehaviorConfig(
            browse_percentage=50.0,   # Less browsing, more focused
            cart_percentage=30.0,     # Higher cart usage
            checkout_percentage=18.0, # Much higher checkout rate
            error_percentage=2.0,
            min_think_time=2.0,       # More deliberate
            max_think_time=15.0,
            session_duration_min=120.0,  # Longer sessions
            session_duration_max=900.0,
            checkout_completion_rate=0.8,  # Higher completion rate
            max_cart_items=10,        # Larger carts
            application_type=app_type
        )
        return UserBehaviorSimulator(config)
    
    @staticmethod
    def create_error_focused_simulator(app_type: ApplicationType = ApplicationType.OTEL_DEMO) -> UserBehaviorSimulator:
        """Create simulator focused on generating errors for testing."""
        config = BehaviorConfig(
            browse_percentage=40.0,
            cart_percentage=20.0,
            checkout_percentage=10.0,
            error_percentage=30.0,    # High error rate for testing
            min_think_time=0.1,       # Quick error generation
            max_think_time=2.0,
            session_duration_min=60.0,
            session_duration_max=300.0,
            error_scenarios_enabled=True,
            application_type=app_type
        )
        return UserBehaviorSimulator(config)


# Example usage and testing
if __name__ == "__main__":
    # Example: Create and run a default simulator
    simulator = BehaviorSimulatorFactory.create_default_simulator(ApplicationType.OTEL_DEMO)
    
    # Simulate a single user session
    journey = simulator.simulate_user_session(duration_seconds=120)
    
    # Print statistics
    stats = simulator.get_session_statistics()
    print(json.dumps(stats, indent=2))
    
    # Export data
    simulator.export_journey_data("user_behavior_data.json")