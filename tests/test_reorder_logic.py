"""Unit tests for inventory replenishment logic and periodic-review safety stock formulas.
Equations:
  T = L + R
  SS = z * sigma_d * sqrt(T)
  Q = max(0, ceil(D_T + SS - S - I))
  Stockout Alert: (S + I) < P90_L
"""

import math
import unittest


def calculate_reorder(
    lead_time: int,
    review_period: int,
    stock_on_hand: int,
    incoming_orders: int,
    service_level_z: float,
    forecast_demand_dt: float,
    sigma_d: float,
    p90_lead_time_demand: float | None = None,
) -> dict:
    T = lead_time + review_period
    exact_ss = service_level_z * sigma_d * math.sqrt(T)
    safety_stock = int(round(exact_ss))
    raw_q = forecast_demand_dt + exact_ss - stock_on_hand - incoming_orders
    order_quantity = max(0, int(math.ceil(raw_q)))

    if p90_lead_time_demand is None:
        avg_daily = forecast_demand_dt / max(1, T)
        p90_lead_time_demand = (avg_daily * lead_time) + (1.28 * sigma_d * math.sqrt(lead_time))

    stockout_risk = (stock_on_hand + incoming_orders) < p90_lead_time_demand

    return {
        "protection_interval": T,
        "exact_safety_stock": exact_ss,
        "safety_stock": safety_stock,
        "order_quantity": order_quantity,
        "stockout_risk": stockout_risk,
    }


class TestReorderLogic(unittest.TestCase):
    def test_standard_reorder_case(self):
        # L = 3, R = 7 -> T = 10. D_T = 120, sigma_d = 5.7, z = 1.65
        # SS = 1.65 * 5.7 * sqrt(10) = 29.74 -> 30
        # S = 40, I = 0
        # Q = max(0, ceil(120 + 29.74 - 40 - 0)) = 110
        result = calculate_reorder(
            lead_time=3,
            review_period=7,
            stock_on_hand=40,
            incoming_orders=0,
            service_level_z=1.65,
            forecast_demand_dt=120.0,
            sigma_d=5.7,
        )
        self.assertEqual(result["protection_interval"], 10)
        self.assertEqual(result["safety_stock"], 30)
        self.assertEqual(result["order_quantity"], 110)

    def test_overstocked_zero_order(self):
        result = calculate_reorder(
            lead_time=3,
            review_period=7,
            stock_on_hand=200,
            incoming_orders=50,
            service_level_z=1.65,
            forecast_demand_dt=120.0,
            sigma_d=5.7,
        )
        self.assertEqual(result["order_quantity"], 0)
        self.assertFalse(result["stockout_risk"])

    def test_critical_stockout_alert(self):
        result = calculate_reorder(
            lead_time=4,
            review_period=7,
            stock_on_hand=5,
            incoming_orders=5,
            service_level_z=1.65,
            forecast_demand_dt=150.0,
            sigma_d=8.0,
            p90_lead_time_demand=65.0,
        )
        self.assertTrue(result["stockout_risk"])
        self.assertGreater(result["order_quantity"], 0)

    def test_service_level_sensitivity(self):
        low_z = calculate_reorder(3, 7, 50, 0, 1.28, 100.0, 5.0)
        mid_z = calculate_reorder(3, 7, 50, 0, 1.65, 100.0, 5.0)
        high_z = calculate_reorder(3, 7, 50, 0, 2.33, 100.0, 5.0)

        self.assertLess(low_z["exact_safety_stock"], mid_z["exact_safety_stock"])
        self.assertLess(mid_z["exact_safety_stock"], high_z["exact_safety_stock"])
        self.assertLessEqual(low_z["order_quantity"], mid_z["order_quantity"])
        self.assertLessEqual(mid_z["order_quantity"], high_z["order_quantity"])


if __name__ == "__main__":
    unittest.main()
